# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
    convert_to_messages,
    convert_to_openai_messages,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import Command, Send

if TYPE_CHECKING:
    from langchain_core.messages import ToolCall
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.runtime import Runtime as LangGraphRuntime
from nemoguardrails.guardrails.guardrails import Guardrails
from nemoguardrails.guardrails.iorails import IORails, _rail_type_configured
from nemoguardrails.guardrails.model_engine import _ENGINE_BASE_URLS
from nemoguardrails.integrations.langchain.exceptions import GuardrailViolation
from nemoguardrails.integrations.langchain.message_utils import (
    create_ai_message,
    is_ai_message,
    is_human_message,
    messages_to_dicts,
)
from nemoguardrails.rails.llm.config import Rails, RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.options import RailsResult, RailStatus, RailType
from nemoguardrails.utils import get_or_create_event_loop

log = logging.getLogger(__name__)


def _iorails_can_build(model: Any) -> bool:
    return model.engine in _ENGINE_BASE_URLS or bool((model.parameters or {}).get("base_url"))


_DISABLE_TOOL_RAILS_HINT = "Pass enable_tool_rails=False to run the middleware without tool rails."


def _tool_rails_config(config: RailsConfig) -> RailsConfig:
    """Project *config* onto the tool rails, so IORails can serve them.

    The input and output rails stay on LLMRails, so their flows (custom Colang, Colang 2.x)
    do not decide whether IORails can serve the tool rails. Prompts, tracing and the other
    top-level settings carry over.

    Raises:
        ValueError: If the config has no ``main`` model, its ``main`` model is one IORails
            cannot build, or IORails cannot serve the projected tool rails.
    """
    main = next((model for model in config.models if model.type == "main"), None)
    if main is None:
        raise ValueError(
            f"Tool rails need a `main` model; IORails reads tool calls in its wire format. {_DISABLE_TOOL_RAILS_HINT}"
        )
    if not _iorails_can_build(main):
        raise ValueError(
            f"Tool rails need a `main` model with engine {sorted(_ENGINE_BASE_URLS)} or a "
            f"`parameters.base_url`; got engine '{main.engine}'. {_DISABLE_TOOL_RAILS_HINT}"
        )

    rail_sections = ("config", RailType.TOOL_CALL.value, RailType.TOOL_RESULT.value)
    projection = config.model_copy(
        deep=True,
        update={
            # IORails builds a client for every model and raises on one it cannot build,
            # such as a FastEmbed embeddings model the input rails use.
            "models": [model for model in config.models if _iorails_can_build(model)],
            "rails": Rails(**{section: getattr(config.rails, section) for section in rail_sections}),
            "flows": [],
            "colang_version": "1.0",
        },
    )
    reason = IORails.unsupported_reason(projection)
    if reason is not None:
        raise ValueError(f"Tool rails cannot run on IORails: {reason}. {_DISABLE_TOOL_RAILS_HINT}")
    return projection


def _event_loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class _LoopBoundGuardrails:
    """Runs tool-rail checks for one tool-only config on IORails.

    A started IORails engine serves only the event loop it started on, so async checks get
    one engine per running loop, and engines of closed loops are dropped.
    """

    def __init__(self, config: RailsConfig):
        self._config = config
        self._engines: Dict[asyncio.AbstractEventLoop, Guardrails] = {}
        self._lock = threading.Lock()
        # Never started: its sync check runs a short-lived engine per call.
        self._sync_engine = Guardrails(config, require_iorails=True)

    def _engine(self) -> Guardrails:
        loop = asyncio.get_running_loop()
        with self._lock:
            for closed in [other for other in self._engines if other.is_closed()]:
                del self._engines[closed]
            engine = self._engines.get(loop)
            if engine is None:
                engine = self._engines[loop] = Guardrails(self._config, require_iorails=True)
            return engine

    async def check_async(
        self, messages: List[dict], rail_type: RailType, tools: Optional[List[dict]] = None
    ) -> RailsResult:
        return await self._engine().check_async(messages, rail_types=[rail_type], tools=tools)

    def check(self, messages: List[dict], rail_type: RailType, tools: Optional[List[dict]] = None) -> RailsResult:
        def run() -> RailsResult:
            return self._sync_engine.check(messages, rail_types=[rail_type], tools=tools)

        if not _event_loop_running():
            return run()
        # The sync check starts its own event loop, which cannot run inside a running one.
        context = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(context.run, run).result()

    async def aclose(self) -> None:
        loop = asyncio.get_running_loop()
        with self._lock:
            engine = self._engines.pop(loop, None)
        if engine is not None:
            await engine.shutdown()


class _UnsafeToolResult(Exception):
    """A tool returned messages the tool-result rail cannot vouch for."""


@dataclass
class _ToolResultReview:
    """A tool's return, with re-emitted history swapped for the originals from state.

    ``terminator`` is the tool message to check, if any; ``terminator_entry`` is the same
    message as it appears in ``result`` (an object or a dict). ``message_keys`` are the dict
    keys whose message lists the review walked in the return's updates and ``Send`` arguments.
    """

    result: Any
    terminator: Optional[ToolMessage] = None
    terminator_entry: Any = field(default=None, repr=False)
    message_keys: Tuple[str, ...] = ("messages",)


class _UnusableRewrite(Exception):
    """A tool rail rewrite the middleware cannot apply to the payload it checked."""


def _rewritten_messages(outcome: RailsResult, sent: List[dict]) -> List[dict]:
    messages = outcome.messages
    if messages is None:
        raise _UnusableRewrite("the check returned no rewritten messages")
    if len(messages) != len(sent):
        raise _UnusableRewrite(f"the check returned {len(messages)} messages for the {len(sent)} it was sent")
    return messages


def _rewritten_tool_args(outcome: RailsResult, sent: List[dict], tool_call: ToolCall) -> Dict[str, Any]:
    """The rewritten arguments of *tool_call*, read from the latest assistant message the check returned.

    Raises:
        _UnusableRewrite: If that message does not hold this one call, or its arguments are not a JSON object.
    """
    messages = _rewritten_messages(outcome, sent)
    assistant = next((m for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "assistant"), {})
    calls = assistant.get("tool_calls") or []
    if len(calls) != 1 or not isinstance(calls[0], dict):
        raise _UnusableRewrite("the rewritten conversation does not hold exactly one tool call")
    function = calls[0].get("function") or {}
    if calls[0].get("id") != tool_call["id"] or function.get("name") != tool_call["name"]:
        raise _UnusableRewrite("the rewritten tool call names another call or tool")
    try:
        args = json.loads(function.get("arguments"))
    except (TypeError, ValueError) as e:
        raise _UnusableRewrite("the rewritten arguments are not JSON") from e
    if not isinstance(args, dict):
        raise _UnusableRewrite(f"the rewritten arguments are a JSON {type(args).__name__}, not an object")
    return args


def _rewritten_tool_content(outcome: RailsResult, sent: List[dict]) -> str:
    """The rewritten content of the checked tool message, the last of the messages sent.

    Raises:
        _UnusableRewrite: If the message at that index is not the same tool message with text content.
    """
    message = _rewritten_messages(outcome, sent)[-1]
    if (
        not isinstance(message, dict)
        or message.get("role") != "tool"
        or message.get("tool_call_id") != sent[-1].get("tool_call_id")
    ):
        raise _UnusableRewrite("the rewritten conversation does not end with the checked tool message")
    content = message.get("content")
    if not isinstance(content, str):
        raise _UnusableRewrite(f"the rewritten content is a {type(content).__name__}, not text")
    return content


def _as_message(entry: Any) -> BaseMessage:
    if isinstance(entry, BaseMessage):
        return entry
    try:
        return convert_to_messages([entry])[0]
    except Exception as e:
        raise _UnsafeToolResult("a message that cannot be read") from e


def _is_tool_call(entry: Any) -> bool:
    """Whether an entry is a tool call rather than a message."""
    if not isinstance(entry, dict):
        return False
    if entry.get("type") == "tool_call":
        return True
    return "name" in entry and "args" in entry and "role" not in entry


def _map_messages(entries: Any, transform: Callable[[Any], Any]) -> Any:
    if not isinstance(entries, list):
        raise _UnsafeToolResult(f"messages of type {type(entries).__name__}")
    mapped = [transform(entry) for entry in entries]
    return entries if all(new is old for new, old in zip(mapped, entries)) else mapped


def _map_message_fields(fields: dict, keys: Tuple[str, ...], transform: Callable[[Any], Any]) -> dict:
    changed = {}
    for key in keys:
        if key in fields:
            mapped = _map_messages(fields[key], transform)
            if mapped is not fields[key]:
                changed[key] = mapped
    return {**fields, **changed} if changed else fields


def _map_send(send: Send, keys: Tuple[str, ...], transform: Callable[[Any], Any]) -> Send:
    arg = send.arg
    if isinstance(arg, dict):
        mapped = _map_message_fields(arg, keys, transform)
    elif isinstance(arg, list):
        # A list of tool calls, as create_agent sends to its tool node, holds no messages.
        # Anything else is vetted as a conversation: the message shorthands LangChain accepts
        # are too many to recognize one by one, and an unrecognized one must not reach a model.
        mapped = arg if all(_is_tool_call(entry) for entry in arg) else _map_messages(arg, transform)
    elif arg is None or isinstance(arg, (str, int, float, bool)):
        mapped = arg
    else:
        # Such as a state model, whose messages cannot be rewritten in place.
        raise _UnsafeToolResult(f"a Send with an argument of type {type(arg).__name__}")
    return send if mapped is arg else Send(send.node, mapped)


def _map_goto(goto: Any, keys: Tuple[str, ...], transform: Callable[[Any], Any]) -> Any:
    if isinstance(goto, Send):
        return _map_send(goto, keys, transform)
    if isinstance(goto, (list, tuple)):
        mapped = [_map_send(item, keys, transform) if isinstance(item, Send) else item for item in goto]
        return goto if all(new is old for new, old in zip(mapped, goto)) else type(goto)(mapped)
    return goto


def _map_command(command: Command, transform: Callable[[Any], Any], keys: Tuple[str, ...]) -> Command:
    update = command.update
    if update is None:
        mapped_update = None
    elif isinstance(update, list):
        mapped_update = _map_messages(update, transform)
    elif isinstance(update, dict):
        mapped_update = _map_message_fields(update, keys, transform)
    else:
        raise _UnsafeToolResult(f"a Command update of type {type(update).__name__}")
    goto = _map_goto(command.goto, keys, transform)
    if mapped_update is update and goto is command.goto:
        return command
    return replace(command, update=mapped_update, goto=goto)


def _map_tool_result(result: Any, transform: Callable[[Any], Any], keys: Tuple[str, ...] = ("messages",)) -> Any:
    """Rebuild a tool's return with *transform* applied to every message it adds to the conversation.

    Accepts the returns LangGraph's ``ToolNode`` accepts: a ``ToolMessage``, a ``Command``, or a
    list of those. In a ``Command``, the messages are a list ``update``, or the lists under *keys*
    in a dict ``update`` and in the arguments of the ``Send`` objects in ``goto``. The container
    keeps its shape; a part *transform* leaves unchanged keeps its identity.

    Raises:
        _UnsafeToolResult: On any other shape.
    """
    if isinstance(result, ToolMessage):
        return transform(result)
    if isinstance(result, Command):
        return _map_command(result, transform, keys)
    if isinstance(result, list) and all(isinstance(item, (Command, ToolMessage)) for item in result):
        mapped = [_map_tool_result(item, transform, keys) for item in result]
        return result if all(new is old for new, old in zip(mapped, result)) else mapped
    raise _UnsafeToolResult(f"a tool return of type {type(result).__name__}")


def _made_call(message: Any, call_id: str) -> bool:
    return isinstance(message, AIMessage) and any(call.get("id") == call_id for call in message.tool_calls)


def _state_fields(state: Any) -> Dict[str, Any]:
    """The named fields of a graph state, whichever way the state class stores them.

    A state declared with ``__slots__`` (a slotted dataclass, say) has no ``__dict__``, and
    reading no fields would hide the conversation and leave the messages in it unchecked.
    """
    if isinstance(state, dict):
        return state
    fields = getattr(state, "__dict__", None)
    if fields:
        return dict(fields)
    names: List[str] = []
    for klass in type(state).__mro__:
        slots = getattr(klass, "__slots__", ())
        names.extend([slots] if isinstance(slots, str) else slots)
    return {name: getattr(state, name) for name in dict.fromkeys(names) if hasattr(state, name)}


def _conversation(state: Any, call_id: str) -> Tuple[Optional[str], Any]:
    """The key and message list of the conversation in *state*: the list with the AI message that made the call.

    A list state is the conversation itself, with no key. A ``ToolNode`` with a custom
    ``messages_key`` keeps the conversation under that key. Without a match it is ``messages``.
    """
    if isinstance(state, list):
        return None, state
    fields = _state_fields(state)
    for key in sorted(fields, key=lambda name: name != "messages"):
        value = fields[key]
        if isinstance(value, list) and any(_made_call(message, call_id) for message in value):
            return key, value
    return "messages", fields.get("messages")


def _messages_by_id(messages: Any) -> Dict[str, BaseMessage]:
    if not isinstance(messages, list):
        return {}
    return {message.id: message for message in messages if isinstance(message, BaseMessage) and message.id}


def _review_tool_result(request: ToolCallRequest, result: Any) -> _ToolResultReview:
    """Find the tool message to check in *result* and vet every other message it adds.

    Only the message answering this call can be checked: the rail links a result to its call
    by id. Any other message passes only when it re-emits conversation history unchanged
    (handoff tools do), and the original from state is emitted in its place, because the
    message reducer replaces history by id and would otherwise accept an edited copy.

    Raises:
        _UnsafeToolResult: If *result* adds anything else.
    """
    call_id = request.tool_call["id"]
    key, conversation = _conversation(request.state, call_id)
    history = _messages_by_id(conversation)
    review = _ToolResultReview(
        result=None, message_keys=tuple(dict.fromkeys(name for name in ("messages", key) if name))
    )

    def vet(entry: Any) -> Any:
        message = _as_message(entry)
        if isinstance(message, RemoveMessage):
            return entry
        if isinstance(message, ToolMessage) and message.tool_call_id == call_id:
            if review.terminator is not None:
                raise _UnsafeToolResult("more than one tool message for this call")
            review.terminator, review.terminator_entry = message, entry
            return entry
        original = history.get(message.id) if message.id else None
        if original is None or type(original) is not type(message) or original != message:
            raise _UnsafeToolResult(f"a {type(message).__name__} that is not unchanged conversation history")
        return original

    review.result = _map_tool_result(result, vet, review.message_keys)
    return review


def _replace_terminator(review: _ToolResultReview, **fields: Any) -> Any:
    """Rebuild the reviewed return with *fields* set on the checked tool message and its artifact dropped.

    The artifact holds the tool's raw output, which a rewritten result must not carry on.
    A tool message the tool returned as a dict comes back as a ``ToolMessage``, so no dict form,
    such as a serialized message's ``kwargs``, carries the original content on.
    """
    replacement = review.terminator.model_copy(update={**fields, "artifact": None})
    entry = review.terminator_entry
    return _map_tool_result(
        review.result, lambda message: replacement if message is entry else message, review.message_keys
    )


class GuardrailsMiddleware(AgentMiddleware):
    def __init__(
        self,
        config_path: Optional[str] = None,
        config_yaml: Optional[str] = None,
        raise_on_violation: bool = False,
        blocked_input_message: str = "I cannot process this request due to content policy.",
        blocked_output_message: str = "I cannot provide this response due to content policy.",
        enable_input_rails: bool = True,
        enable_output_rails: bool = True,
        enable_tool_rails: bool = True,
        blocked_tool_call_message: str = (
            "Tool call blocked due to content policy. The tool was not executed; do not retry this call."
        ),
        blocked_tool_result_message: str = "Tool result withheld due to content policy.",
    ):
        if config_path is not None:
            config = RailsConfig.from_path(config_path)
        elif config_yaml is not None:
            config = RailsConfig.from_content(yaml_content=config_yaml)
        else:
            raise ValueError("Either 'config_path' or 'config_yaml' must be provided to GuardrailsMiddleware")

        self._tool_call_rails_on = enable_tool_rails and _rail_type_configured(config, RailType.TOOL_CALL)
        self._tool_result_rails_on = enable_tool_rails and _rail_type_configured(config, RailType.TOOL_RESULT)
        # Input and output rails stay on LLMRails; tool rails need IORails, which cannot serve
        # the custom Colang flows most input and output rails use.
        self._tool_rails: Optional[_LoopBoundGuardrails] = None
        if self._tool_call_rails_on or self._tool_result_rails_on:
            self._tool_rails = _LoopBoundGuardrails(_tool_rails_config(config))

        self.rails = LLMRails(config=config)
        self.raise_on_violation = raise_on_violation
        self.blocked_input_message = blocked_input_message
        self.blocked_output_message = blocked_output_message
        self.enable_input_rails = enable_input_rails
        self.enable_output_rails = enable_output_rails
        self.enable_tool_rails = enable_tool_rails
        self.blocked_tool_call_message = blocked_tool_call_message
        self.blocked_tool_result_message = blocked_tool_result_message

    def _has_input_rails(self) -> bool:
        return len(self.rails.config.rails.input.flows) > 0

    def _has_output_rails(self) -> bool:
        return len(self.rails.config.rails.output.flows) > 0

    def _convert_to_rails_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        return messages_to_dicts(messages)

    def _get_last_user_message(self, messages: List[BaseMessage]) -> Optional[HumanMessage]:
        for msg in reversed(messages):
            if is_human_message(msg):
                return msg
        return None

    def _get_last_ai_message(self, messages: List[BaseMessage]) -> Optional[AIMessage]:
        for msg in reversed(messages):
            if is_ai_message(msg):
                return msg
        return None

    def _handle_guardrail_failure(
        self,
        result: RailsResult,
        rail_type: str,
        blocked_message: str,
    ) -> None:
        if result.status == RailStatus.BLOCKED:
            failure_message = f"{rail_type.capitalize()} blocked by {result.rail or 'unknown rail'}"

            if self.raise_on_violation:
                raise GuardrailViolation(
                    message=failure_message,
                    result=result,
                    rail_type=rail_type,
                )

            log.warning(failure_message)

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        if not self.enable_input_rails or not self._has_input_rails():
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last_user_message = self._get_last_user_message(messages)
        if not last_user_message:
            return None

        rails_messages = self._convert_to_rails_messages(messages)

        try:
            result = await self.rails.check_async(rails_messages, rail_types=[RailType.INPUT])

            if result.status == RailStatus.BLOCKED:
                self._handle_guardrail_failure(
                    result=result,
                    rail_type="input",
                    blocked_message=self.blocked_input_message,
                )
                blocked_msg = create_ai_message(self.blocked_input_message)
                return {"messages": messages + [blocked_msg], "jump_to": "end"}

            if result.status == RailStatus.MODIFIED:
                log.info("Input modified by rail '%s': content replaced", result.rail or "unknown rail")
                modified_msg = last_user_message.model_copy(update={"content": result.content})
                return {"messages": self._replace_last_human_message(messages, modified_msg)}

            return None

        except GuardrailViolation:
            raise
        except Exception as e:
            log.error(f"Error checking input rails: {e}", exc_info=True)

            if self.raise_on_violation:
                raise GuardrailViolation(
                    message=f"Input rail execution error: {str(e)}",
                    rail_type="input",
                )

            blocked_msg = create_ai_message(self.blocked_input_message)
            return {"messages": messages + [blocked_msg], "jump_to": "end"}

    def _replace_last_human_message(self, messages: list, replacement: HumanMessage) -> list:
        for i in range(len(messages) - 1, -1, -1):
            if is_human_message(messages[i]):
                return messages[:i] + [replacement] + messages[i + 1 :]
        return messages + [replacement]

    def _replace_last_ai_message(self, messages: list, replacement: AIMessage) -> list:
        for i in range(len(messages) - 1, -1, -1):
            if is_ai_message(messages[i]):
                return messages[:i] + [replacement] + messages[i + 1 :]
        return messages + [replacement]

    async def aafter_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        if not self.enable_output_rails or not self._has_output_rails():
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last_ai_message = self._get_last_ai_message(messages)
        if not last_ai_message:
            return None

        rails_messages = self._convert_to_rails_messages(messages)

        try:
            result = await self.rails.check_async(rails_messages, rail_types=[RailType.OUTPUT])

            if result.status == RailStatus.BLOCKED:
                self._handle_guardrail_failure(
                    result=result,
                    rail_type="output",
                    blocked_message=self.blocked_output_message,
                )
                blocked_msg = create_ai_message(self.blocked_output_message)
                return {"messages": self._replace_last_ai_message(messages, blocked_msg)}

            if result.status == RailStatus.MODIFIED:
                log.info("Output modified by rail '%s': content replaced", result.rail or "unknown rail")
                modified_msg = last_ai_message.model_copy(update={"content": result.content})
                return {"messages": self._replace_last_ai_message(messages, modified_msg)}

            return None

        except GuardrailViolation:
            raise
        except Exception as e:
            log.error(f"Error checking output rails: {e}", exc_info=True)

            if self.raise_on_violation:
                raise GuardrailViolation(
                    message=f"Output rail execution error: {str(e)}",
                    rail_type="output",
                )

            blocked_msg = create_ai_message(self.blocked_output_message)
            return {"messages": self._replace_last_ai_message(messages, blocked_msg)}

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        if not self.enable_input_rails or not self._has_input_rails():
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        loop = get_or_create_event_loop()
        return loop.run_until_complete(self.abefore_model(state, runtime))

    def after_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        if not self.enable_output_rails or not self._has_output_rails():
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last_ai_message = self._get_last_ai_message(messages)
        if not last_ai_message:
            return None

        loop = get_or_create_event_loop()
        return loop.run_until_complete(self.aafter_model(state, runtime))

    def _tool_call_payload(self, request: ToolCallRequest) -> tuple[List[dict], List[dict]]:
        messages = convert_to_openai_messages([AIMessage(content="", tool_calls=[request.tool_call])])
        tools = [convert_to_openai_tool(request.tool)] if request.tool is not None else []
        return messages, tools

    def _tool_result_payload(self, request: ToolCallRequest, tool_message: ToolMessage) -> List[dict]:
        # Only this call and its result: a tool_result check re-checks every tool message it is given.
        return convert_to_openai_messages([AIMessage(content="", tool_calls=[request.tool_call]), tool_message])

    def _tool_check_passed(self, outcome: Union[RailsResult, Exception], rail_type: str) -> bool:
        """Whether a tool check lets the call or result through, rewritten or not.

        Raises ``GuardrailViolation`` instead of returning False when ``raise_on_violation``.
        """
        label = rail_type.replace("_", " ").capitalize()
        if isinstance(outcome, Exception):
            log.error("Error checking %s rails: %s", rail_type.replace("_", " "), outcome, exc_info=outcome)
            if self.raise_on_violation:
                raise GuardrailViolation(
                    message=f"{label} rail execution error: {outcome}",
                    rail_type=rail_type,
                ) from outcome
            return False

        if outcome.status in (RailStatus.PASSED, RailStatus.MODIFIED):
            return True

        failure_message = f"{label} blocked by {outcome.rail or 'unknown rail'}"
        # The reason goes to the log only: the model reads the blocked message, and an injected
        # instruction could use the reason to work around the rail.
        log.warning("%s: %s", failure_message, outcome.reason)
        if self.raise_on_violation:
            raise GuardrailViolation(message=failure_message, result=outcome, rail_type=rail_type)
        return False

    def _reject_rewrite(self, outcome: RailsResult, rail_type: str, problem: _UnusableRewrite) -> None:
        """Log a rewrite that cannot be applied, or raise ``GuardrailViolation`` when ``raise_on_violation``."""
        failure_message = (
            f"{rail_type.replace('_', ' ').capitalize()} blocked: its rewrite cannot be applied, {problem}"
        )
        if self.raise_on_violation:
            raise GuardrailViolation(message=failure_message, result=outcome, rail_type=rail_type) from problem
        log.warning(failure_message)

    def _checked_tool_call(
        self, request: ToolCallRequest, sent: List[dict], outcome: Union[RailsResult, Exception]
    ) -> Optional[ToolCallRequest]:
        """The request to run after the tool call check, with any rewritten arguments, or None to block it."""
        if not self._tool_check_passed(outcome, "tool_call"):
            return None
        if outcome.status != RailStatus.MODIFIED:
            return request
        try:
            args = _rewritten_tool_args(outcome, sent, request.tool_call)
        except _UnusableRewrite as e:
            self._reject_rewrite(outcome, "tool_call", e)
            return None
        log.info("Tool call arguments of '%s' rewritten by the tool call rails", request.tool_call["name"])
        return request.override(tool_call={**request.tool_call, "args": args})

    def _checked_tool_result(
        self, review: _ToolResultReview, sent: List[dict], outcome: Union[RailsResult, Exception]
    ) -> Any:
        """The tool's return after the tool result check: unchanged, with the rewritten content, or withheld."""
        if not self._tool_check_passed(outcome, "tool_result"):
            return self._withhold_tool_result(review)
        if outcome.status != RailStatus.MODIFIED:
            return review.result
        try:
            content = _rewritten_tool_content(outcome, sent)
        except _UnusableRewrite as e:
            self._reject_rewrite(outcome, "tool_result", e)
            return self._withhold_tool_result(review)
        log.info("Tool result of '%s' rewritten by the tool result rails", review.terminator.name)
        return _replace_terminator(review, content=content)

    def _blocked_tool_call(self, request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(
            content=self.blocked_tool_call_message,
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )

    def _withhold_tool_result(self, review: _ToolResultReview) -> ToolMessage:
        """The blocked tool message alone, so nothing else the tool returned reaches the graph.

        A ``Command``'s other state updates, its ``goto`` and its ``Send`` arguments are dropped
        with it: the rail judged this result unsafe, and they can carry the same content on.
        """
        return review.terminator.model_copy(
            update={"content": self.blocked_tool_result_message, "status": "error", "artifact": None}
        )

    def _review_tool_result(self, request: ToolCallRequest, result: Any) -> _ToolResultReview:
        """Review a tool's return; one that adds unvetted messages is replaced by one blocked tool message."""
        try:
            return _review_tool_result(request, result)
        except _UnsafeToolResult as e:
            failure_message = f"Tool result blocked: the tool returned {e}"
            if self.raise_on_violation:
                raise GuardrailViolation(message=failure_message, rail_type="tool_result") from e
            log.warning(failure_message)
            return _ToolResultReview(
                result=ToolMessage(
                    content=self.blocked_tool_result_message,
                    tool_call_id=request.tool_call["id"],
                    name=request.tool_call["name"],
                    status="error",
                )
            )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Union[ToolMessage, Command]]],
    ) -> Union[ToolMessage, Command]:
        if self._tool_call_rails_on:
            sent: List[dict] = []
            try:
                sent, tools = self._tool_call_payload(request)
                outcome = await self._tool_rails.check_async(sent, RailType.TOOL_CALL, tools)
            except Exception as e:
                outcome = e
            checked = self._checked_tool_call(request, sent, outcome)
            if checked is None:
                return self._blocked_tool_call(request)
            request = checked

        result = await handler(request)
        if not self._tool_result_rails_on:
            return result

        review = self._review_tool_result(request, result)
        if review.terminator is None:
            return review.result
        sent = []
        try:
            sent = self._tool_result_payload(request, review.terminator)
            outcome = await self._tool_rails.check_async(sent, RailType.TOOL_RESULT)
        except Exception as e:
            outcome = e
        return self._checked_tool_result(review, sent, outcome)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Union[ToolMessage, Command]],
    ) -> Union[ToolMessage, Command]:
        if self._tool_call_rails_on:
            sent: List[dict] = []
            try:
                sent, tools = self._tool_call_payload(request)
                outcome = self._tool_rails.check(sent, RailType.TOOL_CALL, tools)
            except Exception as e:
                outcome = e
            checked = self._checked_tool_call(request, sent, outcome)
            if checked is None:
                return self._blocked_tool_call(request)
            request = checked

        result = handler(request)
        if not self._tool_result_rails_on:
            return result

        review = self._review_tool_result(request, result)
        if review.terminator is None:
            return review.result
        sent = []
        try:
            sent = self._tool_result_payload(request, review.terminator)
            outcome = self._tool_rails.check(sent, RailType.TOOL_RESULT)
        except Exception as e:
            outcome = e
        return self._checked_tool_result(review, sent, outcome)

    async def aclose(self) -> None:
        """Shut down the tool-rails engine bound to the running event loop.

        Call it before that loop closes, for example at the end of the coroutine passed to
        ``asyncio.run``, to release the engine's connections.
        """
        if self._tool_rails is not None:
            await self._tool_rails.aclose()


class InputRailsMiddleware(GuardrailsMiddleware):
    # create_agent adds a ToolNode for any middleware class that overrides a tool hook.
    wrap_tool_call = AgentMiddleware.wrap_tool_call
    awrap_tool_call = AgentMiddleware.awrap_tool_call

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_yaml: Optional[str] = None,
        raise_on_violation: bool = False,
        blocked_input_message: str = "I cannot process this request due to content policy.",
    ):
        super().__init__(
            config_path=config_path,
            config_yaml=config_yaml,
            raise_on_violation=raise_on_violation,
            blocked_input_message=blocked_input_message,
            blocked_output_message="",
            enable_input_rails=True,
            enable_output_rails=False,
            enable_tool_rails=False,
        )

    async def aafter_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        return None

    def after_agent(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        return None


class OutputRailsMiddleware(GuardrailsMiddleware):
    # create_agent adds a ToolNode for any middleware class that overrides a tool hook.
    wrap_tool_call = AgentMiddleware.wrap_tool_call
    awrap_tool_call = AgentMiddleware.awrap_tool_call

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_yaml: Optional[str] = None,
        raise_on_violation: bool = False,
        blocked_output_message: str = "I cannot provide this response due to content policy.",
    ):
        super().__init__(
            config_path=config_path,
            config_yaml=config_yaml,
            raise_on_violation=raise_on_violation,
            blocked_input_message="",
            blocked_output_message=blocked_output_message,
            enable_input_rails=False,
            enable_output_rails=True,
            enable_tool_rails=False,
        )

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        return None

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        return None

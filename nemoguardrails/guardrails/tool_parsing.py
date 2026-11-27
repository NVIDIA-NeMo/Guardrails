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

"""Normalize provider tool definitions and conversation turns for shared rail checks.

OpenAI and NIM use the Chat Completions wire shape. Unregistered providers retain
the same OpenAI-compatible fallback as the model adapter. Parsing requires no
model instance, so checks on caller-supplied messages also use this boundary.
"""

from typing import Optional, cast

from nemoguardrails.guardrails.guardrails_types import LLMMessages, quoted_identity
from nemoguardrails.guardrails.tool_schema import (
    LatestToolCall,
    Tool,
    ToolCallExtractionError,
    ToolExchange,
    ToolResult,
    Toolset,
)
from nemoguardrails.rails.llm.options import ToolViolationType
from nemoguardrails.types import ChatMessage, ToolCall, ToolCallFunction


def parse_tool_definitions(tools: list) -> list[Tool]:
    """Parse OpenAI Chat Completions tool definitions into ``Tool`` objects.

    Each entry has the nested shape ``{"type": "function", "function": {"name",
    "description", "parameters", "strict"}}``; ``function.parameters`` (the JSON
    Schema) maps to ``Tool.arguments_schema``. Entries that are not a dict, lack a
    ``function`` block, or whose function has no non-empty ``name`` are skipped.
    """
    parsed: list[Tool] = []
    for entry in tools:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        parsed.append(
            Tool(
                name=name,
                type=entry.get("type", "function"),
                description=function.get("description"),
                arguments_schema=function.get("parameters"),
                strict=function.get("strict"),
            )
        )
    return parsed


def _tool_result_from_message(message: dict, message_index: int) -> ToolResult:
    """Normalize one OpenAI Chat Completions ``role:"tool"`` message into a ``ToolResult``.

    This shape has no error flag, so ``is_error`` is always ``False``.
    """
    return ToolResult(
        call_id=message.get("tool_call_id"),
        name=message.get("name"),
        content=message.get("content"),
        message_index=message_index,
    )


def _tool_calls_from_message(message: dict) -> list[ToolCall]:
    """Extract the tool calls from one assistant message into ``ToolCall`` objects.
    Malformed tool calls fall back to just id, type, function"""
    try:
        return ChatMessage.from_dict(message).tool_calls or []
    except ValueError:
        return [
            ToolCall(
                id=raw.get("id", ""),
                type=raw.get("type", "function"),
                function=ToolCallFunction(name=(raw.get("function") or {}).get("name", ""), arguments={}),
            )
            for raw in message.get("tool_calls") or []
            if isinstance(raw, dict)
        ]


def extract_tool_exchanges(messages: LLMMessages) -> list[ToolExchange]:
    """Group an OpenAI Chat Completions conversation into per-turn ``ToolExchange``es."""
    exchanges: list[ToolExchange] = []
    # The exchange currently accepting tool results: the most recent assistant tool-call
    # turn, or an orphan opened by a leading/stray tool result. Any other message resets
    # it to None so the next tool result starts a fresh exchange instead of attaching to
    # a closed turn.
    open_exchange: ToolExchange | None = None
    # Every message counts toward the index, so a violation points into the caller's own list.
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "tool":
            if open_exchange is None:
                open_exchange = ToolExchange(calls=[], results=[])
                exchanges.append(open_exchange)
            open_exchange.results.append(_tool_result_from_message(message, message_index))
        elif role == "assistant" and message.get("tool_calls"):
            open_exchange = ToolExchange(calls=_tool_calls_from_message(message), results=[])
            exchanges.append(open_exchange)
        else:
            open_exchange = None
    return exchanges


def _latest_assistant_message(messages: LLMMessages) -> dict | None:
    """The last assistant message in *messages*, or None when there is none."""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return message
    return None


def _string_or_none(value: object) -> str | None:
    """*value* when it is a string, else None, so an error's identity fields hold only strings."""
    return value if isinstance(value, str) else None


def _describe_tool_call(index: int, tool_call_id: str | None) -> str:
    """Name a tool call in an error message by its id, else by its position."""
    if tool_call_id:
        return f"tool call '{quoted_identity(tool_call_id)}'"
    return f"tool call at index {index}"


def _legacy_function_call_error(function_call: object) -> ToolCallExtractionError:
    """The error for a legacy ``function_call``, which no tool rail validates."""
    name = function_call.get("name") if isinstance(function_call, dict) else None
    return ToolCallExtractionError(
        "legacy function_call is not validated by tool rails",
        violation_type=ToolViolationType.LEGACY_FUNCTION_CALL,
        tool_name=_string_or_none(name),
    )


def _malformed_tool_call_field(entry: dict, function: dict) -> str | None:
    """The first identity field of a wire tool call that is not a string, or None when all of them are."""
    tool_call_id = entry.get("id")
    if tool_call_id is not None and not isinstance(tool_call_id, str):
        return "id"
    if not isinstance(entry.get("type", "function"), str):
        return "type"
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return "function name"
    return None


def _parse_latest_tool_call(entry: object, index: int) -> ToolCall:
    """Strictly parse one wire tool call, raising a ``ToolCallExtractionError`` that omits its arguments."""
    if not isinstance(entry, dict):
        raise ToolCallExtractionError(
            f"tool call at index {index} is not an object",
            violation_type=ToolViolationType.MALFORMED_TOOL_CALL,
            index=index,
        )
    tool_call_id = _string_or_none(entry.get("id"))
    function = entry.get("function")
    # A call without a function object is not the Chat Completions shape; ChatMessage.from_dict would
    # read it as the legacy flat shape, which ignores an `arguments` key and validates `{}` instead.
    if not isinstance(function, dict):
        raise ToolCallExtractionError(
            f"{_describe_tool_call(index, tool_call_id)} does not carry a function object",
            violation_type=ToolViolationType.MALFORMED_TOOL_CALL,
            index=index,
            tool_call_id=tool_call_id,
        )
    malformed_field = _malformed_tool_call_field(entry, function)
    if malformed_field is not None:
        # ChatMessage.from_dict does not type these, and a non-string name breaks the per-tool rail lookup.
        raise ToolCallExtractionError(
            f"{_describe_tool_call(index, tool_call_id)} has a malformed {malformed_field}",
            violation_type=ToolViolationType.MALFORMED_TOOL_CALL,
            index=index,
            tool_call_id=tool_call_id,
            tool_name=_string_or_none(function.get("name")) or None,
        )
    # A completed turn's "" arguments are not JSON, so from_dict rejects them, as non-streaming generation does.
    try:
        message = ChatMessage.from_dict({"role": "assistant", "tool_calls": [entry]})
    except ValueError:
        # from_dict's message quotes the raw arguments, so it is dropped rather than chained.
        raise ToolCallExtractionError(
            f"{_describe_tool_call(index, tool_call_id)} has malformed arguments",
            violation_type=ToolViolationType.MALFORMED_ARGUMENTS,
            index=index,
            tool_call_id=tool_call_id,
            tool_name=_string_or_none(function.get("name")),
        ) from None
    return cast(list[ToolCall], message.tool_calls)[0]


def _parsed_or_error(entry: object, index: int) -> LatestToolCall:
    """The parsed call, or the error that keeps it from being validated, so one bad call doesn't stop the rest."""
    try:
        return _parse_latest_tool_call(entry, index)
    except ToolCallExtractionError as error:
        return error


def extract_latest_tool_calls(messages: LLMMessages) -> list[LatestToolCall]:
    """Strictly parse the last assistant message's tool calls; an unparsable call's error takes its place."""
    # Only that message is parsed: ChatMessage rejects roles such as the server's "context".
    message = _latest_assistant_message(messages)
    if message is None:
        return []
    # Checked first: a function_call beside tool_calls would otherwise go unvalidated.
    if message.get("function_call"):
        raise _legacy_function_call_error(message["function_call"])
    raw_calls = message.get("tool_calls")
    if not raw_calls:
        return []
    return [_parsed_or_error(entry, index) for index, entry in enumerate(raw_calls)]


def parse_tools(llm_params: Optional[dict]) -> Toolset:
    """Parse the effective request's tools after the caller has merged defaults."""
    return Toolset(tools=parse_tool_definitions((llm_params or {}).get("tools") or []))

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackManager
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import Runnable

from nemoguardrails.colang.v2_x.lang.colang_ast import Flow
from nemoguardrails.colang.v2_x.runtime.flows import InternalEvent, InternalEvents
from nemoguardrails.context import (
    llm_call_info_var,
    llm_response_metadata_var,
    reasoning_trace_var,
    tool_calls_var,
)
from nemoguardrails.integrations.langchain.message_utils import dicts_to_messages
from nemoguardrails.logging.callbacks import logging_callbacks
from nemoguardrails.logging.explain import LLMCallInfo


class LLMCallException(Exception):
    """A wrapper around the LLM call invocation exception.

    This is used to propagate the exception out of the `generate_async` call (the default behavior is to
    catch it and return an "Internal server error." message.
    """

    def __init__(self, inner_exception: Any):
        super().__init__(f"LLM Call Exception: {str(inner_exception)}")
        self.inner_exception = inner_exception


def _infer_model_name(llm: BaseLanguageModel):
    """Helper to infer the model name based from an LLM instance.

    Because not all models implement correctly _identifying_params from LangChain, we have to
    try to do this manually.
    """
    for attr in ["model", "model_name"]:
        if hasattr(llm, attr):
            val = getattr(llm, attr)
            if isinstance(val, str):
                return val

    model_kwargs = getattr(llm, "model_kwargs", None)
    if model_kwargs and isinstance(model_kwargs, Dict):
        for attr in ["model", "model_name", "name"]:
            val = model_kwargs.get(attr)
            if isinstance(val, str):
                return val

    # If we still can't figure out, return "unknown".
    return "unknown"


async def llm_call(
    llm: Optional[BaseLanguageModel],
    prompt: Union[str, List[dict]],
    model_name: Optional[str] = None,
    model_provider: Optional[str] = None,
    stop: Optional[List[str]] = None,
    custom_callback_handlers: Optional[Sequence[AsyncCallbackHandler]] = None,
    llm_params: Optional[dict] = None,
) -> str:
    """Calls the LLM with a prompt and returns the generated text.

    Args:
        llm: The language model instance to use
        prompt: The prompt string or list of messages
        model_name: Optional model name for tracking
        model_provider: Optional model provider for tracking
        stop: Optional list of stop tokens
        custom_callback_handlers: Optional list of callback handlers
        llm_params: Optional configuration dictionary to pass to the LLM (e.g., temperature, max_tokens)

    Returns:
        The generated text response
    """
    if llm is None:
        raise LLMCallException("No LLM provided to llm_call()")
    _setup_llm_call_info(llm, model_name, model_provider)
    all_callbacks = _prepare_callbacks(custom_callback_handlers)

    generation_llm: Union[BaseLanguageModel, Runnable] = (
        llm.bind(stop=stop, **llm_params) if llm_params and llm is not None else llm
    )

    if isinstance(prompt, str):
        response = await _invoke_with_string_prompt(
            generation_llm, prompt, all_callbacks
        )
    else:
        response = await _invoke_with_message_list(
            generation_llm, prompt, all_callbacks
        )

    _store_reasoning_traces(response)
    _store_tool_calls(response)
    _store_response_metadata(response)
    return _extract_content(response)


def _setup_llm_call_info(
    llm: BaseLanguageModel, model_name: Optional[str], model_provider: Optional[str]
) -> None:
    """Initialize or update LLM call info in context."""
    llm_call_info = llm_call_info_var.get()
    if llm_call_info is None:
        llm_call_info = LLMCallInfo()
        llm_call_info_var.set(llm_call_info)

    llm_call_info.llm_model_name = model_name or _infer_model_name(llm)
    llm_call_info.llm_provider_name = model_provider


def _prepare_callbacks(
    custom_callback_handlers: Optional[Sequence[AsyncCallbackHandler]],
) -> BaseCallbackManager:
    """Prepare callback manager with custom handlers if provided."""
    if custom_callback_handlers and custom_callback_handlers != [None]:
        return BaseCallbackManager(
            handlers=logging_callbacks.handlers + list(custom_callback_handlers),
            inheritable_handlers=logging_callbacks.handlers
            + list(custom_callback_handlers),
        )
    return logging_callbacks


async def _invoke_with_string_prompt(
    llm: Union[BaseLanguageModel, Runnable],
    prompt: str,
    callbacks: BaseCallbackManager,
):
    """Invoke LLM with string prompt."""
    try:
        return await llm.ainvoke(prompt, config=RunnableConfig(callbacks=callbacks))
    except Exception as e:
        raise LLMCallException(e)


async def _invoke_with_message_list(
    llm: Union[BaseLanguageModel, Runnable],
    prompt: List[dict],
    callbacks: BaseCallbackManager,
):
    """Invoke LLM with message list after converting to LangChain format."""
    messages = _convert_messages_to_langchain_format(prompt)

    try:
        return await llm.ainvoke(messages, config=RunnableConfig(callbacks=callbacks))
    except Exception as e:
        raise LLMCallException(e)


def _convert_messages_to_langchain_format(prompt: List[dict]) -> List:
    """Convert message list to LangChain message format."""
    return dicts_to_messages(prompt)


def _store_reasoning_traces(response) -> None:
    if hasattr(response, "additional_kwargs"):
        additional_kwargs = response.additional_kwargs
        if (
            isinstance(additional_kwargs, dict)
            and "reasoning_content" in additional_kwargs
        ):
            reasoning_content = additional_kwargs["reasoning_content"]
            if reasoning_content:
                reasoning_trace_var.set(reasoning_content)


def _store_tool_calls(response) -> None:
    """Extract and store tool calls from response in context."""
    tool_calls = getattr(response, "tool_calls", None)
    tool_calls_var.set(tool_calls)


def _store_response_metadata(response) -> None:
    """Store response metadata excluding content for metadata preservation.

    Also extracts reasoning content from additional_kwargs if available from LangChain.
    """
    if hasattr(response, "model_fields"):
        metadata = {}
        for field_name in response.model_fields:
            if (
                field_name != "content"
            ):  # Exclude content since it may be modified by rails
                metadata[field_name] = getattr(response, field_name)
        llm_response_metadata_var.set(metadata)

    else:
        llm_response_metadata_var.set(None)


def _extract_content(response) -> str:
    """Extract text content from response."""
    if hasattr(response, "content"):
        return response.content
    return str(response)


def get_colang_history(
    events: List[dict],
    include_texts: bool = True,
    remove_retrieval_events: bool = False,
) -> str:
    """Creates a history of user messages and bot responses in colang format.
    user "Hi, how are you today?"
      express greeting
    bot express greeting
      "Greetings! I am the official NVIDIA Benefits Ambassador AI bot and I'm here to assist you."
    user "What can you help me with?"
      ask capabilities
    bot inform capabilities
      "As an AI, I can provide you with a wide range of services, such as ..."

    """

    history = ""

    if not events:
        return history

    # We try to automatically detect if we have a Colang 1.0 or a 2.x history
    # TODO: Think about more robust approach?
    colang_version = "1.0"
    for event in events:
        if isinstance(event, InternalEvent):
            event = {"type": event.name, **event.arguments}

        if event["type"] in InternalEvents.ALL:
            colang_version = "2.x"

    if colang_version == "1.0":
        # We compute the index of the last bot message. We need it so that we include
        # the bot message instruction only for the last one.
        last_bot_intent_idx = len(events) - 1
        while last_bot_intent_idx >= 0:
            if events[last_bot_intent_idx]["type"] == "BotIntent":
                break
            last_bot_intent_idx -= 1

        for idx, event in enumerate(events):
            if event["type"] == "UserMessage" and include_texts:
                history += f'user "{event["text"]}"\n'
            elif event["type"] == "UserIntent":
                if include_texts:
                    history += f"  {event['intent']}\n"
                else:
                    history += f"user {event['intent']}\n"
            elif event["type"] == "BotIntent":
                # If we have instructions, we add them before the bot message.
                # But we only do that for the last bot message.
                if "instructions" in event and idx == last_bot_intent_idx:
                    history += f"# {event['instructions']}\n"
                history += f"bot {event['intent']}\n"
            elif event["type"] == "StartUtteranceBotAction" and include_texts:
                history += f'  "{event["script"]}"\n'
            # We skip system actions from this log
            elif event["type"] == "StartInternalSystemAction" and not event.get(
                "is_system_action"
            ):
                if (
                    remove_retrieval_events
                    and event["action_name"] == "retrieve_relevant_chunks"
                ):
                    continue
                history += f"execute {event['action_name']}\n"
            elif event["type"] == "InternalSystemActionFinished" and not event.get(
                "is_system_action"
            ):
                if (
                    remove_retrieval_events
                    and event["action_name"] == "retrieve_relevant_chunks"
                ):
                    continue

                # We make sure the return value is a string with no new lines
                return_value = str(event["return_value"]).replace("\n", " ")
                history += f"# The result was {return_value}\n"
            elif event["type"] == "mask_prev_user_message":
                utterance_to_replace = get_last_user_utterance(events[:idx])
                # We replace the last user utterance that led to jailbreak rail trigger with a placeholder text
                split_history = history.rsplit(utterance_to_replace, 1)
                placeholder_text = "<<<This text is hidden because the assistant should not talk about this.>>>"
                history = placeholder_text.join(split_history)

    elif colang_version == "2.x":
        new_history: List[str] = []

        # Structure the user/bot intent/action events
        action_group: List[InternalEvent] = []
        current_intent: Optional[str] = None

        previous_event = None
        for event in events:
            if not isinstance(event, InternalEvent):
                # Skip non-internal events
                continue

            if (
                event.name == InternalEvents.USER_ACTION_LOG
                and previous_event
                and events_to_dialog_history([previous_event])
                == events_to_dialog_history([event])
            ):
                # Remove duplicated user action log events that stem from the same user event as the previous event
                continue

            if (
                event.name == InternalEvents.BOT_ACTION_LOG
                or event.name == InternalEvents.USER_ACTION_LOG
            ):
                if len(action_group) > 0 and (
                    current_intent is None
                    or current_intent != event.arguments["intent_flow_id"]
                ):
                    new_history.append(events_to_dialog_history(action_group))
                    new_history.append("")
                    action_group.clear()

                action_group.append(event)
                current_intent = event.arguments["intent_flow_id"]

                previous_event = event
            elif (
                event.name == InternalEvents.BOT_INTENT_LOG
                or event.name == InternalEvents.USER_INTENT_LOG
            ):
                if event.arguments["flow_id"] == current_intent:
                    # Found parent of current group
                    if event.name == InternalEvents.BOT_INTENT_LOG:
                        new_history.append(events_to_dialog_history([event]))
                        new_history.append(events_to_dialog_history(action_group))
                    elif event.arguments["flow_id"] is not None:
                        new_history.append(events_to_dialog_history(action_group))
                        new_history.append(events_to_dialog_history([event]))
                    new_history.append("")
                else:
                    # New unrelated intent
                    if action_group:
                        new_history.append(events_to_dialog_history(action_group))
                        new_history.append("")
                    new_history.append(events_to_dialog_history([event]))
                    new_history.append("")
                # Start a new group
                action_group.clear()
                current_intent = None

                previous_event = event

        if action_group:
            new_history.append(events_to_dialog_history(action_group))

        history = "\n".join(new_history).rstrip("\n")

    return history


def events_to_dialog_history(events: List[InternalEvent]) -> str:
    """Create the dialog history based on provided events."""
    result = ""
    for idx, event in enumerate(events):
        identifier = from_log_event_to_identifier(event.name)
        if idx == 0:
            intent = f"{identifier}: {event.arguments['flow_id']}"
        else:
            intent = f"{event.arguments['flow_id']}"
        param_value = event.arguments["parameter"]
        if param_value is not None:
            if isinstance(param_value, str):
                # convert new lines to \n token, so that few-shot learning won't mislead LLM
                param_value = param_value.replace("\n", "\\n")
                intent = f'{intent} "{param_value}"'
            else:
                intent = f"{intent} {param_value}"
        result += intent
        if idx + 1 < len(events):
            result += "\n  and "
    return result


def from_log_event_to_identifier(event_name: str) -> str:
    """convert log message to prompt interaction identifier."""
    if event_name == InternalEvents.BOT_INTENT_LOG:
        return "bot intent"
    elif event_name == InternalEvents.BOT_ACTION_LOG:
        return "bot action"
    elif event_name == InternalEvents.USER_INTENT_LOG:
        return "user intent"
    elif event_name == InternalEvents.USER_ACTION_LOG:
        return "user action"
    return ""


def flow_to_colang(flow: Union[dict, Flow]) -> str:
    """Converts a flow to colang format.

    Example flow:
    ```
      - user: ask capabilities
      - bot: inform capabilities
    ```

    to colang:

    ```
    user ask capabilities
    bot inform capabilities
    ```

    """

    # TODO: use the source code lines if available.

    colang_flow = ""
    if isinstance(flow, Flow):
        # TODO: generate the flow code from the flow.elements array
        pass
    else:
        for element in flow["elements"]:
            if "_type" not in element:
                raise Exception("bla")
            if element["_type"] == "UserIntent":
                colang_flow += f"user {element['intent_name']}\n"
            elif element["_type"] == "run_action" and element["action_name"] == "utter":
                colang_flow += f"bot {element['action_params']['value']}\n"

    return colang_flow


def get_last_user_utterance(events: List[dict]) -> Optional[str]:
    """Returns the last user utterance from the events."""
    for event in reversed(events):
        if event["type"] == "UserMessage":
            return event["text"]

    return None


def get_retrieved_relevant_chunks(
    events: List[dict], skip_user_message: Optional[bool] = False
) -> Optional[str]:
    """Returns the retrieved chunks for current user utterance from the events."""
    for event in reversed(events):
        if not skip_user_message and event["type"] == "UserMessage":
            break
        if event["type"] == "ContextUpdate" and "relevant_chunks" in event.get(
            "data", {}
        ):
            return (event["data"]["relevant_chunks"] or "").strip()

    return None


def get_last_user_utterance_event(events: List[dict]) -> Optional[dict]:
    """Returns the last user utterance from the events."""
    for event in reversed(events):
        if isinstance(event, dict) and event["type"] == "UserMessage":
            return event

    return None


def get_last_user_utterance_event_v2_x(events: List[dict]) -> Optional[dict]:
    """Returns the last user utterance from the events."""
    for event in reversed(events):
        if isinstance(event, dict) and event["type"] == "UtteranceUserActionFinished":
            return event

    return None


def get_last_user_intent_event(events: List[dict]) -> Optional[dict]:
    """Returns the last user intent from the events."""
    for event in reversed(events):
        if event["type"] == "UserIntent":
            return event

    return None


def get_last_bot_intent_event(events: List[dict]) -> Optional[dict]:
    """Returns the last user intent from the events."""
    for event in reversed(events):
        if event["type"] == "BotIntent":
            return event

    return None


def get_last_bot_utterance_event(events: List[dict]) -> Optional[dict]:
    """Returns the last bot utterance from the events."""
    for event in reversed(events):
        if event["type"] == "StartUtteranceBotAction":
            return event

    return None


def remove_text_messages_from_history(history: str) -> str:
    """Helper that given a history in colang format, removes all texts."""

    # Get rid of messages from the user
    history = re.sub(r'user "[^\n]+"\n {2}', "user ", history)

    # Get rid of one line user messages
    history = re.sub(r"^\s*user [^\n]+\n\n", "", history)

    # Get rid of bot messages
    history = re.sub(r'bot ([^\n]+)\n {2}"[\s\S]*?"', r"bot \1", history)

    return history


def get_first_nonempty_line(s: str) -> Optional[str]:
    """Helper that returns the first non-empty line from a string"""
    if not s:
        return None

    first_nonempty_line = None
    lines = [line.strip() for line in s.split("\n")]
    for line in lines:
        if len(line) > 0:
            first_nonempty_line = line
            break

    return first_nonempty_line


def get_top_k_nonempty_lines(s: str, k: int = 1) -> Optional[List[str]]:
    """Helper that returns a list with the top k non-empty lines from a string.

    If there are less than k non-empty lines, it returns a smaller number of lines."""
    if not s:
        return None

    lines = [line.strip() for line in s.split("\n")]
    # Ignore line comments and empty lines
    lines = [line for line in lines if len(line) > 0 and line[0] != "#"]

    return lines[:k]


def strip_quotes(s: str) -> str:
    """Helper that removes quotes from a string if the entire string is between quotes"""
    if s and s[0] == '"':
        if s[-1] == '"':
            s = s[1:-1]
        else:
            s = s[1:]
    return s


def get_multiline_response(s: str) -> str:
    """Helper that extracts multi-line responses from the LLM.
    Stopping conditions: when a non-empty line ends with a quote or when the token "user" appears after a newline.
    Empty lines at the begging of the string are skipped."""

    # Check if the token "user" appears after a newline, as this would mark a new dialogue turn.
    # Remove everything after this marker.
    if "\nuser" in s:
        # Remove everything after the interrupt signal
        s = s.split("\nuser")[0]

    lines = [line.strip() for line in s.split("\n")]
    result = ""
    for line in lines:
        # Keep getting additional non-empty lines until the message ends
        if len(line) > 0:
            if len(result) == 0:
                result = line
            else:
                result += "\n" + line
            if line.endswith('"'):
                break

    return result


def remove_action_intent_identifiers(lines: List[str]) -> List[str]:
    """Removes the action/intent identifiers."""
    return [
        s.replace("bot intent: ", "")
        .replace("bot action: ", "")
        .replace("user intent: ", "")
        .replace("user action: ", "")
        for s in lines
    ]


def get_initial_actions(strings: List[str]) -> List[str]:
    """Returns the first action before an empty line."""
    previous_strings = []
    for string in strings:
        if string == "":
            break
        previous_strings.append(string)
    return previous_strings


def get_first_user_intent(strings: List[str]) -> Optional[str]:
    """Returns first user intent."""
    for string in strings:
        if string.startswith("user intent: "):
            return string.replace("user intent: ", "")
    return None


def get_first_bot_intent(strings: List[str]) -> Optional[str]:
    """Returns first bot intent."""
    for string in strings:
        if string.startswith("bot intent: "):
            return string.replace("bot intent: ", "")
    return None


def get_first_bot_action(strings: List[str]) -> Optional[str]:
    """Returns first bot action."""
    action_started = False
    action: str = ""
    for string in strings:
        if string.startswith("bot action: "):
            if action != "":
                action += "\n"
            action += string.replace("bot action: ", "")
            action_started = True
        elif (
            string.startswith("  and") or string.startswith("  or")
        ) and action_started:
            action = action + string
        elif string == "":
            action_started = False
            continue
        elif action != "":
            return action
    return action


def escape_flow_name(name: str) -> str:
    """Escape invalid keywords in flow names."""
    # TODO: We need to figure out how we can distinguish from valid flow parameters
    result = (
        name.replace(" and ", "_and_")
        .replace(" or ", "_or_")
        .replace(" as ", "_as_")
        .replace("-", "_")
    )
    result = re.sub(r"\b\d+\b", lambda match: f"_{match.group()}_", result)
    # removes non-word chars and leading digits in a word
    result = re.sub(r"\b\d+|[^\w\s]", "", result)
    return result


def get_and_clear_reasoning_trace_contextvar() -> Optional[str]:
    """Get the current reasoning trace and clear it from the context.

    Returns:
        Optional[str]: The reasoning trace if one exists, None otherwise.
    """
    if reasoning_trace := reasoning_trace_var.get():
        reasoning_trace_var.set(None)
        return reasoning_trace
    return None


def get_and_clear_tool_calls_contextvar() -> Optional[list]:
    """Get the current tool calls and clear them from the context.

    Returns:
        Optional[list]: The tool calls if they exist, None otherwise.
    """
    if tool_calls := tool_calls_var.get():
        tool_calls_var.set(None)
        return tool_calls
    return None


def extract_tool_calls_from_events(events: list) -> Optional[list]:
    """Extract tool_calls from BotToolCalls events.

    Args:
        events: List of events to search through

    Returns:
        tool_calls if found in BotToolCalls event, None otherwise
    """
    for event in events:
        if event.get("type") == "BotToolCalls":
            return event.get("tool_calls")
    return None


def extract_bot_thinking_from_events(events: list):
    for event in events:
        if event.get("type") == "BotThinking":
            return event.get("content")


def get_and_clear_response_metadata_contextvar() -> Optional[dict]:
    """Get the current response metadata and clear it from the context.

    Returns:
        Optional[dict]: The response metadata if it exists, None otherwise.
    """
    if metadata := llm_response_metadata_var.get():
        llm_response_metadata_var.set(None)
        return metadata
    return None

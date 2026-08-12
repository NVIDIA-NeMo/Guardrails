# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Colang-facing helpers for the LLM generation path.

Conversation-history rendering, event accessors, and the completion parsers that depend
on Colang intent/action vocabulary, used by the Colang runtimes and by
``nemoguardrails.actions.llm.generation``.

This module imports the Colang v2.x AST and runtime, so importing it pulls in the Colang
runtime. Two groups of helpers were moved out for that reason, and are re-exported below
so existing imports keep working:

- model invocation -> :mod:`nemoguardrails.llm.call`
- Colang-free completion text helpers -> :mod:`nemoguardrails.llm.completion_parsing`

Prefer importing those from their canonical modules; a rail action that imports them from
here will drag the Colang runtime into its dependency graph.
"""

import re
from typing import Any, Dict, List, Optional, Union

from nemoguardrails.colang.v2_x.lang.colang_ast import Flow
from nemoguardrails.colang.v2_x.runtime.flows import InternalEvent, InternalEvents
from nemoguardrails.context import (
    llm_response_metadata_var,
    reasoning_trace_var,
    tool_calls_var,
)

# Re-exported for backward compatibility only; see the module docstring for the canonical homes.
from nemoguardrails.llm.call import llm_call, warn_if_truncated  # noqa: F401
from nemoguardrails.llm.completion_parsing import (  # noqa: F401
    get_multiline_response,
    remove_action_intent_identifiers,
    strip_quotes,
)


def _extract_user_text_from_event(event_text: Union[str, List[Dict[str, Any]]]) -> str:
    """Flatten a multimodal user-message payload into a string for colang history.

    Multimodal user events carry ``event_text`` as a list of OpenAI-style
    content parts (``[{"type": "text", "text": "..."}, {"type": "image_url",
    "image_url": {...}}, ...]``). Including the full list in the colang
    history bloats the context with raw base64 data; this helper extracts the
    visible text parts and appends a ``[+ image]`` marker when one or more
    image parts were present.

    Non-string text fields (``None`` or other types) inside a content part
    are skipped so the ``" ".join(...)`` step cannot crash. If the message
    is image-only, the result is just ``"[+ image]"`` without a leading
    space.

    Args:
        event_text: Either a string (already flat) or a list of multimodal
            content parts.

    Returns:
        The flattened text. A list input always produces a string; a string
        input is returned unchanged.
    """
    if isinstance(event_text, list):
        text_parts = []
        has_images = False
        for item in event_text:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
                elif item.get("type") == "image_url":
                    has_images = True
        text = " ".join(text_parts)
        if has_images:
            text = f"{text} [+ image]".strip() if text else "[+ image]"
        return text
    return event_text


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
                history += f'user "{_extract_user_text_from_event(event["text"])}"\n'
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
            elif event["type"] == "StartInternalSystemAction" and not event.get("is_system_action"):
                if remove_retrieval_events and event["action_name"] == "retrieve_relevant_chunks":
                    continue
                history += f"execute {event['action_name']}\n"
            elif event["type"] == "InternalSystemActionFinished" and not event.get("is_system_action"):
                if remove_retrieval_events and event["action_name"] == "retrieve_relevant_chunks":
                    continue

                # We make sure the return value is a string with no new lines
                return_value = str(event["return_value"]).replace("\n", " ")
                history += f"# The result was {return_value}\n"
            elif event["type"] == "mask_prev_user_message":
                utterance_to_replace = get_last_user_utterance(events[:idx])
                if utterance_to_replace:
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
                and events_to_dialog_history([previous_event]) == events_to_dialog_history([event])
            ):
                # Remove duplicated user action log events that stem from the same user event as the previous event
                continue

            if event.name == InternalEvents.BOT_ACTION_LOG or event.name == InternalEvents.USER_ACTION_LOG:
                if len(action_group) > 0 and (
                    current_intent is None or current_intent != event.arguments["intent_flow_id"]
                ):
                    new_history.append(events_to_dialog_history(action_group))
                    new_history.append("")
                    action_group.clear()

                action_group.append(event)
                current_intent = event.arguments["intent_flow_id"]

                previous_event = event
            elif event.name == InternalEvents.BOT_INTENT_LOG or event.name == InternalEvents.USER_INTENT_LOG:
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
            return _extract_user_text_from_event(event["text"])

    return None


def get_retrieved_relevant_chunks(events: List[dict], skip_user_message: Optional[bool] = False) -> Optional[str]:
    """Returns the retrieved chunks for current user utterance from the events."""
    for event in reversed(events):
        if not skip_user_message and event["type"] == "UserMessage":
            break
        if event["type"] == "ContextUpdate" and "relevant_chunks" in event.get("data", {}):
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


def _has_unclosed_quote(s: str) -> bool:
    """Check if a string has an unclosed double quote (ignoring escaped quotes)."""
    # Strip escaped backslashes first so that \\" is not misread as \" (escaped quote).
    return s.replace("\\\\", "").replace('\\"', "").count('"') % 2 == 1


_MAX_QUOTE_CONTINUATION_LINES = 50


def get_first_bot_action(strings: List[str]) -> Optional[str]:
    """Returns first bot action."""
    action_started = False
    action: str = ""
    continuation_lines = 0
    for string in strings:
        # Collect continuation lines for multi-line quoted strings,
        # joining with escaped newlines to keep the Colang statement valid.
        if action and _has_unclosed_quote(action):
            if continuation_lines >= _MAX_QUOTE_CONTINUATION_LINES:
                # Safety bound: stop collecting to avoid absorbing all input.
                return action
            action += "\\n" + string
            continuation_lines += 1
            if not _has_unclosed_quote(action):
                continuation_lines = 0
            continue
        if string.startswith("bot action: "):
            if action != "":
                action += "\n"
            action += string.replace("bot action: ", "")
            action_started = True
        elif (string.startswith("  and") or string.startswith("  or")) and action_started:
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
    result = name.replace(" and ", "_and_").replace(" or ", "_or_").replace(" as ", "_as_").replace("-", "_")
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
    """Extract tool_calls from runtime events.

    ``StartToolCallBotAction`` carries the tool calls that passed tool-output
    rails and should be returned to the caller. ``BotToolCalls`` is used as a
    fallback for paths that do not emit the post-rail action event.
    """
    bot_tool_calls = None

    for event in events:
        if event.get("type") == "StartToolCallBotAction":
            tool_calls = event.get("tool_calls")
            if tool_calls is not None:
                return tool_calls
        elif event.get("type") == "BotToolCalls":
            bot_tool_calls = event.get("tool_calls")

    return bot_tool_calls


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

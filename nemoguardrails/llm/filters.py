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
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Tuple

from nemoguardrails.actions.llm.utils import (
    get_colang_history,
    remove_action_intent_identifiers,
)


def colang(events: List[dict]) -> str:
    """Filter that turns an array of events into a colang history."""
    return get_colang_history(events)


def co_v2(
    events: List[dict],
) -> str:
    """Creates a history of user messages and bot responses in colang format.

    user said "Hi, how are you today?"
    bot say "Greetings! I am the official NVIDIA Benefits Ambassador AI bot and I'm here to assist you."
    user said "What can you help me with?"
    bot say "As an AI, I can provide you with a wide range of services, such as ..."

    """

    history = ""

    if not events:
        return history

    system_actions = [
        "retrieve_relevant_chunks",
        "create_event",
        "wolfram alpha request",
        "summarize_document",
        "apify",
        "bing_search",
        "google_search",
        "google_serper",
        "openweather_query",
        "searx_search",
        "serp_api_query",
        "wikipedia_query",
        "wolframalpha_query",
        "zapier_nla_query",
        "call activefence api",
        "call gcpnlp api",
        "jailbreak_detection_heuristics",
        "self_check_hallucination",
        "llama_guard_check_input",
        "llama_guard_check_output",
        "alignscore_check_facts",
        "alignscore request",
        "self_check_facts",
        "self_check_input",
        "self_check_output",
        "AddFlowsAction",
        "RemoveFlowsAction",
        "CheckForActiveEventMatchAction",
        "CheckFlowDefinedAction",
        "CheckValidFlowExistsAction",
        "generate_bot_message",
        "GenerateFlowAction",
        "GenerateFlowContinuationAction",
        "GenerateFlowFromInstructionsAction",
        "GenerateFlowFromNameAction",
        "generate_intent_steps_message",
        "generate_next_step",
        "GenerateUserIntentAction",
        "GenerateValueAction",
        "GetLastUserMessageAction",
    ]

    # TODO: figure out why we have a duplicated StartUtteranceBotAction event.
    processed_ids = set()

    for idx, event in enumerate(events):
        if isinstance(event, dict):
            if event.get("uid") is None or event["uid"] not in processed_ids:
                if event["type"] == "UtteranceUserActionFinished":
                    history += f'  user said "{event["final_transcript"]}"\n'
                elif event["type"] == "StartUtteranceBotAction":
                    history += f'  bot say "{event["script"]}"\n'

                elif event["type"] == "StartTool":
                    s = f"  await {event['flow_name']}"
                    for k, v in event.items():
                        if k in [
                            "type",
                            "uid",
                            "event_created_at",
                            "source_uid",
                            "flow_name",
                        ]:
                            continue
                        s += f' ${k}="{v}"'
                    history += s + "\n"

                elif (
                    event["type"].endswith("ActionFinished")
                    and event.get("action_name")
                    and event["action_name"] not in system_actions
                ):
                    # history += f"  await {str(event['action_name'])}()\n"
                    history += f"  # {str(event.get('return_value'))}\n"

                processed_ids.add(event.get("uid"))

    return history


def colang_without_identifiers(events: List[dict]) -> str:
    """Filter that turns an array of events into a colang history."""
    return remove_action_intent_identifiers([get_colang_history(events)])[0]


def to_messages(colang_history: str) -> List[dict]:
    """Filter that given a history in colang format, returns all messages."""
    messages = []

    # For now, we use a simple heuristic. The line `user "xxx"` gets translated to
    # a message from the user, and the rest gets translated to messages from the assistant.
    lines = colang_history.split("\n")

    bot_lines = []
    for i, line in enumerate(lines):
        if line.startswith('user "'):
            # If we have bot lines in the buffer, we first add a bot message.
            if bot_lines:
                messages.append({"type": "assistant", "content": "\n".join(bot_lines)})
                bot_lines = []

            messages.append({"type": "user", "content": line[6:-1]})

        elif line.strip() == "":
            # On empty lines, we also reset the bot buffer.
            if bot_lines:
                messages.append({"type": "assistant", "content": "\n".join(bot_lines)})
                bot_lines = []
        else:
            if i > 0 and lines[i - 1].startswith('user "'):
                line = "User intent: " + line.strip()
            elif line.startswith("user "):
                line = "User intent: " + line[5:].strip()
            elif line.startswith("bot "):
                line = "Bot intent: " + line[4:].strip()
            elif line.startswith('  "'):
                line = "Bot message: " + line[2:].strip()
            bot_lines.append(line)

    # Check if there is a last message from the bot.
    if bot_lines:
        messages.append({"type": "bot", "content": "\n".join(bot_lines)})

    return messages


def to_messages_v2(colang_history: str) -> List[dict]:
    """Filter that given a history in colang 2.0 format, returns all messages."""
    messages = []

    # For now, we use a simple heuristic. The line `user "xxx"` gets translated to
    # a message from the user, and the rest gets translated to messages from the assistant.
    lines = colang_history.split("\n")

    user_lines = []
    bot_lines = []
    for line in lines:
        if line.startswith("user action:"):
            if len(bot_lines) > 0:
                messages.append({"type": "assistant", "content": "\n".join(bot_lines)})
                bot_lines = []
            user_lines.append(line)
        elif line.startswith("user intent:") or line.startswith("bot"):
            if len(user_lines) > 0:
                messages.append({"type": "user", "content": "\n".join(user_lines)})
                user_lines = []
            bot_lines.append(line)
        elif line.strip() != "":
            if len(user_lines) > 0:
                user_lines.append(line)
            elif len(bot_lines) > 0:
                bot_lines.append(line)

    if len(user_lines) > 0:
        messages.append({"type": "user", "content": "\n".join(user_lines)})
        user_lines = []
    elif len(bot_lines) > 0:
        messages.append({"type": "assistant", "content": "\n".join(bot_lines)})
        bot_lines = []

    return messages


def to_intent_messages(colang_history: str) -> List[dict]:
    messages = []

    lines = colang_history.split("\n")
    for i, line in enumerate(lines):
        if line.startswith('user "'):
            continue
        else:
            if i > 0 and lines[i - 1].startswith('user "'):
                line = "User intent: " + line.strip()
                messages.append({"type": "user", "content": line})
            elif line.startswith("user "):
                line = "User intent: " + line[5:].strip()
                messages.append({"type": "user", "content": line})
            elif line.startswith("bot "):
                line = "Bot intent: " + line[4:].strip()
                messages.append({"type": "assistant", "content": line})
            elif line.startswith('  "'):
                continue

    return messages


def to_intent_messages_2(colang_history: str) -> List[dict]:
    messages = []

    lines = colang_history.split("\n")
    for i, line in enumerate(lines):
        if line.startswith('user "'):
            messages.append({"type": "user", "content": line[6:-1]})
        else:
            if i > 0 and lines[i - 1].startswith('user "'):
                continue
            if line.startswith("bot "):
                line = "Bot intent: " + line[4:].strip()
                messages.append({"type": "assistant", "content": line})
            elif line.startswith('  "'):
                line = "Bot message: " + line[2:].strip()
                messages.append({"type": "assistant", "content": line})

    return messages


def verbose_v1(colang_history: str) -> str:
    """Filter that given a history in colang format, returns a verbose version of the history."""
    lines = colang_history.split("\n")
    for i, line in enumerate(lines):
        if line.startswith('user "'):
            lines[i] = 'User message: "' + line[6:]
        elif (
            line.startswith("  ")
            and i > 0
            and lines[i - 1].startswith("User message: ")
        ):
            lines[i] = "User intent: " + line.strip()
        elif line.startswith("user "):
            lines[i] = "User intent: " + line[5:].strip()
        elif line.startswith("bot "):
            lines[i] = "Bot intent: " + line[4:]
        elif line.startswith('  "'):
            lines[i] = "Bot message: " + line[2:]

    return "\n".join(lines)


def to_chat_messages(events: List[dict]) -> str:
    """Filter that turns an array of events into a sequence of user/assistant messages.

    Properly handles multimodal content by preserving the structure when the content
    is in the format of a Message object with potential image_url content.
    """
    messages = []
    for event in events:
        if event["type"] == "UserMessage":
            # Preserve the original structure when possible to support multimodal content
            content = event["text"]
            messages.append({"role": "user", "content": content})
        elif event["type"] == "StartUtteranceBotAction":
            messages.append({"role": "assistant", "content": event["script"]})
        elif event["type"] == "SystemMessage" and "content" in event:
            # Handle system messages that might contain multimodal content
            messages.append({"role": "system", "content": event["content"]})

    return messages


def user_assistant_sequence(events: List[dict]) -> str:
    """Filter that turns an array of events into a sequence of user/assistant messages.

    The output will look like:
       ```
       User: hi
       Assistant: Hello there!
       User: What can you do?
       Assistant: I can help with many things.
       ```

    For multimodal content, it extracts text content and indicates if there were images.
    """
    history_items = []
    for event in events:
        if event["type"] == "UserMessage":
            content = event["text"]
            # Handle multimodal content by extracting text
            if isinstance(content, list):
                text_parts = []
                has_images = False
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            has_images = True
                text_content = " ".join(text_parts)
                if has_images:
                    text_content += " [+ image]"
                history_items.append("User: " + text_content)
            else:
                # Regular text content
                history_items.append("User: " + str(content))
        elif event["type"] == "StartUtteranceBotAction":
            history_items.append("Assistant: " + event["script"])

    return "\n".join(history_items)


def remove_text_messages(colang_history: str):
    """Filters that given a history in colang format, removes all texts."""

    # Get rid of messages from the user
    colang_history = re.sub(r'user "[^\n]+"\n {2}', "user ", colang_history)

    # Get rid of one line user messages
    colang_history = re.sub(r"^\s*user [^\n]+\n\n", "", colang_history)

    # Get rid of bot messages
    colang_history = re.sub(r'bot ([^\n]+)\n {2}"[\s\S]*?"', r"bot \1", colang_history)

    return colang_history


def first_turns(colang_history: str, n: int) -> str:
    """Returns the first n turns from a given colang history."""
    lines = colang_history.split("\n")
    turn_count = 0
    i = 0
    while i < len(lines):
        if lines[i].startswith('user "') or lines[i].startswith("user action: "):
            turn_count += 1
        if turn_count == n + 1:
            break
        i += 1

    return "\n".join(lines[0:i])


def last_turns(colang_history: str, n: int) -> str:
    """Returns the last n turns from a given colang history."""
    lines = colang_history.split("\n")
    turn_count = 0
    i = len(lines) - 1
    while i > 0:
        if lines[i].startswith('user "') or lines[i].startswith("user action: "):
            turn_count += 1
        if turn_count == n:
            break
        i -= 1

    return "\n".join(lines[i:])


def indent(text: str, n_spaces: int) -> str:
    """Indents the provided text with the provided number of spaces."""
    return textwrap.indent(text, " " * n_spaces)


def _previous_line(lines: List[str], i: int):
    """Returns the previous lines, skipping comments."""

    i = i - 1
    while i > 0 and lines[i].strip().startswith("#"):
        i -= 1
    return lines[i]


def remove_trailing_new_line(s: str):
    if s.endswith("\n"):
        s = s[:-1]
    return s


def conversation_to_events(conversation: List) -> List[dict]:
    """Filter that given a conversation, returns a list of events."""
    events = []
    for turn in conversation:
        if "user" in turn:
            events.append(
                {
                    "type": "UtteranceUserActionFinished",
                    "final_transcript": turn["user"],
                }
            )

        if "user_intent" in turn:
            events.append(
                {
                    "type": "UserIntent",
                    "intent": turn["user_intent"],
                }
            )

        if "bot" in turn:
            events.append(
                {
                    "type": "StartUtteranceBotAction",
                    "script": turn["bot"],
                }
            )

        if "bot_intent" in turn:
            events.append(
                {
                    "type": "BotIntent",
                    "intent": turn["bot_intent"],
                }
            )

    return events

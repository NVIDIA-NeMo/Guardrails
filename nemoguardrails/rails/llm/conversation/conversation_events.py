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

"""Conversation message conversion to Colang events."""

from collections.abc import MutableMapping
from typing import Any, List, Protocol

from nemoguardrails.colang.v2_x.runtime.flows import Action
from nemoguardrails.rails.llm.utils import get_history_cache_key
from nemoguardrails.utils import new_event_dict, new_uuid

__all__ = [
    "ConversationEventRails",
    "events_for_messages",
    "events_history_cache_prefix",
    "store_events_history_cache_entry",
]


class ConversationEventRails(Protocol):
    @property
    def config(self) -> Any: ...

    @property
    def events_history_cache(self) -> MutableMapping[str, list[dict]]: ...


def events_history_cache_prefix(
    events_history_cache: MutableMapping[str, list[dict]],
    messages: List[dict],
) -> tuple[int, list[dict]]:
    """Return the longest cached event prefix for the provided messages."""
    p = len(messages) - 1
    while p > 0:
        cache_key = get_history_cache_key(messages[0:p])
        if cache_key in events_history_cache:
            return p, events_history_cache[cache_key].copy()

        p -= 1

    return 0, []


def store_events_history_cache_entry(
    events_history_cache: MutableMapping[str, list[dict]],
    messages: List[dict],
    events: list[dict],
) -> None:
    """Store events for a complete v1 message history cache key."""
    cache_key = get_history_cache_key(messages)
    events_history_cache[cache_key] = events


def events_for_messages(rails: ConversationEventRails, messages: List[dict], state: Any):
    """Return Colang events corresponding to OpenAI-style messages."""
    events = []

    if rails.config.colang_version == "1.0":
        p, events = events_history_cache_prefix(rails.events_history_cache, messages)

        for idx in range(p, len(messages)):
            msg = messages[idx]
            if msg["role"] == "user":
                events.append(
                    {
                        "type": "UtteranceUserActionFinished",
                        "final_transcript": msg["content"],
                    }
                )

                if idx != len(messages) - 1:
                    events.append(
                        {
                            "type": "UserMessage",
                            "text": msg["content"],
                        }
                    )

            elif msg["role"] == "assistant":
                if msg.get("tool_calls"):
                    events.append({"type": "BotToolCalls", "tool_calls": msg["tool_calls"]})
                else:
                    action_uid = new_uuid()
                    start_event = new_event_dict(
                        "StartUtteranceBotAction",
                        script=msg["content"],
                        action_uid=action_uid,
                    )
                    finished_event = new_event_dict(
                        "UtteranceBotActionFinished",
                        final_script=msg["content"],
                        is_success=True,
                        action_uid=action_uid,
                    )
                    events.extend([start_event, finished_event])
            elif msg["role"] == "context":
                events.append({"type": "ContextUpdate", "data": msg["content"]})
            elif msg["role"] == "event":
                events.append(msg["event"])
            elif msg["role"] == "system":
                events.append({"type": "SystemMessage", "content": msg["content"]})
            elif msg["role"] == "tool":
                if idx == len(messages) - 1:
                    user_message = None
                    for prev_msg in reversed(messages[:idx]):
                        if prev_msg["role"] == "user":
                            user_message = prev_msg["content"]
                            break

                    if user_message:
                        if rails.config.rails.tool_input.flows:
                            tool_messages = []
                            for tool_idx in range(len(messages)):
                                if messages[tool_idx]["role"] == "tool":
                                    tool_messages.append(
                                        {
                                            "content": messages[tool_idx]["content"],
                                            "name": messages[tool_idx].get("name", "unknown"),
                                            "tool_call_id": messages[tool_idx].get("tool_call_id", ""),
                                        }
                                    )

                            events.append(
                                {
                                    "type": "UserToolMessages",
                                    "tool_messages": tool_messages,
                                }
                            )

                        else:
                            events.append({"type": "UserMessage", "text": user_message})

    else:
        for idx in range(len(messages)):
            msg = messages[idx]
            if msg["role"] == "user":
                events.append(
                    {
                        "type": "UtteranceUserActionFinished",
                        "final_transcript": msg["content"],
                    }
                )

            elif msg["role"] == "assistant":
                raise ValueError(
                    "Providing `assistant` messages as input is not supported for Colang 2.0 configurations."
                )
            elif msg["role"] == "context":
                events.append({"type": "ContextUpdate", "data": msg["content"]})
            elif msg["role"] == "event":
                events.append(msg["event"])
            elif msg["role"] == "system":
                events.append({"type": "SystemMessage", "content": msg["content"]})
            elif msg["role"] == "tool":
                action_uid = msg["tool_call_id"]
                return_value = msg["content"]
                action: Action = state.actions[action_uid]
                events.append(
                    new_event_dict(
                        f"{action.name}Finished",
                        action_uid=action_uid,
                        action_name=action.name,
                        status="success",
                        is_success=True,
                        return_value=return_value,
                        events=[],
                    )
                )

    return events

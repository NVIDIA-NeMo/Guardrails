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

"""Bot message extraction from Colang runtime events."""

import re
from dataclasses import dataclass, field
from typing import List

__all__ = ["BotMessageFromEvents", "bot_message_from_colang_events"]


@dataclass
class BotMessageFromEvents:
    message: dict
    extra_events: List[dict] = field(default_factory=list)


def bot_message_from_colang_events(
    colang_version: str,
    events: List[dict],
) -> BotMessageFromEvents:
    """Return the outward message represented by Colang runtime events."""
    if colang_version == "1.0":
        return _bot_message_from_colang_1_events(events)
    return _bot_message_from_colang_2_events(events)


def _bot_message_from_colang_1_events(events: List[dict]) -> BotMessageFromEvents:
    responses = []
    exception = None

    for event in events:
        if event["type"] == "StartUtteranceBotAction":
            # Check if we need to remove a message
            if event["script"] == "(remove last message)":
                responses = responses[0:-1]
            else:
                responses.append(event["script"])
        elif event["type"].endswith("Exception"):
            exception = event

    if exception:
        return BotMessageFromEvents(message={"role": "exception", "content": exception})

    return BotMessageFromEvents(message=_assistant_message_from_responses(responses))


def _bot_message_from_colang_2_events(events: List[dict]) -> BotMessageFromEvents:
    responses = []
    response_tool_calls = []
    response_events = []

    for event in events:
        start_action_match = re.match(r"Start(.*Action)", event["type"])

        if start_action_match:
            action_name = start_action_match[1]
            # TODO: is there an elegant way to extract just the arguments?
            arguments = {
                k: v
                for k, v in event.items()
                if k != "type" and k != "uid" and k != "event_created_at" and k != "source_uid" and k != "action_uid"
            }
            response_tool_calls.append(
                {
                    "id": event["action_uid"],
                    "type": "function",
                    "function": {"name": action_name, "arguments": arguments},
                }
            )

        elif event["type"] == "UtteranceBotActionFinished":
            responses.append(event["final_script"])
        else:
            # We just append the event
            response_events.append(event)

    message = _assistant_message_from_responses(responses)
    if response_tool_calls:
        message["tool_calls"] = response_tool_calls
    if response_events:
        message["events"] = response_events

    return BotMessageFromEvents(message=message)


def _assistant_message_from_responses(responses: List) -> dict:
    # Ensure all items in responses are strings
    string_responses = [str(response) if not isinstance(response, str) else response for response in responses]
    return {"role": "assistant", "content": "\n".join(string_responses)}

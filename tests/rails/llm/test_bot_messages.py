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

from nemoguardrails.rails.llm.generation.bot_messages import bot_message_from_colang_events


def test_bot_message_from_colang_1_events_joins_utterances_and_removes_last_message():
    result = bot_message_from_colang_events(
        "1.0",
        [
            {"type": "StartUtteranceBotAction", "script": "First"},
            {"type": "StartUtteranceBotAction", "script": "Second"},
            {"type": "StartUtteranceBotAction", "script": "(remove last message)"},
            {"type": "StartUtteranceBotAction", "script": 3},
        ],
    )

    assert result.message == {"role": "assistant", "content": "First\n3"}
    assert result.extra_events == []


def test_bot_message_from_colang_1_events_returns_exception_message():
    exception_event = {"type": "CustomException", "message": "blocked"}

    result = bot_message_from_colang_events(
        "1.0",
        [
            {"type": "StartUtteranceBotAction", "script": "First"},
            exception_event,
        ],
    )

    assert result.message == {"role": "exception", "content": exception_event}


def test_bot_message_from_colang_2_events_extracts_text_tool_calls_and_response_events():
    response_event = {"type": "CustomEvent", "value": 1}

    result = bot_message_from_colang_events(
        "2.x",
        [
            {
                "type": "StartGestureBotAction",
                "action_uid": "gesture-1",
                "uid": "event-1",
                "event_created_at": "now",
                "source_uid": "source-1",
                "gesture": "Wave",
            },
            {"type": "UtteranceBotActionFinished", "final_script": "Hello"},
            response_event,
        ],
    )

    assert result.message == {
        "role": "assistant",
        "content": "Hello",
        "tool_calls": [
            {
                "id": "gesture-1",
                "type": "function",
                "function": {
                    "name": "GestureBotAction",
                    "arguments": {"gesture": "Wave"},
                },
            }
        ],
        "events": [response_event],
    }


def test_bot_message_from_colang_2_events_returns_empty_assistant_message_for_no_outputs():
    result = bot_message_from_colang_events("2.x", [])

    assert result.message == {"role": "assistant", "content": ""}

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

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions.actions import ActionResult
from tests.utils import FakeLLM, any_event_conforms, event_conforms


@pytest.fixture
def rails_config():
    return RailsConfig.parse_object(
        {
            "models": [
                {
                    "type": "main",
                    "engine": "fake",
                    "model": "fake",
                }
            ],
            "user_messages": {
                "express greeting": ["Hello!"],
            },
            "flows": [
                {
                    "elements": [
                        {"user": "express greeting"},
                        {"execute": "increase_counter"},
                        {"bot": "express greeting"},
                    ]
                }
            ],
            "bot_messages": {
                "express greeting": ["Hello! How are you?"],
            },
        }
    )


@pytest.mark.asyncio
async def test_simple_context_update_from_action(rails_config):
    llm = FakeLLM(
        responses=[
            "  express greeting",
            "  express greeting",
        ]
    )

    async def increase_counter(context: dict):
        counter = context.get("counter", 0) + 1
        return ActionResult(context_updates={"counter": counter})

    llm_rails = LLMRails(config=rails_config, llm=llm)
    llm_rails.runtime.register_action(increase_counter)

    events = [{"type": "UtteranceUserActionFinished", "final_transcript": "Hello!"}]

    new_events = await llm_rails.runtime.generate_events(events)

    events.extend(new_events)
    events.append({"type": "UtteranceUserActionFinished", "final_transcript": "Hello!"})

    new_events = await llm_rails.runtime.generate_events(events)

    # The last event before listen should be a context update for the counter to "2"
    assert any_event_conforms(
        {"type": "ContextUpdate", "data": {"counter": 2}}, new_events
    )
    assert event_conforms({"type": "Listen"}, new_events[-1])

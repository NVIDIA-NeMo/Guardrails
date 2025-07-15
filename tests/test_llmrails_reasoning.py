# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict, List, Optional, Union

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.llmrails import _get_action_details_from_flow_id
from tests.utils import FakeLLM, clean_events, event_sequence_conforms


@pytest.fixture
def rails_config():
    return RailsConfig.parse_object(
        {
            "models": [
                {
                    "type": "main",
                    "engine": "fake",
                    "model": "fake",
                    "reasoning_config": {
                        "start_token": "<think>",
                        "end_token": "</think>",
                    },
                }
            ],
            "user_messages": {
                "express greeting": ["Hello!"],
                "ask math question": ["What is 2 + 2?", "5 + 9"],
            },
            "flows": [
                {
                    "elements": [
                        {"user": "express greeting"},
                        {"bot": "express greeting"},
                    ]
                },
                {
                    "elements": [
                        {"user": "ask math question"},
                        {"execute": "compute"},
                        {"bot": "provide math response"},
                        {"bot": "ask if user happy"},
                    ]
                },
            ],
            "bot_messages": {
                "express greeting": ["Hello! How are you?"],
                "provide response": ["The answer is 234", "The answer is 1412"],
            },
        }
    )


@pytest.mark.asyncio
async def test_1(rails_config):
    llm = FakeLLM(
        responses=[
            "<think>some redundant CoT text 1</think>\n  express greeting",
            "<think>some redundant CoT text 2</think>\n  ask math question",
            '<think>some redundant CoT text 3</think>\n  "The answer is 5"',
            '<think>some important COT text</think>\n  "Are you happy with the result?"',
        ]
    )

    async def compute(what: Optional[str] = "2 + 3"):
        return eval(what)

    llm_rails = LLMRails(config=rails_config, llm=llm)
    llm_rails.runtime.register_action(compute)

    messages = [{"role": "user", "content": "Hello!"}]
    bot_message = await llm_rails.generate_async(messages=messages)

    assert bot_message == {"role": "assistant", "content": "Hello! How are you?"}
    messages.append(bot_message)

    messages.append({"role": "user", "content": "2 + 3"})
    bot_message = await llm_rails.generate_async(messages=messages)
    assert bot_message == {
        "role": "assistant",
        "content": "<think>some important COT text</think>The answer is 5\nAre you happy with the result?",
    }

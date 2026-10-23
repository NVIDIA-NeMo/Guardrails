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

import json

import pytest

from nemoguardrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationResponse
from tests.recorded.cassette import normalize_body
from tests.recorded.rails.public_api.configs import OPENAI_BASELINE_CONFIG
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.vcr, pytest.mark.asyncio]


async def test_non_passthrough_tool_round_trip(openai_api_key, vcr):
    config = load_config(OPENAI_BASELINE_CONFIG)
    config.passthrough = False
    rails = LLMRails(config)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up the current weather in a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "enum": ["Paris"]}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "Look up the weather in Paris, then quote the tool result exactly."}]
    first = await rails.generate_async(
        messages=messages,
        options={"llm_params": {"tools": tools, "tool_choice": "required", "parallel_tool_calls": False}},
    )
    assert isinstance(first, GenerationResponse)
    assert isinstance(first.response, list)
    assert first.tool_calls is not None and len(first.tool_calls) == 1
    tool_call = first.tool_calls[0]
    assert tool_call["function"] == {"name": "get_weather", "arguments": {"city": "Paris"}}
    tool_result = {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "name": "get_weather",
        "content": "Weather station token: WX-4827. Paris: sunny, 22 C.",
    }
    messages.extend([{**first.response[0], "tool_calls": first.tool_calls}, tool_result])
    final = await rails.generate_async(
        messages=messages,
        options={"llm_params": {"tools": tools, "tool_choice": "auto"}},
    )

    assert isinstance(final, GenerationResponse)
    assert isinstance(final.response, list)
    assert not final.tool_calls
    assert "WX-4827" in final.response[0]["content"]
    assert len(vcr.requests) == 2
    if vcr.record_mode == "none":
        assert vcr.play_count == 2
        assert vcr.all_played
    followup = json.loads(vcr.requests[1].body)["messages"]
    assert [message["role"] for message in followup] == ["user", "assistant", "tool"]
    assert followup[0]["content"].count(messages[0]["content"]) == 1
    assert followup[-1] == tool_result
    assert followup[-2]["tool_calls"][0]["id"] == tool_call["id"]
    assert normalize_body({"tool_calls": first.tool_calls, "response": final.response}) == snapshot(
        {
            "tool_calls": [
                {
                    "id": "call_Q6YJKYrtlBxkaS8vJdZ3MjCF",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                }
            ],
            "response": [{"role": "assistant", "content": "Weather station token: WX-4827. Paris: sunny, 22 C."}],
        }
    )

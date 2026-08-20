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

"""Tool calling surfaced through the public ``generate_async`` API.

The tools schema is supplied per-call via ``options.llm_params`` (which are spread as
kwargs onto the provider request); ``tool_choice`` forces the call so the recorded
response is deterministic. The provider's tool call is surfaced on
``GenerationResponse.tool_calls`` by the generation_response assembly.
"""

from __future__ import annotations

import json

import pytest

from nemoguardrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationResponse
from tests.llm.clients._helpers import simulated_model
from tests.recorded.assertions import assert_request_payload
from tests.recorded.rails.public_api.configs import OPENAI_TOOLS_CONFIG, OPENAI_TOOLS_MODEL
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]

WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city to get the weather for."},
                },
                "required": ["city"],
            },
        },
    }
]

# Force the model to call get_weather so the recorded response is deterministic.
WEATHER_TOOL_CHOICE = {"type": "function", "function": {"name": "get_weather"}}


@pytest.mark.vcr
async def test_openai_generate_async_surfaces_tool_calls(openai_api_key, record_mode, recorded_cassette_path):
    rails = LLMRails(load_config(OPENAI_TOOLS_CONFIG), verbose=False)

    result = await rails.generate_async(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        options={"llm_params": {"tools": WEATHER_TOOLS, "tool_choice": WEATHER_TOOL_CHOICE}},
    )

    assert isinstance(result, GenerationResponse)
    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1

    # The tool_calls entries are surfaced in the provider-native shape, so assert the
    # surfaced call without assuming a key layout; the exact structure (incl. the frozen
    # tool-call id) is pinned by the snapshot below.
    serialized = json.dumps(result.tool_calls)
    assert "get_weather" in serialized
    assert "Paris" in serialized

    if record_mode == "none":
        # The tools schema is the behavioral fingerprint: it must reach the request body.
        assert_request_payload(
            recorded_cassette_path,
            model=OPENAI_TOOLS_MODEL,
            expected_params={"tools": WEATHER_TOOLS, "tool_choice": WEATHER_TOOL_CHOICE},
        )

    assert json.loads(serialized) == snapshot(
        [
            {
                "id": "call_qOIyw3cmenGpsFnt2SMSUgNG",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
            }
        ]
    )


async def test_tool_messages_reach_openai_request_body():
    """Assistant tool calls and tool results survive into the provider request body."""
    captured = []
    async with simulated_model("openai_generate_text.json", on_request=captured.append) as model:
        rails = LLMRails(load_config(OPENAI_TOOLS_CONFIG), llm=model, verbose=False)
        await rails.generate_async(
            messages=[
                {"role": "user", "content": "What is the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"temperature": 18}',
                    "tool_call_id": "call_weather",
                },
                {"role": "user", "content": "Summarize the result."},
            ]
        )

    assert len(captured) == 1
    payload = json.loads(captured[0].content)
    assistant = next(message for message in payload["messages"] if message["role"] == "assistant")
    tool = next(message for message in payload["messages"] if message["role"] == "tool")
    assert assistant["tool_calls"][0]["function"] == {
        "name": "get_weather",
        "arguments": '{"city": "Paris"}',
    }
    assert tool == {
        "role": "tool",
        "content": '{"temperature": 18}',
        "tool_call_id": "call_weather",
    }

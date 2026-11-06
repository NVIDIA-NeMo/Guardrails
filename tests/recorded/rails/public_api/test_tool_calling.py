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
from nemoguardrails.exceptions import LLMCallException
from nemoguardrails.rails.llm.options import GenerationResponse
from tests.recorded.cassette import normalize_body
from tests.recorded.rails.public_api.configs import (
    OPENAI_BASELINE_CONFIG,
    OPENAI_INVALID_MODEL_CONFIG,
    PASSTHROUGH_SANITIZED_HISTORY_CONFIG,
    TOOL_CALL_BLOCK_CONFIG,
    TOOL_ROUND_TRIP_RAILS_CONFIG,
)
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.vcr, pytest.mark.asyncio]


async def test_non_passthrough_tool_round_trip(openai_api_key, vcr):
    """Replay a string-prompt tool round trip and verify native roles and result-dependent output."""
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


def _tool(name, properties=None):
    """Build a function-tool schema with all supplied properties required and no extra properties."""
    properties = properties or {}
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def _assert_exchanges(vcr, count):
    """Check the recorded request count and require every interaction to be consumed during replay."""
    assert len(vcr.requests) == count
    if vcr.record_mode == "none":
        assert vcr.play_count == count
        assert vcr.all_played


async def test_parallel_tools_with_input_result_and_output_rails(openai_api_key, vcr):
    """Replay parallel calls with rewritten inputs, sanitized results, and output-rail formatting."""
    rails = LLMRails(load_config(TOOL_ROUND_TRIP_RAILS_CONFIG))
    tools = [_tool("get_paris"), _tool("get_berlin")]
    messages: list[dict] = [{"role": "user", "content": "PRIVATE_USER"}]
    first = await rails.generate_async(
        messages=messages,
        options={"llm_params": {"tools": tools, "tool_choice": "required", "parallel_tool_calls": True}},
    )
    assert isinstance(first, GenerationResponse)
    assert isinstance(first.response, list)
    assert first.tool_calls is not None
    assert {call["function"]["name"] for call in first.tool_calls} == {"get_paris", "get_berlin"}
    assert len(first.tool_calls) == 2
    assert first.response[0]["content"] == ""
    results = [
        {"role": "tool", "tool_call_id": call["id"], "content": "UNTRUSTED-" + call["function"]["name"]}
        for call in reversed(first.tool_calls)
    ]
    messages.extend([{**first.response[0], "tool_calls": first.tool_calls}, *results])
    final = await rails.generate_async(
        messages=messages, options={"llm_params": {"tools": tools, "tool_choice": "auto"}}
    )
    assert isinstance(final, GenerationResponse)
    assert isinstance(final.response, list)
    assert not final.tool_calls
    assert final.response[0]["content"].startswith("Checked: ")
    assert all(token in final.response[0]["content"] for token in ("PARIS-4827", "BERLIN-6912"))
    messages.extend(
        [
            final.response[0],
            {"role": "user", "content": "Repeat both station tokens from earlier results. Do not call any tools."},
        ]
    )
    later = await rails.generate_async(
        messages=messages, options={"llm_params": {"tools": tools, "tool_choice": "auto"}}
    )
    assert isinstance(later, GenerationResponse)
    assert isinstance(later.response, list)
    assert not later.tool_calls
    assert all(token in later.response[0]["content"] for token in ("PARIS-4827", "BERLIN-6912"))
    _assert_exchanges(vcr, 3)
    for request in vcr.requests:
        sent = json.loads(request.body)["messages"]
        assert "PRIVATE_USER" not in json.dumps(sent)
        assert "UNTRUSTED" not in json.dumps(sent)
    sent_results = [message for message in json.loads(vcr.requests[1].body)["messages"] if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in sent_results] == [message["tool_call_id"] for message in results]
    assert {message["content"] for message in sent_results} == {"PARIS-4827: sunny", "BERLIN-6912: rainy"}
    assert normalize_body({"calls": first.tool_calls, "answer": final.response, "later": later.response}) == snapshot(
        {
            "calls": [
                {
                    "id": "call_Q7TqZIsJermIKQ1WuAZVsbUM",
                    "type": "function",
                    "function": {"name": "get_paris", "arguments": {}},
                },
                {
                    "id": "call_Qx1rJ4AAmpBzrRcnJGkkIvns",
                    "type": "function",
                    "function": {"name": "get_berlin", "arguments": {}},
                },
            ],
            "answer": [
                {
                    "role": "assistant",
                    "content": """\
Checked: Paris weather result: "PARIS-4827: sunny"

Berlin weather result: "BERLIN-6912: rainy"\
""",
                }
            ],
            "later": [
                {
                    "role": "assistant",
                    "content": """\
Checked: Paris token: **PARIS-4827: sunny**  \n\
Berlin token: **BERLIN-6912: rainy**\
""",
                }
            ],
        }
    )


async def test_sequential_dependent_tool_rounds(openai_api_key, vcr):
    """Replay dependent tool rounds and verify earlier results remain in subsequent provider requests."""
    config = load_config(OPENAI_BASELINE_CONFIG)
    config.passthrough = False
    rails = LLMRails(config)
    tools = [
        _tool("get_weather", {"city": {"type": "string", "enum": ["Paris"]}}),
        _tool("get_advice", {"weather_token": {"type": "string"}}),
    ]
    messages: list[dict] = [
        {
            "role": "user",
            "content": "Get the Paris weather, pass its weather_token to get_advice, then quote both tool results exactly.",
        }
    ]
    outputs = []
    for name, result in [
        ("get_weather", "Weather weather_token: WX-4827. Paris: sunny."),
        ("get_advice", "ADVICE-6912: bring sunglasses."),
    ]:
        response = await rails.generate_async(
            messages=messages,
            options={
                "llm_params": {
                    "tools": tools,
                    "tool_choice": {"type": "function", "function": {"name": name}},
                    "parallel_tool_calls": False,
                }
            },
        )
        assert isinstance(response, GenerationResponse)
        assert isinstance(response.response, list)
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        call = response.tool_calls[0]
        assert call["function"]["name"] == name
        if name == "get_advice":
            assert call["function"]["arguments"] == {"weather_token": "WX-4827"}
        outputs.append(response.tool_calls)
        messages.extend(
            [
                {**response.response[0], "tool_calls": response.tool_calls},
                {"role": "tool", "tool_call_id": call["id"], "content": result},
            ]
        )
    final = await rails.generate_async(
        messages=messages, options={"llm_params": {"tools": tools, "tool_choice": "auto"}}
    )
    assert isinstance(final, GenerationResponse)
    assert isinstance(final.response, list)
    assert not final.tool_calls
    assert all(token in final.response[0]["content"] for token in ("WX-4827", "ADVICE-6912"))
    _assert_exchanges(vcr, 3)
    sent = json.loads(vcr.requests[-1].body)["messages"]
    assert [message["role"] for message in sent] == ["user", "assistant", "tool", "assistant", "tool"]
    assert normalize_body({"calls": outputs, "answer": final.response}) == snapshot(
        {
            "calls": [
                [
                    {
                        "id": "call_a7OunQMdySJjyeikWcTAnqWM",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                    }
                ],
                [
                    {
                        "id": "call_1fvEaAd9WrbOhkAU9onz1sVn",
                        "type": "function",
                        "function": {"name": "get_advice", "arguments": {"weather_token": "WX-4827"}},
                    }
                ],
            ],
            "answer": [
                {
                    "role": "assistant",
                    "content": """\
Weather tool result (quoted exactly):  \n\
Weather weather_token: WX-4827. Paris: sunny.

Advice tool result (quoted exactly):  \n\
ADVICE-6912: bring sunglasses.\
""",
                }
            ],
        }
    )


async def test_provider_tool_call_rejected_by_rail_is_not_returned(openai_api_key, vcr):
    """Verify a real recorded provider call is replaced by a refusal when its tool-call rail blocks it."""
    rails = LLMRails(load_config(TOOL_CALL_BLOCK_CONFIG))
    response = await rails.generate_async(
        messages=[{"role": "user", "content": "Get the weather."}],
        options={"llm_params": {"tools": [_tool("get_weather")], "tool_choice": "required"}},
    )
    _assert_exchanges(vcr, 1)
    provider = json.loads(vcr.responses[0]["body"]["string"])
    assert provider["choices"][0]["message"]["tool_calls"]
    assert isinstance(response, GenerationResponse)
    assert isinstance(response.response, list)
    assert not response.tool_calls
    assert normalize_body(response.response) == snapshot([{"role": "assistant", "content": "Tool call rejected."}])


async def test_tool_followup_invalid_model_raises(openai_api_key, vcr):
    """Replay a provider 404 on a tool follow-up and verify the library preserves the underlying error."""
    config = load_config(OPENAI_INVALID_MODEL_CONFIG)
    config.passthrough = False
    rails = LLMRails(config)
    with pytest.raises(LLMCallException) as error:
        await rails.generate_async(
            messages=[
                {"role": "user", "content": "Get the weather."},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_weather", "content": "WX-4827: sunny"},
            ]
        )
    assert getattr(error.value.inner_exception, "status_code", None) == 404
    _assert_exchanges(vcr, 1)
    assert [message["role"] for message in json.loads(vcr.requests[0].body)["messages"]] == [
        "user",
        "assistant",
        "tool",
    ]


async def test_passthrough_retains_sanitized_user_history(openai_api_key, vcr):
    """Replay two passthrough turns using sanitized history while leaving caller messages unchanged."""
    rails = LLMRails(load_config(PASSTHROUGH_SANITIZED_HISTORY_CONFIG))
    messages = [{"role": "user", "content": "PRIVATE_USER"}]
    first = await rails.generate_async(messages=messages)
    assert isinstance(first, dict)
    assert messages == [{"role": "user", "content": "PRIVATE_USER"}]
    messages.extend([first, {"role": "user", "content": "What is the fictional weather station named?"}])
    final = await rails.generate_async(messages=messages)
    assert isinstance(final, dict)
    assert "MAPLE-4827" in final["content"]
    assert messages[0] == {"role": "user", "content": "PRIVATE_USER"}
    _assert_exchanges(vcr, 2)
    for request in vcr.requests:
        sent = json.loads(request.body)["messages"]
        assert sent[0] == {
            "role": "user",
            "content": "The fictional weather station is named MAPLE-4827. Reply exactly: noted",
        }
        assert "PRIVATE_USER" not in json.dumps(sent)
    assert normalize_body({"first": first, "final": final}) == snapshot(
        {"first": {"role": "assistant", "content": "noted"}, "final": {"role": "assistant", "content": "MAPLE-4827"}}
    )

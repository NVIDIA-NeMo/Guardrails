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

"""Real provider completions delivered as checked, delayed server tool SSE."""

import json
from copy import deepcopy

import httpx
import pytest

from nemoguardrails.server import api
from nemoguardrails.server.datastore.memory_store import MemoryStore
from tests.recorded.cassette import decode_body_json
from tests.recorded.rails.public_api.configs import (
    OPENAI_INVALID_MODEL,
    TOOL_CALL_BLOCK_CONFIG,
    TOOL_ROUND_TRIP_RAILS_CONFIG,
    WEATHER_TOOL,
    WEATHER_TOOL_CHOICE,
)
from tests.recorded.rails_config import load_config
from tests.recorded.server.buffered_tool_streaming import (
    ToolStreamRecorder,
    _assert_provider_calls,
    _backend,
    _calls,
    _config,
    _request,
    _wire_requests,
)
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.vcr(ignore_hosts=["testserver"]), pytest.mark.asyncio]
ENGINES = ["llmrails", "llmrails-chat", "iorails"]


@pytest.mark.parametrize("provider", ["openai", "nim"])
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("parallel", [False, True], ids=["single-stateless", "parallel-thread"])
async def test_sdk_tool_round_trip(openai_api_key, nvidia_inference_key, monkeypatch, vcr, provider, engine, parallel):
    tool_stream = ToolStreamRecorder()
    config = _config(provider, nvidia_inference_key)
    store = MemoryStore()
    monkeypatch.setattr(api, "datastore", store)
    cities = ["Paris", "Berlin"] if parallel else ["Paris"]
    messages = [
        {
            "role": "user",
            "content": f"Call get_weather once for each city: {', '.join(cities)}. After tools, quote their results exactly.",
        }
    ]
    request = _request(config, messages, parallel_tool_calls=parallel)
    if parallel:
        request["guardrails"]["thread_id"] = "recorded-buffered-tools"
    choice = ("auto" if provider == "nim" else "required") if parallel else WEATHER_TOOL_CHOICE
    if parallel and provider == "nim":
        request["max_tokens"] = 1024
    original = deepcopy(messages)
    async with _backend(config, engine, monkeypatch):
        first = await tool_stream({**request, "tool_choice": choice})
        _assert_provider_calls(vcr, 0, first)
        calls = first["tool_calls"]
        assert sorted(json.loads(call["function"]["arguments"])["city"] for call in calls) == sorted(cities)
        assert len({call["id"] for call in calls}) == len(cities)
        results = [
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": f"{json.loads(call['function']['arguments'])['city']}: WX-4827, sunny.",
            }
            for call in reversed(calls)
        ]
        followup_messages = results if parallel else [*messages, first, *results]
        final = await tool_stream({**request, "messages": followup_messages, "tool_choice": "auto"})
        assert not final.get("tool_calls")
        assert "sunny" in final["content"].lower()
        provider_answer = decode_body_json(vcr.responses[1]["body"])["choices"][0]["message"]["content"]
        assert final["content"].strip() == provider_answer.strip()
        later = None
        if parallel:
            later = await tool_stream(
                {
                    **request,
                    "messages": [{"role": "user", "content": "Repeat only the weather token from earlier, no tools."}],
                    "tool_choice": "auto",
                }
            )
            assert not later.get("tool_calls") and "WX-4827" in later["content"]
    assert messages == original
    wire = _wire_requests(vcr, 3 if parallel else 2)
    assert wire[0]["tools"] == [WEATHER_TOOL]
    assert wire[0]["parallel_tool_calls"] is parallel
    assert wire[0]["tool_choice"] == choice
    assert wire[1]["tool_choice"] == "auto"
    sent = wire[1]["messages"]
    assert sent[-len(results) :] == results
    assert sent[-len(results) - 1]["tool_calls"] == calls
    if parallel:
        stored_json = await store.get("thread-recorded-buffered-tools")
        assert stored_json is not None
        stored = json.loads(stored_json)
        assert stored[1]["tool_calls"] == calls
        assert stored[2 : 2 + len(results)] == results
        history = [message for message in wire[2]["messages"] if message["role"] != "system"]
        assert history[1]["tool_calls"] == calls
        assert history[2 : 2 + len(results)] == results
    assert {"calls": _calls(first), "answer": final["content"], "later": later} == snapshot(
        {
            "openai/llmrails/False": {
                "calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
                "answer": "Paris: WX-4827, sunny.",
                "later": None,
            },
            "nim/llmrails/False": {
                "calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
                "answer": "Quote: “Paris: WX-4827, sunny.”",
                "later": None,
            },
            "openai/llmrails-chat/False": {
                "calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
                "answer": "Paris: WX-4827, sunny.",
                "later": None,
            },
            "nim/llmrails-chat/False": {
                "calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
                "answer": """\
The current weather in **Paris** is:  \n\
- **Condition**: sunny  \n\

Would you like to check another city or need additional details about Paris's weather? 😊\
""",
                "later": None,
            },
            "openai/iorails/False": {
                "calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
                "answer": "Paris: WX-4827, sunny.",
                "later": None,
            },
            "nim/iorails/False": {
                "calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
                "answer": "The weather in Paris is **sunny** with the condition code **WX-4827**.",
                "later": None,
            },
            "openai/llmrails/True": {
                "calls": [
                    {"name": "get_weather", "arguments": {"city": "Paris"}},
                    {"name": "get_weather", "arguments": {"city": "Berlin"}},
                ],
                "answer": """\
"Paris: WX-4827, sunny."
"Berlin: WX-4827, sunny."\
""",
                "later": {"content": "WX-4827, sunny.", "role": "assistant"},
            },
            "nim/llmrails/True": {
                "calls": [
                    {"name": "get_weather", "arguments": {"city": "Paris"}},
                    {"name": "get_weather", "arguments": {"city": "Berlin"}},
                ],
                "answer": """\
Berlin: WX-4827, sunny.  \n\
Paris: WX-4827, sunny.\
""",
                "later": {"content": "WX-4827", "role": "assistant"},
            },
            "openai/llmrails-chat/True": {
                "calls": [
                    {"name": "get_weather", "arguments": {"city": "Paris"}},
                    {"name": "get_weather", "arguments": {"city": "Berlin"}},
                ],
                "answer": """\
Paris: WX-4827, sunny.  \n\
Berlin: WX-4827, sunny.\
""",
                "later": {"content": "WX-4827, sunny.", "role": "assistant"},
            },
            "nim/llmrails-chat/True": {
                "calls": [
                    {"name": "get_weather", "arguments": {"city": "Paris"}},
                    {"name": "get_weather", "arguments": {"city": "Berlin"}},
                ],
                "answer": """\
- **Paris**: WX-4827, sunny.  \n\
- **Berlin**: WX-4827, sunny.\
""",
                "later": {"content": "WX-4827", "role": "assistant"},
            },
            "openai/iorails/True": {
                "calls": [
                    {"name": "get_weather", "arguments": {"city": "Paris"}},
                    {"name": "get_weather", "arguments": {"city": "Berlin"}},
                ],
                "answer": """\
“Paris: WX-4827, sunny.”  \n\
“Berlin: WX-4827, sunny.”\
""",
                "later": {"content": "WX-4827, sunny.", "role": "assistant"},
            },
            "nim/iorails/True": {
                "calls": [
                    {"name": "get_weather", "arguments": {"city": "Paris"}},
                    {"name": "get_weather", "arguments": {"city": "Berlin"}},
                ],
                "answer": """\
Berlin: WX-4827, sunny.  \n\
Paris: WX-4827, sunny.\
""",
                "later": {"content": "WX-4827", "role": "assistant"},
            },
        }
    )[f"{provider}/{engine}/{parallel}"]
    assert (
        tool_stream.events
        == snapshot(
            {
                "openai/llmrails/False": [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\": \\"Paris\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Paris: WX-4827, sunny."}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                ],
                "nim/llmrails/False": [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\": \\"Paris\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Quote: \\u201cParis: WX-4827, sunny.\\u201d"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                ],
                "openai/llmrails-chat/False": [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\": \\"Paris\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Paris: WX-4827, sunny."}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                ],
                "nim/llmrails-chat/False": [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\": \\"Paris\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "The current weather in **Paris**"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": " is:  \\n- **Condition**: sunny  \\n"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "\\nWould you like to check another"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": " city or need additional details"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": " about Paris's weather? \\ud83d\\ude0a"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                ],
                "openai/iorails/False": [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\": \\"Paris\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Paris: WX-4827, sunny."}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                ],
                "nim/iorails/False": [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\": \\"Paris\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "The weather in Paris is **sunny*"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "* with the condition code **WX-4"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "827**."}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                ],
                "openai/llmrails/True": [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\": \\"Paris\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_1]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 1}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "{\\"city\\": \\"Berlin\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "\\"Paris: WX-4827, sunny.\\"\\n\\"Berlin"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": ": WX-4827, sunny.\\""}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "WX-4827, sunny."}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                ],
                "nim/llmrails/True": [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\": \\"Paris\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_1]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 1}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "{\\"city\\": \\"Berlin\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Berlin: WX-4827, sunny.  \\nParis:"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": " WX-4827, sunny."}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "WX-4827"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                ],
                "openai/llmrails-chat/True": [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\": \\"Paris\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_1]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 1}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "{\\"city\\": \\"Berlin\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Paris: WX-4827, sunny.  \\nBerlin:"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": " WX-4827, sunny."}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "WX-4827, sunny."}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                ],
                "nim/llmrails-chat/True": [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\": \\"Paris\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_1]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 1}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "{\\"city\\": \\"Berlin\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "- **Paris**: WX-4827, sunny.  \\n-"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": " **Berlin**: WX-4827, sunny."}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "WX-4827"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                ],
                "openai/iorails/True": [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\": \\"Paris\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_1]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 1}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "{\\"city\\": \\"Berlin\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "\\u201cParis: WX-4827, sunny.\\u201d  \\n\\u201cBerl"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "in: WX-4827, sunny.\\u201d"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "WX-4827, sunny."}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                ],
                "nim/iorails/True": [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\": \\"Paris\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_1]", "function": {"name": "get_weather", "arguments": ""}, "type": "function", "index": 1}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "{\\"city\\": \\"Berlin\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Berlin: WX-4827, sunny.  \\nParis:"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": " WX-4827, sunny."}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "WX-4827"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "nvidia/nvidia/nemotron-3-nano-30b-a3b", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                ],
            }
        )[f"{provider}/{engine}/{parallel}"]
    )


@pytest.mark.parametrize("engine", ENGINES)
async def test_sequential_dependent_tools(openai_api_key, monkeypatch, vcr, engine):
    tool_stream = ToolStreamRecorder()
    config = _config("openai", None)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_token",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"token": {"type": "string"}},
                    "required": ["token"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    messages = [
        {
            "role": "user",
            "content": "Look up a token, use that exact token to look up weather, then quote both results.",
        }
    ]
    request = _request(config, messages, tools=tools, parallel_tool_calls=False)
    outputs = []
    async with _backend(config, engine, monkeypatch):
        for index, name in enumerate(["lookup_token", "lookup_weather"]):
            message = await tool_stream({**request, "tool_choice": {"type": "function", "function": {"name": name}}})
            _assert_provider_calls(vcr, index, message)
            (call,) = message["tool_calls"]
            assert call["function"]["name"] == name
            if index:
                assert json.loads(call["function"]["arguments"]) == {"token": "DEP-6912"}
            outputs.append(_calls(message))
            messages.extend(
                [
                    message,
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": "DEP-6912" if index == 0 else "WX-4827: sunny",
                    },
                ]
            )
        final = await tool_stream({**request, "tool_choice": "auto"})
    assert "DEP-6912" in final["content"] and "WX-4827" in final["content"]
    wire = _wire_requests(vcr, 3)
    assert wire[2]["messages"][-4:] == messages[-4:]
    assert {"calls": outputs, "answer": final} == snapshot(
        {
            "llmrails": {
                "calls": [
                    [{"name": "lookup_token", "arguments": {}}],
                    [{"name": "lookup_weather", "arguments": {"token": "DEP-6912"}}],
                ],
                "answer": {
                    "content": """\
Token result: **DEP-6912**

Weather result: **WX-4827: sunny**\
""",
                    "role": "assistant",
                },
            },
            "llmrails-chat": {
                "calls": [
                    [{"name": "lookup_token", "arguments": {}}],
                    [{"name": "lookup_weather", "arguments": {"token": "DEP-6912"}}],
                ],
                "answer": {
                    "content": """\
Token result: **"DEP-6912"**

Weather result: **"WX-4827: sunny"**\
""",
                    "role": "assistant",
                },
            },
            "iorails": {
                "calls": [
                    [{"name": "lookup_token", "arguments": {}}],
                    [{"name": "lookup_weather", "arguments": {"token": "DEP-6912"}}],
                ],
                "answer": {
                    "content": """\
Token result: **DEP-6912**

Weather result: **WX-4827: sunny**\
""",
                    "role": "assistant",
                },
            },
        }
    )[engine]
    assert (
        tool_stream.events
        == snapshot(
            {
                "llmrails": [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "lookup_token", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "lookup_weather", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"token\\": \\"DEP-6912\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Token result: **DEP-6912**\\n\\nWeat"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "her result: **WX-4827: sunny**"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                ],
                "llmrails-chat": [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "lookup_token", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "lookup_weather", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"token\\": \\"DEP-6912\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Token result: **\\"DEP-6912\\"**\\n\\nWe"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "ather result: **\\"WX-4827: sunny\\""}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "**"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                ],
                "iorails": [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "lookup_token", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "lookup_weather", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"token\\": \\"DEP-6912\\"}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Token result: **DEP-6912**\\n\\nWeat"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "her result: **WX-4827: sunny**"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                ],
            }
        )[engine]
    )


@pytest.mark.parametrize("passthrough", [False, True])
async def test_tool_call_rail_blocks_provider_call(openai_api_key, monkeypatch, vcr, passthrough):
    tool_stream = ToolStreamRecorder()
    config = load_config(TOOL_CALL_BLOCK_CONFIG)
    async with _backend(config, "llmrails" if passthrough else "llmrails-chat", monkeypatch):
        result = await tool_stream(
            _request(
                config, [{"role": "user", "content": "Call get_weather for Paris."}], tool_choice=WEATHER_TOOL_CHOICE
            )
        )
    _wire_requests(vcr, 1)
    emitted = decode_body_json(vcr.responses[0]["body"])["choices"][0]["message"]["tool_calls"]
    assert len(emitted) == 1
    assert result == snapshot({"role": "assistant", "content": "Tool call rejected."})
    assert emitted[0]["id"] not in json.dumps(result)
    assert (
        tool_stream.events
        == snapshot(
            {
                False: [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Tool call rejected."}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

"""
                ],
                True: [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Tool call rejected."}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

"""
                ],
            }
        )[passthrough]
    )


@pytest.mark.parametrize("passthrough", [False, True])
async def test_input_result_output_rails_sanitize_stream_and_history(openai_api_key, monkeypatch, vcr, passthrough):
    tool_stream = ToolStreamRecorder()
    config = load_config(TOOL_ROUND_TRIP_RAILS_CONFIG)
    if passthrough:
        config.rails.dialog.single_call.enabled = False
    store = MemoryStore()
    monkeypatch.setattr(api, "datastore", store)
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }
        for name in ["get_paris", "get_berlin"]
    ]
    request = _request(config, [{"role": "user", "content": "PRIVATE_USER"}], tools=tools, parallel_tool_calls=True)
    request["guardrails"]["thread_id"] = "recorded-sanitized-tools"
    async with _backend(config, "llmrails" if passthrough else "llmrails-chat", monkeypatch):
        first = await tool_stream({**request, "tool_choice": "required"})
        _assert_provider_calls(vcr, 0, first)
        assert {call["name"] for call in _calls(first)} == {"get_paris", "get_berlin"}
        results = [
            {"role": "tool", "tool_call_id": call["id"], "content": "PRIVATE_RESULT"}
            for call in reversed(first["tool_calls"])
        ]
        original = deepcopy(results)
        final = await tool_stream({**request, "messages": results, "tool_choice": "auto"})
        assert final["content"].startswith("Checked: ")
        assert "PARIS-4827" in final["content"] and "BERLIN-6912" in final["content"]
        later = await tool_stream(
            {
                **request,
                "messages": [{"role": "user", "content": "Repeat both weather tokens from earlier."}],
                "tool_choice": "auto",
            }
        )
    wire = _wire_requests(vcr, 3)
    assert results == original
    stored_json = await store.get("thread-recorded-sanitized-tools")
    assert stored_json is not None
    stored = json.loads(stored_json)
    assert "PRIVATE_USER" not in json.dumps(wire)
    # The server persists caller history; rails re-sanitize it on later turns.
    assert stored[0]["content"] == "PRIVATE_USER"
    assert stored[2:4] == original
    assert "PRIVATE_RESULT" not in json.dumps(wire) + json.dumps(final)
    assert {message["content"] for message in wire[1]["messages"] if message["role"] == "tool"} == {
        "PARIS-4827: sunny",
        "BERLIN-6912: rainy",
    }
    assert "PARIS-4827" in later["content"] and "BERLIN-6912" in later["content"]
    assert {"calls": _calls(first), "answer": final, "later": later} == snapshot(
        {
            False: {
                "calls": [{"name": "get_paris", "arguments": {}}, {"name": "get_berlin", "arguments": {}}],
                "answer": {
                    "content": """\
Checked: PARIS result: `"PARIS-4827: sunny"`  \n\

BERLIN result: `"BERLIN-6912: rainy"`\
""",
                    "role": "assistant",
                },
                "later": {
                    "content": """\
Checked: PARIS-4827: sunny"  \n\

"BERLIN-6912: rainy\
""",
                    "role": "assistant",
                },
            },
            True: {
                "calls": [{"name": "get_paris", "arguments": {}}, {"name": "get_berlin", "arguments": {}}],
                "answer": {
                    "content": """\
Checked: PARIS-4827: sunny

BERLIN-6912: rainy\
""",
                    "role": "assistant",
                },
                "later": {
                    "content": """\
Checked: PARIS-4827: sunny

BERLIN-6912: rainy\
""",
                    "role": "assistant",
                },
            },
        }
    )[passthrough]
    assert (
        tool_stream.events
        == snapshot(
            {
                False: [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "get_paris", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_1]", "function": {"name": "get_berlin", "arguments": ""}, "type": "function", "index": 1}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "{}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Checked: PARIS result: `\\"PARIS-4"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "827: sunny\\"`  \\n\\nBERLIN result: `"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "\\"BERLIN-6912: rainy\\"`"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Checked: PARIS-4827: sunny\\"  \\n\\n\\""}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "BERLIN-6912: rainy"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                ],
                True: [
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_0]", "function": {"name": "get_paris", "arguments": ""}, "type": "function", "index": 0}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"id": "[TOOL_CALL_1]", "function": {"name": "get_berlin", "arguments": ""}, "type": "function", "index": 1}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "{}"}}]}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Checked: PARIS-4827: sunny\\n\\nBERL"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "IN-6912: rainy"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                    """\
data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "guardrails": {"config_id": "recorded-buffered-tools"}, "choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant"}}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Checked: PARIS-4827: sunny\\n\\nBERL"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "IN-6912: rainy"}, "finish_reason": null}]}

data: {"id": "[COMPLETION_ID]", "created": 0, "model": "gpt-5.4-nano", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]

""",
                ],
            }
        )[passthrough]
    )


@pytest.mark.parametrize("provider", ["openai", "nim"])
@pytest.mark.parametrize("engine", ENGINES)
async def test_tool_followup_invalid_model_fails_closed(
    openai_api_key, nvidia_inference_key, monkeypatch, vcr, provider, engine
):
    config = _config(provider, nvidia_inference_key)
    config.models[0].model = OPENAI_INVALID_MODEL
    messages = [
        {"role": "user", "content": "Get Paris weather."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_weather", "content": "WX-4827: sunny"},
    ]
    async with _backend(config, engine, monkeypatch):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://testserver"
        ) as client:
            response = await client.post("/v1/chat/completions", json=_request(config, messages, tool_choice="none"))
    assert response.status_code == (404 if provider == "openai" else 403)
    assert response.headers["content-type"].startswith("application/json")
    assert set(response.json()) == {"error"}
    _wire_requests(vcr, 1)
    assert (
        response.json()
        == snapshot(
            {
                "openai/llmrails": {
                    "error": {
                        "message": "The model `gpt-nonexistent-test-model` does not exist or you do not have access to it.",
                        "type": "not_found_error",
                        "param": None,
                        "code": "model_not_found",
                    }
                },
                "nim/llmrails": {
                    "error": {
                        "message": "The requested model 'gpt-nonexistent-test-model' is not available for this API key, or the model name is invalid. Check the models available to you and try again.",
                        "type": "permission_error",
                        "param": "model",
                        "code": "403",
                    }
                },
                "openai/llmrails-chat": {
                    "error": {
                        "message": "The model `gpt-nonexistent-test-model` does not exist or you do not have access to it.",
                        "type": "not_found_error",
                        "param": None,
                        "code": "model_not_found",
                    }
                },
                "nim/llmrails-chat": {
                    "error": {
                        "message": "The requested model 'gpt-nonexistent-test-model' is not available for this API key, or the model name is invalid. Check the models available to you and try again.",
                        "type": "permission_error",
                        "param": "model",
                        "code": "403",
                    }
                },
                "openai/iorails": {
                    "error": {
                        "message": "The model `gpt-nonexistent-test-model` does not exist or you do not have access to it.",
                        "type": "not_found_error",
                        "param": None,
                        "code": "model_not_found",
                    }
                },
                "nim/iorails": {
                    "error": {
                        "message": "The requested model 'gpt-nonexistent-test-model' is not available for this API key, or the model name is invalid. Check the models available to you and try again.",
                        "type": "permission_error",
                        "param": "model",
                        "code": "403",
                    }
                },
            }
        )[f"{provider}/{engine}"]
    )

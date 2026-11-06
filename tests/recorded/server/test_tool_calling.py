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
from unittest.mock import AsyncMock

import httpx
import pytest

from nemoguardrails import LLMRails
from nemoguardrails.rails.llm.config import MessageTemplate, TaskPrompt
from nemoguardrails.server import api
from tests.recorded.rails.public_api.configs import OPENAI_BASELINE_CONFIG, OPENAI_MODEL
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.vcr(ignore_hosts=["testserver"]), pytest.mark.asyncio]


async def test_non_passthrough_chat_tool_round_trip(openai_api_key, monkeypatch, vcr):
    """Replay a server tool round trip with a chat prompt and verify the response and provider history."""
    config = load_config(OPENAI_BASELINE_CONFIG)
    config.passthrough = False
    config.prompts = [
        TaskPrompt(
            task="general",
            messages=[
                MessageTemplate(type="system", content="{{ general_instructions }}"),
                "{{ history | to_chat_messages }}",
            ],
        )
    ]
    rails = LLMRails(config)
    monkeypatch.setattr(api, "_get_rails", AsyncMock(return_value=rails))
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
    request = {"model": OPENAI_MODEL, "guardrails": {"config_id": "recorded"}, "tools": tools}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://testserver") as client:
        first = await client.post(
            "/v1/chat/completions",
            json={**request, "messages": messages, "tool_choice": "required", "parallel_tool_calls": False},
        )
        assert first.status_code == 200
        first_choice = first.json()["choices"][0]
        assert first_choice["finish_reason"] == "tool_calls"
        assistant = first_choice["message"]
        assert len(assistant["tool_calls"]) == 1
        tool_call = assistant["tool_calls"][0]
        assert tool_call["function"]["name"] == "get_weather"
        assert json.loads(tool_call["function"]["arguments"]) == {"city": "Paris"}
        tool_result = {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": "Weather station token: WX-4827. Paris: sunny, 22 C.",
        }
        messages.extend([assistant, tool_result])
        final = await client.post("/v1/chat/completions", json={**request, "messages": messages, "tool_choice": "auto"})

    assert final.status_code == 200
    final_choice = final.json()["choices"][0]
    assert final_choice["finish_reason"] == "stop"
    assert "WX-4827" in final_choice["message"]["content"]
    assert not final_choice["message"].get("tool_calls")
    assert len(vcr.requests) == 2
    if vcr.record_mode == "none":
        assert vcr.play_count == 2
        assert vcr.all_played
    followup = json.loads(vcr.requests[1].body)["messages"]
    assert [message["role"] for message in followup] == ["system", "user", "assistant", "tool"]
    assert followup[-1] == tool_result
    assert followup[-2]["tool_calls"] == assistant["tool_calls"]
    assert {"first": first_choice, "final": final_choice} == snapshot(
        {
            "first": {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_e1PDmXTwImG05kqcK58GtVRb",
                            "function": {"arguments": '{"city": "Paris"}', "name": "get_weather"},
                            "type": "function",
                        }
                    ],
                },
            },
            "final": {
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": "Weather station token: WX-4827. Paris: sunny, 22 C.", "role": "assistant"},
            },
        }
    )


async def test_tool_thread_replays_stored_history(openai_api_key, monkeypatch, vcr):
    """Replay a server thread continuation and verify stored calls and results reach the provider."""
    from nemoguardrails.server.datastore.memory_store import MemoryStore

    config = load_config(OPENAI_BASELINE_CONFIG)
    config.passthrough = False
    rails = LLMRails(config)
    store = MemoryStore()
    monkeypatch.setattr(api, "_get_rails", AsyncMock(return_value=rails))
    monkeypatch.setattr(api, "datastore", store)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }
    ]
    request = {
        "model": OPENAI_MODEL,
        "guardrails": {"config_id": "recorded", "thread_id": "recorded-tool-thread"},
        "tools": tools,
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://testserver") as client:
        first = await client.post(
            "/v1/chat/completions",
            json={
                **request,
                "messages": [{"role": "user", "content": "Get the weather, then quote the tool result exactly."}],
                "tool_choice": "required",
            },
        )
        assert first.status_code == 200
        first_choice = first.json()["choices"][0]
        assert first_choice["finish_reason"] == "tool_calls"
        calls = first_choice["message"]["tool_calls"]
        assert len(calls) == 1
        result = {"role": "tool", "tool_call_id": calls[0]["id"], "content": "WX-4827: sunny, 22 C."}
        second = await client.post(
            "/v1/chat/completions", json={**request, "messages": [result], "tool_choice": "auto"}
        )
        assert second.status_code == 200
        second_choice = second.json()["choices"][0]
        assert second_choice["finish_reason"] == "stop"
        assert "WX-4827" in second_choice["message"]["content"]
        third = await client.post(
            "/v1/chat/completions",
            json={
                **request,
                "messages": [
                    {"role": "user", "content": "Repeat the weather station token from earlier. Do not call tools."}
                ],
                "tool_choice": "auto",
            },
        )
        assert third.status_code == 200
    third_choice = third.json()["choices"][0]
    assert third_choice["finish_reason"] == "stop"
    assert "WX-4827" in third_choice["message"]["content"]
    assert len(vcr.requests) == 3
    if vcr.record_mode == "none":
        assert vcr.play_count == 3 and vcr.all_played
    for wire_request in vcr.requests[1:]:
        sent = json.loads(wire_request.body)["messages"]
        assert sent[1]["tool_calls"] == calls
        assert sent[2] == result
    stored_json = await store.get("thread-recorded-tool-thread")
    assert stored_json is not None
    stored = json.loads(stored_json)
    assert stored[1]["tool_calls"] == calls
    assert stored[2] == result
    assert {"first": first_choice, "second": second_choice, "third": third_choice} == snapshot(
        {
            "first": {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_SW3DPBL65CwvN4PKyA82S6vQ",
                            "function": {"arguments": "{}", "name": "get_weather"},
                            "type": "function",
                        }
                    ],
                },
            },
            "second": {
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": "WX-4827: sunny, 22 C.", "role": "assistant"},
            },
            "third": {"finish_reason": "stop", "index": 0, "message": {"content": "WX-4827", "role": "assistant"}},
        }
    )


async def test_tool_followup_provider_error_uses_server_error_envelope(openai_api_key, monkeypatch, vcr):
    """Replay a failed provider follow-up and verify the server returns its standard error envelope."""
    from tests.recorded.rails.public_api.configs import OPENAI_INVALID_MODEL_CONFIG

    config = load_config(OPENAI_INVALID_MODEL_CONFIG)
    config.passthrough = False
    rails = LLMRails(config)
    monkeypatch.setattr(api, "_get_rails", AsyncMock(return_value=rails))
    messages = [
        {"role": "user", "content": "Get the weather."},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_weather", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_weather", "content": "WX-4827: sunny"},
    ]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://testserver") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": OPENAI_MODEL, "guardrails": {"config_id": "recorded"}, "messages": messages},
        )
    assert response.status_code == 404
    assert set(response.json()) == {"error"}
    assert len(vcr.requests) == 1
    if vcr.record_mode == "none":
        assert vcr.play_count == 1 and vcr.all_played
    assert response.json() == snapshot(
        {
            "error": {
                "message": "The model `gpt-nonexistent-test-model` does not exist or you do not have access to it.",
                "type": "not_found_error",
                "param": None,
                "code": "model_not_found",
            }
        }
    )

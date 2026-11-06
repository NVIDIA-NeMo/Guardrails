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

import httpx
import pytest
from openai.types.chat import ChatCompletionChunk

from nemoguardrails import LLMRails
from nemoguardrails.server import api
from tests.recorded.assertions import assert_request_payload
from tests.recorded.rails.public_api.configs import (
    OPENAI_BASELINE_CONFIG,
    OPENAI_MODEL,
    WEATHER_TOOL,
    WEATHER_TOOL_CHOICE,
)
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.vcr(ignore_hosts=["testserver"]), pytest.mark.asyncio]


async def test_backend_tool_calls_stream_as_openai_sse(
    openai_api_key, monkeypatch, record_mode, recorded_cassette_path, vcr
):
    config = load_config(OPENAI_BASELINE_CONFIG)
    config.models[0].parameters.update(
        {
            "tools": [WEATHER_TOOL],
            "tool_choice": WEATHER_TOOL_CHOICE,
            "stream_options": {"include_usage": True},
            "max_tokens": 1024,
        }
    )
    rails = LLMRails(config)

    async def get_rails(config_ids, model_name=None):
        assert config_ids == ["recorded-tool-stream"]
        assert model_name == OPENAI_MODEL
        return rails

    monkeypatch.setattr(api, "_get_rails", get_rails)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://testserver") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": OPENAI_MODEL,
                "messages": [{"role": "user", "content": "Call get_weather for Paris. Do not answer in prose."}],
                "stream": True,
                "stop": ["Paris"],
                "guardrails": {"config_id": "recorded-tool-stream"},
            },
        )
    assert response.status_code == 200
    assert len(vcr.requests) == 1
    if record_mode == "none":
        assert vcr.play_count == 1
        assert vcr.all_played
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    events = [ChatCompletionChunk.model_validate_json(frame) for frame in frames[:-1]]
    assert len({event.id for event in events}) == 1
    assert all(event.model == OPENAI_MODEL for event in events)
    choices = [event.choices[0] for event in events]
    tool_choices = [choice for choice in choices if choice.delta.tool_calls]
    assert len(tool_choices) > 1
    assert all(choice.finish_reason is None for choice in tool_choices)
    assert all(choice.delta.content is None for choice in tool_choices)
    calls = [call for choice in tool_choices for call in choice.delta.tool_calls or []]
    assert all(call.index == 0 for call in calls)
    assert calls[0].id
    assert calls[0].type == "function"
    assert "".join(call.function.name or "" for call in calls if call.function) == "get_weather"
    assert json.loads("".join(call.function.arguments or "" for call in calls if call.function)) == {"city": "Paris"}
    terminal = [choice for choice in choices if choice.finish_reason]
    assert len(terminal) == 1
    assert terminal[0].finish_reason == "tool_calls"
    assert terminal[0].delta.tool_calls is None
    assert choices.index(terminal[0]) > choices.index(tool_choices[-1])
    normalized = []
    ids = {}
    for choice in choices:
        data = choice.model_dump(exclude_none=True)
        for call in data["delta"].get("tool_calls", []):
            if "id" in call:
                call["id"] = ids.setdefault(call["id"], f"[TOOL_CALL_{len(ids)}]")
        normalized.append(data)
    if record_mode == "none":
        assert_request_payload(
            recorded_cassette_path,
            model=OPENAI_MODEL,
            stream=True,
            expected_params={
                "tools": [WEATHER_TOOL],
                "tool_choice": WEATHER_TOOL_CHOICE,
            },
        )
    assert normalized == snapshot(
        [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "[TOOL_CALL_0]",
                            "function": {"arguments": "", "name": "get_weather"},
                            "type": "function",
                        }
                    ]
                },
                "index": 0,
            },
            {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"'}}]}, "index": 0},
            {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "city"}}]}, "index": 0},
            {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '":"'}}]}, "index": 0},
            {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "Paris"}}]}, "index": 0},
            {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"}'}}]}, "index": 0},
            {"delta": {}, "finish_reason": "tool_calls", "index": 0},
        ]
    )

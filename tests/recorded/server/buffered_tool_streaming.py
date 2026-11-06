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
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import httpx
from openai import AsyncOpenAI
from openai.lib.streaming.chat import ChatCompletionStreamState

from nemoguardrails import Guardrails, LLMRails
from nemoguardrails.server import api
from tests.recorded.cassette import decode_body_json
from tests.recorded.rails.public_api.configs import (
    NIM_TOOL_CALL_CONFIG,
    OPENAI_BASELINE_CONFIG,
    WEATHER_TOOL,
)
from tests.recorded.rails_config import load_config
from tests.recorded.server.configs import TOOL_CHAT_PROMPT


def _config(provider, credential):
    config = load_config(OPENAI_BASELINE_CONFIG if provider == "openai" else NIM_TOOL_CALL_CONFIG)
    config.models[0].parameters["timeout"] = 30
    if provider == "nim":
        config.models[0].api_key_env_var = "NVIDIA_INFERENCE_KEY"
        config.models[0].parameters.update(api_key=credential, chat_template_kwargs={"enable_thinking": False})
    return config


@asynccontextmanager
async def _backend(config, engine, monkeypatch):
    config.passthrough = engine != "llmrails-chat"
    if engine == "llmrails-chat" and not config.prompts:
        config.prompts = [TOOL_CHAT_PROMPT.model_copy(deep=True)]
    if engine == "iorails":
        for model in config.models:
            model.parameters.pop("max_retries", None)
        config.rails.tool_call.flows = ["tool call validation"]
        config.rails.tool_result.flows = ["tool result validation"]
        async with Guardrails(config, require_iorails=True) as rails:
            assert rails.use_iorails_engine
            monkeypatch.setattr(api, "_get_rails", AsyncMock(return_value=rails))
            yield rails
    else:
        rails = LLMRails(config)
        monkeypatch.setattr(api, "_get_rails", AsyncMock(return_value=rails))
        yield rails


def _request(config, messages, **params):
    return {
        "model": config.models[0].model,
        "messages": messages,
        "tools": [WEATHER_TOOL],
        "stream": True,
        "guardrails": {"config_id": "recorded-buffered-tools"},
        **params,
    }


class ToolStreamRecorder:
    """Capture downstream SSE separately from the upstream provider cassettes."""

    def __init__(self):
        self.events: list[str] = []

    async def __call__(self, request):
        return await _stream(request, self.events)


def _normalized_sse(body):
    tool_ids = {}
    normalized = []
    for line in body.splitlines(keepends=True):
        if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
            normalized.append(line)
            continue
        payload = json.loads(line[6:])
        if "id" in payload:
            payload["id"] = "[COMPLETION_ID]"
        if "created" in payload:
            payload["created"] = 0
        for choice in payload["choices"]:
            for call in choice["delta"].get("tool_calls", []):
                if "id" in call:
                    call["id"] = tool_ids.setdefault(call["id"], f"[TOOL_CALL_{len(tool_ids)}]")
        ending = line[len(line.rstrip("\r\n")) :]
        normalized.append(f"data: {json.dumps(payload)}{ending}")
    return "".join(normalized)


async def _stream(request, events):
    """Consume the real ASGI route through the SDK and independently check its wire framing."""
    responses = []

    async def capture(response):
        await response.aread()
        responses.append(response)

    async with AsyncOpenAI(
        api_key="local-asgi-test",
        base_url="http://testserver/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), event_hooks={"response": [capture]}),
    ) as client:
        params = dict(request)
        guardrails = params.pop("guardrails")
        stream = await client.chat.completions.create(**params, extra_body={"guardrails": guardrails})
        chunks = [chunk async for chunk in stream]
    assert len(responses) == 1
    response = responses[0]
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert frames[-1] == "[DONE]" and frames.count("[DONE]") == 1
    payloads = [json.loads(frame) for frame in frames[:-1]]
    assert payloads[0]["guardrails"]["config_id"] == request["guardrails"]["config_id"]
    assert all("guardrails" not in payload for payload in payloads[1:])
    assert len(chunks) == len(frames) - 1
    assert len({chunk.id for chunk in chunks}) == 1
    assert len({chunk.created for chunk in chunks}) == 1
    assert all(chunk.object == "chat.completion.chunk" and chunk.model == request["model"] for chunk in chunks)
    choices = [choice for chunk in chunks for choice in chunk.choices]
    assert len(choices) >= 3
    assert choices[0].delta.role == "assistant"
    assert all(choice.index == 0 and choice.finish_reason is None for choice in choices[:-1])
    terminal = choices[-1]
    assert terminal.delta.model_dump(exclude_none=True) == {}
    state = ChatCompletionStreamState()
    for chunk in chunks:
        state.handle_chunk(chunk)
    message = state.get_final_completion().choices[0].message.model_dump(exclude_none=True)
    assert message["role"] == "assistant"
    call_headers = [call for choice in choices for call in choice.delta.tool_calls or [] if call.id]
    assert [call.index for call in call_headers] == list(range(len(message.get("tool_calls", []))))
    for index, call in enumerate(message.get("tool_calls", [])):
        assert call.pop("index") == index
        assert call["id"] and call["type"] == "function"
        assert isinstance(json.loads(call["function"]["arguments"]), dict)
    assert terminal.finish_reason == ("tool_calls" if message.get("tool_calls") else "stop")
    events.append(_normalized_sse(response.text))
    return message


def _wire_requests(vcr, count):
    assert len(vcr.requests) == count
    if vcr.record_mode == "none":
        assert vcr.play_count == count and vcr.all_played
    requests = [decode_body_json(request.body) for request in vcr.requests]
    # ASGI must execute on replay: recording testserver could hide a regression in the route itself.
    assert all("testserver" not in request.uri for request in vcr.requests)
    assert all(not request.get("stream", False) for request in requests)
    return requests


def _assert_provider_calls(vcr, index, message):
    provider = decode_body_json(vcr.responses[index]["body"])["choices"][0]["message"]
    actual = message["tool_calls"]
    assert len(actual) == len(provider["tool_calls"])
    for public, wire in zip(actual, provider["tool_calls"]):
        assert public["id"] == wire["id"]
        assert public["type"] == wire["type"]
        assert public["function"]["name"] == wire["function"]["name"]
        assert json.loads(public["function"]["arguments"]) == json.loads(wire["function"]["arguments"])


def _calls(message):
    return [
        {"name": call["function"]["name"], "arguments": json.loads(call["function"]["arguments"])}
        for call in message["tool_calls"]
    ]

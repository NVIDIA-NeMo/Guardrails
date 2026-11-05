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

"""Refresh provider evidence independently of the reviewed field inventory.

Allowed replies preserve the complete sanitized recording while supported text
still reaches the checker. Synthetic block decisions reuse that same exchange.
Strict source/conformance audits and injected future fields have separate tests.
"""

import json

import httpx
import pytest

from nemoguardrails.server.experimental._content_checker import (
    ContentBlocked,
    ContentInspectionPolicy,
    StreamBufferingPolicy,
)
from nemoguardrails.server.experimental.provider.sse import iter_sse_events
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.chat_completions import (
    LITELLM_CHAT_COMPLETIONS_ENDPOINT,
)
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.responses import LITELLM_RESPONSES_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT
from tests.recorded.cassette import cassette_request_jsons
from tests.server.experimental.checkers import StaticChecker
from tests.server.experimental.responses_proxy import responses_proxy

pytestmark = [
    pytest.mark.recorded,
    pytest.mark.vcr(ignore_hosts=["recorded-responses.test"]),
    pytest.mark.asyncio,
]

OPENAI_BASE_URL = "https://api.openai.com/v1"
GATEWAY_BASE_URL = "https://inference-api.nvidia.com/v1"
NATIVE_MODEL = "gpt-5-nano-2025-08-07"
GATEWAY_MODEL = "openai/openai/gpt-5-mini"
INPUT = "Reply with exactly: Hello."


def provider_request(model, operation, stream):
    request = {"model": model, "stream": stream}
    native = model == NATIVE_MODEL
    if operation == "responses":
        request.update(
            input=INPUT,
            store=False,
            max_output_tokens=256 if native else 1024,
            reasoning={"effort": "minimal" if native else "low"},
        )
    else:
        request.update(
            messages=[{"role": "user", "content": INPUT}],
            max_completion_tokens=1024,
            reasoning_effort="low",
        )
        if stream:
            request["stream_options"] = {"include_usage": True}
    return request


async def stream_events(wire):
    async def chunks():
        yield wire

    return [event async for event in iter_sse_events(chunks(), max_event_bytes=1024 * 1024)]


def response_text(payload, operation):
    if operation == "chat/completions":
        return "".join(choice["message"]["content"] for choice in payload["choices"])
    return "".join(
        part["text"]
        for item in payload["output"]
        if item["type"] == "message" and item["role"] == "assistant"
        for part in item["content"]
        if part["type"] == "output_text"
    )


async def recorded_text(upstream, operation, stream):
    if not stream:
        return response_text(upstream.json(), operation), 1
    events = await stream_events(upstream.content)
    payloads = [json.loads(event.data) for event in events if event.data not in (None, b"[DONE]")]
    if operation == "chat/completions":
        deltas = [
            choice["delta"]["content"]
            for payload in payloads
            for choice in payload["choices"]
            if choice["delta"].get("content")
        ]
        assert events[-1].data == b"[DONE]"
    else:
        deltas = [payload["delta"] for payload in payloads if payload["type"] == "response.output_text.delta"]
        completed = next(
            payload["response"] for payload in reversed(payloads) if payload["type"] == "response.completed"
        )
        assert response_text(completed, operation) == "".join(deltas)
    return "".join(deltas), len(deltas) + 1


def assert_consumed_once(cassette):
    assert len(cassette.requests) == 1
    assert cassette.play_count == 1
    assert cassette.all_played


async def assert_forward_compatible_exchange(
    api_key, base_url, endpoint, request, record_mode, recorded_cassette_path, cassette, operation, stream
):
    async with httpx.AsyncClient(timeout=60) as client:
        upstream = await client.post(
            base_url + "/" + operation,
            json=request,
            headers={"authorization": f"Bearer {api_key}", "accept-encoding": "identity"},
        )
    assert upstream.status_code == 200
    assert len(cassette.requests) == 1
    if record_mode != "none":
        return

    assert_consumed_once(cassette)
    assert cassette_request_jsons(recorded_cassette_path) == [request]
    text, window = await recorded_text(upstream, operation, stream)
    assert text

    for unknown_fields in ("allow", "forbid"):
        for blocked in (False, True):
            cassette.rewind()
            checker = StaticChecker(
                ContentInspectionPolicy(True, True, StreamBufferingPolicy(window, 0) if stream else None),
                output_decisions=[ContentBlocked("recorded test block")] if blocked else [],
            )
            dispatches = []
            async with responses_proxy(
                api_key, checker, dispatches, endpoint=endpoint, base_url=base_url, unknown_fields=unknown_fields
            ) as client:
                response = await client.post("/v1/" + operation, json=request)
            assert_consumed_once(cassette)
            assert len(dispatches) == 1
            assert json.loads(dispatches[0].body) == request
            assert checker.text_calls == [("input", INPUT), ("output", text)]
            if not blocked:
                assert response.status_code == 200
                assert response.content == upstream.content
                assert response.headers["content-type"] == upstream.headers["content-type"]
            else:
                assert response.status_code == (200 if stream else 400)
                assert "content_blocked" in response.text
                assert "unsupported_" not in response.text
                if stream:
                    received = [event for event in await stream_events(response.content) if event.data is not None]
                    errors = received[:-1] if operation == "chat/completions" else received
                    assert len(errors) == 1
                    error = json.loads(errors[0].data)
                    if operation == "chat/completions":
                        assert received[-1].data == b"[DONE]"
                        assert error["error"]["code"] == "content_blocked"
                    else:
                        assert error["type"] == "error"
                        assert error["code"] == "content_blocked"


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
async def test_native_responses_forward_compatibility(openai_api_key, record_mode, recorded_cassette_path, vcr, stream):
    request = provider_request(NATIVE_MODEL, "responses", stream)
    await assert_forward_compatible_exchange(
        openai_api_key,
        OPENAI_BASE_URL,
        RESPONSES_ENDPOINT,
        request,
        record_mode,
        recorded_cassette_path,
        vcr,
        "responses",
        stream,
    )


@pytest.mark.parametrize(
    "operation,endpoint",
    [
        pytest.param("responses", LITELLM_RESPONSES_ENDPOINT, id="responses"),
        pytest.param("chat/completions", LITELLM_CHAT_COMPLETIONS_ENDPOINT, id="chat"),
    ],
)
@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
async def test_gateway_forward_compatibility(
    nvidia_inference_key, record_mode, recorded_cassette_path, vcr, stream, operation, endpoint
):
    request = provider_request(GATEWAY_MODEL, operation, stream)
    await assert_forward_compatible_exchange(
        nvidia_inference_key,
        GATEWAY_BASE_URL,
        endpoint,
        request,
        record_mode,
        recorded_cassette_path,
        vcr,
        operation,
        stream,
    )

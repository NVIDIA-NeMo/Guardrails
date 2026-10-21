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
from jsonschema import Draft202012Validator

from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentBlocked,
    ContentInspectionPolicy,
    StreamBufferingPolicy,
)
from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.provider.sse import iter_sse_events
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT
from tests.recorded.cassette import cassette_request_jsons
from tests.server.experimental.checkers import StaticChecker
from tests.server.experimental.responses_proxy import responses_proxy as proxy

pytestmark = [
    pytest.mark.recorded,
    pytest.mark.vcr(ignore_hosts=["recorded-responses.test"]),
    pytest.mark.asyncio,
]

BASE_URL = "https://api.openai.com/v1"
MODEL = "gpt-5-nano-2025-08-07"
INPUT = "Reply with exactly: Hello, recorded Responses!"
ENDPOINT = RESPONSES_ENDPOINT
BUFFERED_RECORDING = "test_native_responses_preserve_policy_and_wire[buffered].yaml"
STREAMING_RECORDING = "test_native_responses_preserve_policy_and_wire[streaming].yaml"


def responses_request(stream, *, model=MODEL):
    return {
        "model": model,
        "input": INPUT,
        "stream": stream,
        "store": False,
        "max_output_tokens": 256,
        "reasoning": {"effort": "minimal"},
        "metadata": {"test": "native-responses"},
    }


async def capture(api_key, request):
    async with httpx.AsyncClient(timeout=60) as client:
        return await client.post(
            BASE_URL + "/responses",
            json=request,
            headers={"authorization": f"Bearer {api_key}", "accept-encoding": "identity"},
        )


async def stream_events(wire):
    async def chunks():
        yield wire

    return [event async for event in iter_sse_events(chunks(), max_event_bytes=1024 * 1024)]


def assistant_text(payload):
    return "".join(
        part["text"]
        for item in payload["output"]
        if item["type"] == "message" and item["role"] == "assistant"
        for part in item["content"]
        if part["type"] == "output_text"
    )


def assert_consumed_once(cassette):
    assert len(cassette.requests) == 1
    assert cassette.play_count == 1
    assert cassette.all_played


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
async def test_native_responses_preserve_policy_and_wire(
    openai_api_key, record_mode, recorded_cassette_path, vcr, stream
):
    request = responses_request(stream)
    upstream = await capture(openai_api_key, request)
    assert upstream.status_code == 200
    if record_mode != "none":
        assert len(vcr.requests) == 1
        return

    assert_consumed_once(vcr)
    assert cassette_request_jsons(recorded_cassette_path) == [request]
    document = export_guard_contract(ENDPOINT)
    ENDPOINT.guarded_request_model.validate_payload(request)
    Draft202012Validator(document["request"]).validate(request)
    if stream:
        events = await stream_events(upstream.content)
        payloads = [json.loads(event.data) for event in events]
        assert payloads[0]["type"] == "response.created"
        assert payloads[-1]["type"] == "response.completed"
        deltas = [payload["delta"] for payload in payloads if payload["type"] == "response.output_text.delta"]
        text = "".join(deltas)
        assert assistant_text(payloads[-1]["response"]) == text
        assert ENDPOINT.stream is not None
        adapter = ENDPOINT.stream.create_adapter()
        for event, payload in zip(events, payloads):
            Draft202012Validator(document["stream"]).validate(payload)
            adapter.classify_event(event)
        adapter.validate_end_of_stream()
    else:
        payload = upstream.json()
        text = assistant_text(payload)
        parsed = ENDPOINT.guarded_response_model.validate_payload(payload)
        assert parsed.locate_guarded_message(payload).message.content == text
        Draft202012Validator(document["response"]).validate(payload)
    assert text

    for inspected in (True, False):
        vcr.rewind()
        checker = StaticChecker(
            ContentInspectionPolicy(inspected, inspected, StreamBufferingPolicy(1, 0) if inspected else None)
        )
        dispatches = []
        async with proxy(openai_api_key, checker, dispatches) as client:
            response = await client.post("/v1/responses", json=request)
        assert_consumed_once(vcr)
        assert len(dispatches) == 1
        assert json.loads(dispatches[0].body) == request
        assert response.status_code == 200
        assert response.content == upstream.content
        assert response.headers["content-type"] == upstream.headers["content-type"]
        assert checker.text_calls == (
            [("input", INPUT)] + [("output", delta) for delta in (deltas if stream else [text])] if inspected else []
        )


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
async def test_native_provider_error_is_preserved(openai_api_key, record_mode, recorded_cassette_path, vcr, stream):
    request = responses_request(stream, model="recorded-invalid-responses-model")
    upstream = await capture(openai_api_key, request)
    assert upstream.status_code == 404
    assert upstream.json()["error"]["code"] == "model_not_found"
    if record_mode != "none":
        assert len(vcr.requests) == 1
        return

    assert_consumed_once(vcr)
    assert cassette_request_jsons(recorded_cassette_path) == [request]
    vcr.rewind()
    checker = StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)))
    dispatches = []
    async with proxy(openai_api_key, checker, dispatches) as client:
        response = await client.post("/v1/responses", json=request)
    assert_consumed_once(vcr)
    assert len(dispatches) == 1
    assert response.status_code == upstream.status_code
    assert response.content == upstream.content
    assert checker.text_calls == [("input", INPUT)]


@pytest.mark.vcr(BUFFERED_RECORDING, record_mode="none")
async def test_recorded_buffered_output_is_withheld_on_block(openai_api_key, vcr):
    upstream = await capture(openai_api_key, responses_request(False))
    assert_consumed_once(vcr)
    text = assistant_text(upstream.json())
    vcr.rewind()
    checker = StaticChecker(ContentInspectionPolicy(True, True), output_decisions=[ContentBlocked("blocked by test")])
    dispatches = []
    async with proxy(openai_api_key, checker, dispatches) as client:
        response = await client.post("/v1/responses", json=responses_request(False))
    assert_consumed_once(vcr)
    assert len(dispatches) == 1
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "content_blocked"
    assert text not in response.text
    assert checker.text_calls == [("input", INPUT), ("output", text)]


@pytest.mark.vcr(STREAMING_RECORDING, record_mode="none")
@pytest.mark.parametrize("allowed_windows", [0, 1], ids=["first-window", "later-window"])
@pytest.mark.parametrize("chunk_size,context_size", [(1, 0), (2, 1)], ids=["single-delta", "overlapping-context"])
async def test_recorded_stream_releases_only_approved_prefix(
    openai_api_key, vcr, allowed_windows, chunk_size, context_size
):
    upstream = await capture(openai_api_key, responses_request(True))
    assert_consumed_once(vcr)
    events = await stream_events(upstream.content)
    text_events = [
        (index, payload["delta"])
        for index, event in enumerate(events)
        if (payload := json.loads(event.data))["type"] == "response.output_text.delta" and payload["delta"]
    ]
    assert len(text_events) > (allowed_windows + 1) * chunk_size
    approved_deltas = allowed_windows * chunk_size
    released_end = text_events[approved_deltas - 1][0] + 1 if approved_deltas else 0
    checker = StaticChecker(
        ContentInspectionPolicy(True, True, StreamBufferingPolicy(chunk_size, context_size)),
        output_decisions=[ContentAllowed()] * allowed_windows + [ContentBlocked("blocked by test")],
    )
    vcr.rewind()
    dispatches = []
    async with proxy(openai_api_key, checker, dispatches) as client:
        response = await client.post("/v1/responses", json=responses_request(True))
    assert_consumed_once(vcr)
    assert len(dispatches) == 1
    assert response.status_code == 200
    received = await stream_events(response.content)
    assert [event.raw for event in received[:-1]] == [event.raw for event in events[:released_end]]
    assert received[-1].event_type == b"error"
    assert json.loads(received[-1].data)["code"] == "content_blocked"
    assert b"[DONE]" not in response.content
    assert checker.text_calls == [("input", INPUT)] + [
        (
            "output",
            "".join(
                text for _, text in text_events[max(0, window * chunk_size - context_size) : (window + 1) * chunk_size]
            ),
        )
        for window in range(allowed_windows + 1)
    ]

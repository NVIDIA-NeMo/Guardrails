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

"""Replay gateway Responses evidence through the actual bound policy and HTTP path.

Capture completes before policy validation, so a new unsupported shape is not
silently truncated or changed to fit the contract. Recording is followed by the
mandatory offline replay, where bound-policy and passthrough assertions run.
"""

import json
from functools import partial

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
from nemoguardrails.server.experimental.provider.stream import StreamEventRole
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

BASE_URL = "https://inference-api.nvidia.com/v1"
MODEL = "openai/openai/gpt-5-nano"
INPUT = "Reply with exactly: Hello."
ENDPOINT = LITELLM_RESPONSES_ENDPOINT
proxy = partial(responses_proxy, endpoint=ENDPOINT, base_url=BASE_URL)


def responses_request(stream, *, model=MODEL):
    return {
        "model": model,
        "input": INPUT,
        "stream": stream,
        "store": False,
        "max_output_tokens": 256,
        "reasoning": {"effort": "minimal"},
    }


async def capture(api_key, request):
    async with httpx.AsyncClient(timeout=60) as client:
        return await client.post(
            BASE_URL + "/responses",
            json=request,
            headers={"authorization": f"Bearer {api_key}", "accept-encoding": "identity"},
        )


async def events(wire):
    async def chunks():
        yield wire

    return [event async for event in iter_sse_events(chunks(), max_event_bytes=1024 * 1024)]


async def validate_inventory(upstream, request):
    document = export_guard_contract(ENDPOINT)
    base_document = export_guard_contract(RESPONSES_ENDPOINT)
    ENDPOINT.guarded_request_model.validate_payload(request)
    Draft202012Validator(document["request"]).validate(request)
    if not request["stream"]:
        payload = upstream.json()
        assert payload.keys() <= base_document["response"]["properties"].keys()
        assert payload["access_programs"] is None
        assert payload["billing"] == {"payer": "developer"}
        assert payload["store"] is False
        assert payload["frequency_penalty"] == payload["presence_penalty"] == 0
        assert set(payload["tool_usage"]) == {"image_gen", "web_search"}
        parsed = ENDPOINT.guarded_response_model.validate_payload(payload)
        Draft202012Validator(document["response"]).validate(payload)
        return parsed.locate_guarded_message(payload).message.content

    assert ENDPOINT.stream is not None
    adapter = ENDPOINT.stream.create_adapter()
    validator = Draft202012Validator(document["stream"])
    text = []
    kinds = []
    for event in await events(upstream.content):
        if event.data not in (None, b"[DONE]"):
            payload = json.loads(event.data)
            validator.validate(payload)
            assert payload["model"] == MODEL
            assert "provider_specific_fields" not in payload
            if "response" in payload:
                assert payload["response"].keys() <= base_document["response"]["properties"].keys()
            kinds.append(payload["type"])
        classified = adapter.classify_event(event)
        if classified.role is StreamEventRole.GUARDED_TEXT:
            assert classified.text is not None
            text.append(classified.text)
    adapter.validate_end_of_stream()
    assert set(kinds) == {
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.completed",
    }
    assert kinds[-1] == "response.completed"
    return "".join(text)


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
async def test_gateway_responses_preserve_policy_and_wire(
    nvidia_inference_key, record_mode, recorded_cassette_path, vcr, stream
):
    request = responses_request(stream)
    upstream = await capture(nvidia_inference_key, request)
    assert upstream.status_code == 200
    if record_mode != "none":
        assert len(vcr.requests) == 1
        return

    assert vcr.play_count == 1
    assert vcr.all_played

    assert cassette_request_jsons(recorded_cassette_path) == [request]
    text = await validate_inventory(upstream, request)
    assert text
    for inspected in (True, False):
        vcr.rewind()
        checker = StaticChecker(
            ContentInspectionPolicy(inspected, inspected, StreamBufferingPolicy(1, 0) if inspected else None)
        )
        dispatches = []
        async with proxy(nvidia_inference_key, checker, dispatches) as client:
            response = await client.post("/v1/responses", json=request)
        assert response.status_code == upstream.status_code
        assert response.content == upstream.content
        assert response.headers["content-type"] == upstream.headers["content-type"]
        assert len(dispatches) == len(vcr.requests) == vcr.play_count == 1
        assert vcr.all_played
        assert json.loads(dispatches[0].body) == request
        if inspected:
            assert checker.text_calls[0] == ("input", INPUT)
            assert "".join(value for kind, value in checker.text_calls if kind == "output") == text
        else:
            assert checker.calls == []


@pytest.mark.vcr("test_gateway_responses_preserve_policy_and_wire[buffered].yaml", record_mode="none")
async def test_gateway_recorded_buffered_block_withholds_output(nvidia_inference_key, vcr):
    checker = StaticChecker(ContentInspectionPolicy(True, True), output_decisions=[ContentBlocked("blocked by test")])
    dispatches = []
    async with proxy(nvidia_inference_key, checker, dispatches) as client:
        response = await client.post("/v1/responses", json=responses_request(False))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "content_blocked"
    assert len(dispatches) == len(vcr.requests) == vcr.play_count == 1
    assert vcr.all_played
    assert checker.text_calls[0] == ("input", INPUT)
    assert checker.text_calls[1][0] == "output"
    assert checker.text_calls[1][1] not in response.text


@pytest.mark.vcr("test_gateway_responses_preserve_policy_and_wire[streaming].yaml", record_mode="none")
@pytest.mark.parametrize("allowed_windows", [0, 1], ids=["first-window", "later-window"])
async def test_gateway_recorded_stream_releases_only_approved_prefix(nvidia_inference_key, vcr, allowed_windows):
    upstream = await capture(nvidia_inference_key, responses_request(True))
    recorded_events = await events(upstream.content)
    deltas = [
        (index, payload["delta"])
        for index, event in enumerate(recorded_events)
        if event.data not in (None, b"[DONE]")
        and (payload := json.loads(event.data))["type"] == "response.output_text.delta"
        and payload["delta"]
    ]
    assert len(deltas) > allowed_windows
    released_end = deltas[allowed_windows - 1][0] + 1 if allowed_windows else 0
    assert vcr.play_count == 1
    vcr.rewind()
    checker = StaticChecker(
        ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)),
        output_decisions=[ContentAllowed()] * allowed_windows + [ContentBlocked("blocked by test")],
    )
    dispatches = []
    async with proxy(nvidia_inference_key, checker, dispatches) as client:
        response = await client.post("/v1/responses", json=responses_request(True))
    assert response.status_code == 200
    received = await events(response.content)
    assert [event.raw for event in received[:-1]] == [event.raw for event in recorded_events[:released_end]]
    assert received[-1].event_type == b"error"
    assert json.loads(received[-1].data)["code"] == "content_blocked"
    assert b"[DONE]" not in response.content
    assert checker.text_calls == [("input", INPUT)] + [("output", text) for _, text in deltas[: allowed_windows + 1]]
    assert len(dispatches) == len(vcr.requests) == vcr.play_count == 1
    assert vcr.all_played

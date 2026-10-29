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

"""Refreshable route evidence, independent of the dialect's declared inventory.

Capture completes before policy validation. Offline replay is the acceptance
gate: new gateway fields must remain in the cassette and receive a policy review.
Checker verdicts are synthetic; provider response bodies are actual recordings.
"""

import json
from itertools import product

import httpx
import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentBlocked,
    ContentInspectionPolicy,
    StreamBufferingPolicy,
)
from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.provider.sse import iter_sse_events
from nemoguardrails.server.experimental.provider.stream import UnsupportedProviderStream
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.chat_completions import (
    LITELLM_CHAT_COMPLETIONS_ENDPOINT,
)
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.responses import LITELLM_RESPONSES_ENDPOINT
from tests.recorded.cassette import cassette_request_jsons
from tests.server.experimental.checkers import StaticChecker
from tests.server.experimental.conformance.openai.litellm.support import response_validator
from tests.server.experimental.responses_proxy import responses_proxy

pytestmark = [pytest.mark.recorded, pytest.mark.asyncio, pytest.mark.vcr(ignore_hosts=["recorded-responses.test"])]

BASE_URL = "https://inference-api.nvidia.com/v1"
PROMPT = "Reply with exactly: Hello."
ROUTES = [
    pytest.param("openai/openai/gpt-4o-mini", id="openai-4o-mini"),
    pytest.param("openai/openai/gpt-5-nano", id="openai-5-nano"),
    pytest.param("openai/openai/gpt-6.1-sol", id="openai-6.1-sol"),
    pytest.param("openai/openai/gpt-6-sol", id="openai-6-sol"),
    pytest.param("openai/openai/gpt-5.4", id="openai-5.4"),
    pytest.param("openai/openai/gpt-5.1", id="openai-5.1"),
    pytest.param("openai/openai/gpt-5-mini", id="openai-5-mini"),
]
OPERATIONS = [pytest.param("chat/completions", id="chat"), pytest.param("responses", id="responses")]


def request_body(model, operation, stream):
    request = {"model": model, "stream": stream}
    if operation == "chat/completions":
        request.update(messages=[{"role": "user", "content": PROMPT}], max_completion_tokens=256)
        if stream:
            request["stream_options"] = {"include_usage": True}
        if model.endswith("gpt-5-nano"):
            request["reasoning_effort"] = "minimal"
    else:
        request.update(input=PROMPT, store=False, max_output_tokens=256)
        if model.endswith("gpt-5-nano"):
            request["reasoning"] = {"effort": "minimal"}
    if model not in {"openai/openai/gpt-4o-mini", "openai/openai/gpt-5-nano"}:
        if operation == "chat/completions":
            request.update(max_completion_tokens=1024, reasoning_effort="low")
        else:
            request.update(max_output_tokens=1024, reasoning={"effort": "low"})
    return request


async def events(wire):
    async def chunks():
        yield wire

    return [event async for event in iter_sse_events(chunks(), max_event_bytes=1024 * 1024)]


def buffered_text(payload, operation):
    if operation == "chat/completions":
        return payload["choices"][0]["message"]["content"]
    message = next(item for item in payload["output"] if item["type"] == "message")
    return message["content"][0]["text"]


def delta_text(payload, operation):
    if operation == "chat/completions":
        return "".join(choice["delta"].get("content") or "" for choice in payload["choices"])
    return payload["delta"] if payload["type"] == "response.output_text.delta" else ""


def assert_route_inventory(payloads, model, operation, stream):
    """Pin independently observed locations; do not derive them from guard models."""
    if operation == "responses":
        snapshots = [p["response"] for p in payloads if "response" in p] if stream else payloads
        assert any("content_filters" in p for p in snapshots) is False
        return
    bridge = model in {
        "openai/openai/gpt-4o-mini",
        "openai/openai/gpt-5-mini",
        "openai/openai/gpt-6-sol",
        "openai/openai/gpt-6.1-sol",
    }
    assert any("frequency_penalty" in p for p in payloads) == bridge
    assert any("content_filters" in p for p in payloads) is False
    if stream and bridge:
        metadata_chunks = [p for p in payloads if p.get("provider_specific_fields")]
        assert len(metadata_chunks) == 1
        terminal = metadata_chunks[0]
        if model != "openai/openai/gpt-5-mini":
            assert any(c.get("finish_reason") == "stop" for c in terminal["choices"])
        expected = {"frequency_penalty", "presence_penalty", "tool_usage", "service_tier"}
        if model == "openai/openai/gpt-6-sol":
            expected.add("access_programs")
        assert set(terminal["provider_specific_fields"]) == expected
    if not stream and bridge:
        assert payloads[0]["billing"] == {"payer": "developer"}


async def assert_unsupported_reasoning(request, upstream, payloads, stream, endpoint, vcr, key):
    """Retain real nonempty reasoning items and pin the current rejection boundary."""
    location = "delta" if stream else "message"
    items = [c[location].get("reasoning_items") for p in payloads for c in p["choices"]]
    assert any(items)
    contract = export_guard_contract(endpoint)
    validator = Draft202012Validator(contract["stream" if stream else "response"])
    violations = [error for payload in payloads for error in validator.iter_errors(payload)]
    assert violations
    assert any("reasoning_items" in str(error) for error in violations)
    if stream:
        adapter = endpoint.stream.create_adapter()
        recorded_events = await events(upstream.content)
        with pytest.raises(UnsupportedProviderStream, match="reasoning_items"):
            for failed_index, event in enumerate(recorded_events):
                adapter.classify_event(event)
    else:
        with pytest.raises(ValidationError, match="reasoning_items"):
            endpoint.guarded_response_model.validate_payload(payloads[0])

    for inspect_input, inspect_output, window in product([False, True], [False, True], [1, 100] if stream else [100]):
        vcr.rewind()
        dispatches = []
        checker = StaticChecker(
            ContentInspectionPolicy(
                inspect_input, inspect_output, StreamBufferingPolicy(window, 0) if inspect_output else None
            )
        )
        async with responses_proxy(key, checker, dispatches, endpoint=endpoint, base_url=BASE_URL) as client:
            response = await client.post("/v1/chat/completions", json=request)
        assert len(dispatches) == len(vcr.requests) == vcr.play_count == 1
        assert vcr.all_played
        assert json.loads(dispatches[0].body) == request
        assert [value for kind, value in checker.text_calls if kind == "input"] == ([PROMPT] if inspect_input else [])
        if inspect_output:
            assert response.status_code == (200 if stream else 502)
            assert "unsupported_" in response.text
            assert response.content != upstream.content
            if stream:
                received = await events(response.content)
                assert received[-1].data == b"[DONE]"
                expected_prefix = recorded_events[:failed_index] if window == 1 else []
                assert [event.raw for event in received[:-2]] == [event.raw for event in expected_prefix]
            if window == 100:
                assert not any(kind == "output" for kind, _ in checker.text_calls)
        else:
            assert response.status_code == 200 and response.content == upstream.content
            assert not any(kind == "output" for kind, _ in checker.text_calls)


@pytest.mark.parametrize("model", ROUTES)
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
async def test_gateway_route_preserves_policy_and_wire(
    model, operation, stream, nvidia_inference_key, record_mode, recorded_cassette_path, vcr
):
    request = request_body(model, operation, stream)
    async with httpx.AsyncClient(timeout=60) as client:
        upstream = await client.post(
            BASE_URL + "/" + operation,
            json=request,
            headers={"authorization": f"Bearer {nvidia_inference_key}", "accept-encoding": "identity"},
        )
    assert upstream.status_code == 200
    assert len(vcr.requests) == 1
    if record_mode != "none":
        return
    assert vcr.play_count == 1 and vcr.all_played
    assert cassette_request_jsons(recorded_cassette_path) == [request]
    endpoint = LITELLM_CHAT_COMPLETIONS_ENDPOINT if operation == "chat/completions" else LITELLM_RESPONSES_ENDPOINT
    contract = export_guard_contract(endpoint)
    recorded_events = await events(upstream.content) if stream else []
    payloads = (
        [json.loads(e.data) for e in recorded_events if e.data not in (None, b"[DONE]")]
        if stream
        else [upstream.json()]
    )
    assert_route_inventory(payloads, model, operation, stream)
    upstream_validator = response_validator(operation)
    if not stream:
        upstream_validator.validate(payloads[0])
    elif operation == "responses":
        for payload in payloads:
            if "response" in payload:
                upstream_validator.validate(payload["response"])
    if model == "openai/openai/gpt-5-mini" and operation == "chat/completions":
        await assert_unsupported_reasoning(request, upstream, payloads, stream, endpoint, vcr, nvidia_inference_key)
        return
    validator = Draft202012Validator(contract["stream" if stream else "response"])
    for payload in payloads:
        validator.validate(payload)
    if stream:
        assert endpoint.stream is not None
        adapter = endpoint.stream.create_adapter()
        for event in recorded_events:
            adapter.classify_event(event)
        adapter.validate_end_of_stream()
        deltas = [
            (index, delta_text(json.loads(e.data), operation))
            for index, e in enumerate(recorded_events)
            if e.data not in (None, b"[DONE]") and delta_text(json.loads(e.data), operation)
        ]
        text = "".join(text for _, text in deltas)
        if operation == "chat/completions":
            assert recorded_events[-1].data == b"[DONE]"
            assert any(p.get("usage") for p in payloads)
            assert any(c.get("finish_reason") == "stop" for p in payloads for c in p["choices"])
        else:
            assert any(p["type"] == "response.completed" and p["response"]["usage"] for p in payloads)
    else:
        endpoint.guarded_response_model.validate_payload(payloads[0])
        text = buffered_text(payloads[0], operation)
    assert text

    async def exchange(checker):
        vcr.rewind()
        dispatches = []
        async with responses_proxy(
            nvidia_inference_key, checker, dispatches, endpoint=endpoint, base_url=BASE_URL
        ) as client:
            response = await client.post("/v1/" + operation, json=request)
        assert len(dispatches) == len(vcr.requests) == vcr.play_count == 1
        assert vcr.all_played
        assert json.loads(dispatches[0].body) == request
        return response

    for inspect_input, inspect_output, window in product([False, True], [False, True], [1, 100] if stream else [100]):
        checker = StaticChecker(
            ContentInspectionPolicy(
                inspect_input, inspect_output, StreamBufferingPolicy(window, 0) if inspect_output else None
            )
        )
        response = await exchange(checker)
        assert response.status_code == 200
        assert response.content == upstream.content
        assert response.headers["content-type"] == upstream.headers["content-type"]
        assert [value for kind, value in checker.text_calls if kind == "input"] == ([PROMPT] if inspect_input else [])
        assert "".join(value for kind, value in checker.text_calls if kind == "output") == (
            text if inspect_output else ""
        )

    # A large window reaches terminal metadata before checking this short answer.
    for window in [1, 100] if stream else [100]:
        checker = StaticChecker(
            ContentInspectionPolicy(True, True, StreamBufferingPolicy(window, 0)),
            output_decisions=[ContentBlocked("recorded test block")],
        )
        response = await exchange(checker)
        assert response.status_code == (200 if stream else 400)
        assert "content_blocked" in response.text
        assert "unsupported_" not in response.text
        assert checker.text_calls[0] == ("input", PROMPT)
        assert len(checker.text_calls) == 2
        if stream:
            received = await events(response.content)
            error_index = -2 if operation == "chat/completions" else -1
            assert received[:error_index] == []
            assert (received[-1].data == b"[DONE]") == (operation == "chat/completions")

    if stream:
        assert len(deltas) > 1, "Refresh must retain a multi-delta sample for later-window blocking"
        checker = StaticChecker(
            ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)),
            output_decisions=[ContentAllowed(), ContentBlocked("later block")],
        )
        response = await exchange(checker)
        received = await events(response.content)
        error_index = -2 if operation == "chat/completions" else -1
        assert [e.raw for e in received[:error_index]] == [e.raw for e in recorded_events[: deltas[0][0] + 1]]
        assert "content_blocked" in received[error_index].data.decode()
        assert checker.text_calls == [("input", PROMPT), ("output", deltas[0][1]), ("output", deltas[1][1])]

    # Input blocking must not consume the stored upstream interaction.
    vcr.rewind()
    checker = StaticChecker(ContentInspectionPolicy(True, True), input_decision=ContentBlocked("input block"))
    dispatches = []
    async with responses_proxy(
        nvidia_inference_key, checker, dispatches, endpoint=endpoint, base_url=BASE_URL
    ) as client:
        response = await client.post("/v1/" + operation, json=request)
    assert response.status_code == 400 and "content_blocked" in response.text
    assert dispatches == [] and vcr.play_count == 0
    assert checker.text_calls == [("input", PROMPT)]

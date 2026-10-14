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

"""Replay observed Chat exchanges with synthetic, provider-neutral verdicts.

This suite exercises protocol integration, not real Rails configuration or
socket delivery. Blocking cases reuse successful recordings without rewriting
them or recording safety-service decisions.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi import FastAPI

from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentBlocked,
    ContentInspectionPolicy,
    StreamBufferingPolicy,
)
from nemoguardrails.server.experimental._provider_transport import ProviderTransport
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.chat_completions import (
    LITELLM_CHAT_COMPLETIONS_ENDPOINT,
)
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_chat_router
from tests.recorded.assertions import assert_request_payload
from tests.recorded.cassette import cassette_request_jsons
from tests.recorded.snapshots import snapshot
from tests.server.experimental.checkers import StaticChecker

pytestmark = [
    pytest.mark.recorded,
    pytest.mark.vcr(ignore_hosts=["recorded-proxy.test"]),
    pytest.mark.asyncio,
]

NVIDIA_INFERENCE_BASE_URL = "https://inference-api.nvidia.com/v1"
OPENAI_MODEL = "openai/openai/gpt-5-nano"
INVALID_OPENAI_MODEL = "openai/openai/recorded-invalid-model"
USER_MESSAGE = "Hello! Can you help me with a short greeting?"
BUFFERED_RECORDING = "test_buffered_chat_preserves_guarded_constrained_and_opaque_contract_fields.yaml"
STREAMING_RECORDING = "test_streaming_chat_preserves_declared_events_and_opaque_metadata.yaml"
BLOCKED_ERROR = {
    "error": {"message": "blocked by test", "type": "guardrails_violation", "param": None, "code": "content_blocked"}
}


@pytest.fixture
def checker():
    return StaticChecker(ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0)))


@pytest.fixture(autouse=True)
def single_upstream_exchange(vcr):
    """Require one provider interaction per pass, including beneath dispatch."""
    yield
    assert len(vcr.requests) == 1
    if vcr.record_mode == "none":
        assert vcr.play_count == 1
        assert vcr.all_played


def chat_request(*, model=OPENAI_MODEL, stream=False):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": USER_MESSAGE}],
        "temperature": 0.7,
        "max_tokens": 1024,
        "n": 1,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


@asynccontextmanager
async def recorded_proxy(api_key, checker, dispatched=None) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as upstream_client:
        dispatcher = ProviderTransport(
            upstream_client,
            base_url=NVIDIA_INFERENCE_BASE_URL,
            mount_path="/v1",
        )

        async def buffered_dispatch(request):
            if dispatched is not None:
                dispatched.append(request)
            return await dispatcher.dispatch(request)

        async def streaming_dispatch(request):
            if dispatched is not None:
                dispatched.append(request)
            return await dispatcher.stream_dispatch(request)

        app = FastAPI()
        app.include_router(
            create_openai_chat_router(
                endpoint=LITELLM_CHAT_COMPLETIONS_ENDPOINT,
                checker=checker,
                dispatch=buffered_dispatch,
                stream_dispatch=streaming_dispatch,
            )
        )
        headers = {"authorization": f"Bearer {api_key}", "content-type": "application/json"}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://recorded-proxy.test",
            headers=headers,
        ) as client:
            yield client


def normalize_provider_metadata(value):
    value = deepcopy(value)
    if isinstance(value, dict):
        for key, nested in list(value.items()):
            if key == "id":
                value[key] = "[RECORDED_RESPONSE_ID]"
            elif key == "created":
                value[key] = 0
            elif key == "system_fingerprint" and nested is not None:
                value[key] = "[RECORDED_SYSTEM_FINGERPRINT]"
            else:
                value[key] = normalize_provider_metadata(nested)
    elif isinstance(value, list):
        value = [normalize_provider_metadata(item) for item in value]
    return value


def stream_events(body):
    events = []
    for event in body.split(b"\n\n"):
        data = [line.removeprefix(b"data: ") for line in event.splitlines() if line.startswith(b"data: ")]
        if not data:
            continue
        payload = b"\n".join(data)
        events.append("[DONE]" if payload == b"[DONE]" else json.loads(payload))
    return events


def recorded_provider_response(cassette_path):
    interactions = yaml.safe_load(cassette_path.read_text(encoding="utf-8"))["interactions"]
    assert len(interactions) == 1
    return interactions[0]["response"]


async def assert_allowed_relay_is_transparent(
    api_key, request, guarded_response, guarded_dispatches, cassette, *, stream
):
    assert cassette.play_count == 1
    assert cassette.all_played
    cassette.rewind()
    uninspected_checker = StaticChecker(ContentInspectionPolicy(False, False))
    uninspected_dispatches = []
    async with recorded_proxy(api_key, uninspected_checker, uninspected_dispatches) as client:
        uninspected_response = await client.post("/v1/chat/completions", json=request)

    assert uninspected_checker.calls == []
    assert len(guarded_dispatches) == len(uninspected_dispatches) == 1
    guarded_request, uninspected_request = guarded_dispatches[0], uninspected_dispatches[0]
    assert replace(guarded_request, headers=()) == replace(uninspected_request, headers=())
    assert [(key, value) for key, value in guarded_request.headers if key != b"accept-encoding"] == [
        (key, value) for key, value in uninspected_request.headers if key != b"accept-encoding"
    ]
    assert dict(guarded_request.headers)[b"accept-encoding"] == b"identity"
    assert dict(uninspected_request.headers)[b"accept-encoding"] != b"identity"
    assert guarded_response.status_code == uninspected_response.status_code
    assert guarded_response.content == uninspected_response.content
    guarded_headers = dict(guarded_response.headers)
    uninspected_headers = dict(uninspected_response.headers)
    if stream:
        guarded_headers.pop("content-length", None)
        uninspected_headers.pop("content-length", None)
    assert guarded_headers == uninspected_headers


def summarize_stream(events):
    payloads = [event for event in events if isinstance(event, dict)]
    choices = [choice for payload in payloads for choice in payload.get("choices") or []]
    deltas = [choice.get("delta") or {} for choice in choices]
    usage = next((payload["usage"] for payload in payloads if payload.get("usage")), None)
    return {
        "content": "".join(delta.get("content") or "" for delta in deltas),
        "model": next((payload.get("model") for payload in payloads if payload.get("model")), None),
        "finish_reason": next((choice.get("finish_reason") for choice in choices if choice.get("finish_reason")), None),
        "usage": usage,
        "event_kinds": sorted(
            {
                kind
                for kind, present in (
                    ("content", any("content" in delta for delta in deltas)),
                    ("role", any("role" in delta for delta in deltas)),
                    ("finish", any(choice.get("finish_reason") for choice in choices)),
                    ("usage", usage is not None),
                    ("done", events[-1] == "[DONE]"),
                )
                if present
            }
        ),
    }


def assert_recorded_request(record_mode, cassette_path, expected):
    if record_mode != "none":
        return
    assert_request_payload(
        cassette_path,
        model=expected["model"],
        stream=expected["stream"],
        expected_params={
            "messages": expected["messages"],
            "temperature": expected["temperature"],
            "max_tokens": expected["max_tokens"],
            "n": expected["n"],
        },
    )
    recorded = cassette_request_jsons(cassette_path)[-1]
    assert recorded == expected


async def test_buffered_chat_preserves_guarded_constrained_and_opaque_contract_fields(
    nvidia_inference_key, record_mode, recorded_cassette_path, checker, vcr
):
    request = chat_request()
    dispatched = []
    async with recorded_proxy(nvidia_inference_key, checker, dispatched) as client:
        response = await client.post("/v1/chat/completions", json=request)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["choices"]) == 1
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["message"]["content"]
    assert checker.text_calls == [
        ("input", USER_MESSAGE),
        ("output", payload["choices"][0]["message"]["content"]),
    ]
    assert_recorded_request(record_mode, recorded_cassette_path, request)
    if record_mode == "none":
        await assert_allowed_relay_is_transparent(
            nvidia_inference_key, request, response, dispatched, vcr, stream=False
        )
        provider_response = recorded_provider_response(recorded_cassette_path)
        assert response.status_code == provider_response["status"]["code"]
        assert normalize_provider_metadata(payload) == provider_response["body"]["parsed_body"]
    assert normalize_provider_metadata(payload) == snapshot(
        {
            "id": "[RECORDED_RESPONSE_ID]",
            "created": 0,
            "model": "openai/openai/gpt-5-nano",
            "object": "chat.completion",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": """\
Sure! Here are some short greeting options you can use, depending on the vibe:

Casual
- Hey there!
- Hi!
- Hi there!

Friendly/cheerful
- Hello there!
- Hi, nice to meet you!

Neutral/professional (email or meeting)
- Hello,
- Hi [Name],
- Good morning/afternoon,

General short
- Hello!
- Hi there!
- Hey!

If you share who it’s for and the tone you want (casual, professional, etc.), I can tailor a version for you. Want it in another language too?\
""",
                        "role": "assistant",
                        "provider_specific_fields": {"refusal": None},
                        "annotations": [],
                    },
                    "provider_specific_fields": {},
                }
            ],
            "usage": {
                "completion_tokens": 898,
                "prompt_tokens": 17,
                "total_tokens": 915,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 768,
                    "rejected_prediction_tokens": 0,
                },
                "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
            },
            "service_tier": "default",
        }
    )


async def test_streaming_chat_preserves_declared_events_and_opaque_metadata(
    nvidia_inference_key, record_mode, recorded_cassette_path, checker, vcr
):
    request = chat_request(stream=True)
    dispatched = []
    async with recorded_proxy(nvidia_inference_key, checker, dispatched) as client:
        response = await client.post("/v1/chat/completions", json=request)

    assert response.status_code == 200
    assert response.headers["content-type"].partition(";")[0] == "text/event-stream"
    events = stream_events(response.content)
    assert events[-1] == "[DONE]"
    summary = summarize_stream(events)
    content = summary["content"]
    assert content
    assert checker.text_calls[0] == ("input", USER_MESSAGE)
    assert "".join(value for kind, value in checker.text_calls if kind == "output") == content
    assert_recorded_request(record_mode, recorded_cassette_path, request)
    if record_mode == "none":
        await assert_allowed_relay_is_transparent(nvidia_inference_key, request, response, dispatched, vcr, stream=True)
        provider_response = recorded_provider_response(recorded_cassette_path)
        assert response.status_code == provider_response["status"]["code"]
        assert normalize_provider_metadata(events) == provider_response["body"]["parsed_body"]
    assert summary == snapshot(
        {
            "content": """\
Here are a few short greeting options you can use. Pick the tone you want:

- Casual: Hi there! / Hey!
- Neutral: Hello!
- Friendly: Hi, nice to meet you.
- Polite/Professional: Hello, nice to meet you. / Hello, how can I help today?

What context are you using it for (text, email, in person) and what tone should it have? I can tailor it.\
""",
            "model": "openai/openai/gpt-5-nano",
            "finish_reason": "stop",
            "usage": {
                "completion_tokens": 739,
                "prompt_tokens": 17,
                "total_tokens": 756,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 640,
                    "rejected_prediction_tokens": 0,
                },
                "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
                "cost": 0.00021759429999999998,
            },
            "event_kinds": ["content", "done", "finish", "role", "usage"],
        }
    )


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
async def test_real_provider_error_is_preserved_before_output_inspection(
    nvidia_inference_key, record_mode, recorded_cassette_path, stream, checker, vcr
):
    request = chat_request(model=INVALID_OPENAI_MODEL, stream=stream)
    dispatched = []
    async with recorded_proxy(nvidia_inference_key, checker, dispatched) as client:
        response = await client.post("/v1/chat/completions", json=request)

    assert 400 <= response.status_code <= 499
    assert checker.text_calls == [("input", USER_MESSAGE)]
    assert_recorded_request(record_mode, recorded_cassette_path, request)
    if record_mode == "none":
        await assert_allowed_relay_is_transparent(
            nvidia_inference_key, request, response, dispatched, vcr, stream=stream
        )
        provider_response = recorded_provider_response(recorded_cassette_path)
        assert response.status_code == provider_response["status"]["code"]
        assert normalize_provider_metadata(response.json()) == provider_response["body"]["parsed_body"]
    assert {
        "status_code": response.status_code,
        "content_type": response.headers["content-type"].partition(";")[0],
        "body": normalize_provider_metadata(response.json()),
    } == snapshot(
        {
            "status_code": 403,
            "content_type": "application/json",
            "body": {
                "error": {
                    "message": "key not allowed to access model. This key can only access models=['default-models']. Tried to access openai/openai/recorded-invalid-model",
                    "type": "key_model_access_denied",
                    "param": "model",
                    "code": "403",
                }
            },
        }
    )


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
async def test_nvidia_chat_source_profile_preserves_allowed_relay(
    nvidia_inference_key, record_mode, recorded_cassette_path, stream, checker, vcr
):
    request = chat_request(stream=stream)
    request["messages"][0]["content"] = "Give a one-sentence greeting for a coworker."
    dispatched = []
    async with recorded_proxy(nvidia_inference_key, checker, dispatched) as client:
        response = await client.post("/v1/chat/completions", json=request)

    assert response.status_code == 200
    assert checker.text_calls[0] == ("input", request["messages"][0]["content"])
    assert_recorded_request(record_mode, recorded_cassette_path, request)
    if stream:
        events = stream_events(response.content)
        assert events[-1] == "[DONE]"
        content = summarize_stream(events)["content"]
        assert content
        assert "".join(value for kind, value in checker.text_calls if kind == "output") == content
    else:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        assert content
        assert checker.text_calls == [("input", request["messages"][0]["content"]), ("output", content)]

    if record_mode == "none":
        await assert_allowed_relay_is_transparent(
            nvidia_inference_key, request, response, dispatched, vcr, stream=stream
        )


@pytest.mark.vcr(BUFFERED_RECORDING, record_mode="none")
async def test_recorded_buffered_output_is_withheld_on_block(nvidia_inference_key, vcr_cassette_dir, checker):
    checker.output_decisions = [ContentBlocked("blocked by test")]
    provider_body = recorded_provider_response(Path(vcr_cassette_dir) / BUFFERED_RECORDING)["body"]["parsed_body"]
    content = provider_body["choices"][0]["message"]["content"]
    dispatched = []
    async with recorded_proxy(nvidia_inference_key, checker, dispatched) as client:
        response = await client.post("/v1/chat/completions", json=chat_request())

    assert len(dispatched) == 1
    assert checker.text_calls == [("input", USER_MESSAGE), ("output", content)]
    assert response.status_code == 400
    assert response.json() == BLOCKED_ERROR
    assert content not in response.text


@pytest.mark.vcr(STREAMING_RECORDING, record_mode="none")
@pytest.mark.parametrize("allowed_windows", [0, 1], ids=["first-window", "later-window"])
@pytest.mark.parametrize("chunk_size,context_size", [(1, 0), (2, 1)], ids=["single-delta", "overlapping-context"])
async def test_recorded_stream_releases_only_approved_prefix(
    nvidia_inference_key, vcr_cassette_dir, checker, allowed_windows, chunk_size, context_size
):
    checker.policy = ContentInspectionPolicy(True, True, StreamBufferingPolicy(chunk_size, context_size))
    checker.output_decisions = [ContentAllowed()] * allowed_windows + [ContentBlocked("blocked by test")]
    provider_events = recorded_provider_response(Path(vcr_cassette_dir) / STREAMING_RECORDING)["body"]["parsed_body"]
    text_events = [
        (index, event["choices"][0]["delta"]["content"])
        for index, event in enumerate(provider_events)
        if isinstance(event, dict) and event.get("choices") and event["choices"][0].get("delta", {}).get("content")
    ]
    assert len(text_events) > (allowed_windows + 1) * chunk_size
    approved_deltas = allowed_windows * chunk_size
    released_end = text_events[approved_deltas - 1][0] + 1 if approved_deltas else 0
    dispatched = []
    async with recorded_proxy(nvidia_inference_key, checker, dispatched) as client:
        response = await client.post("/v1/chat/completions", json=chat_request(stream=True))

    assert len(dispatched) == 1
    assert checker.text_calls == [("input", USER_MESSAGE)] + [
        (
            "output",
            "".join(
                text for _, text in text_events[max(0, window * chunk_size - context_size) : (window + 1) * chunk_size]
            ),
        )
        for window in range(allowed_windows + 1)
    ]
    assert all(check.input_message.content == USER_MESSAGE for kind, check in checker.calls if kind == "output")
    assert response.status_code == 200
    assert response.headers["content-type"].partition(";")[0] == "text/event-stream"
    received = stream_events(response.content)
    assert received[-1] == "[DONE]"
    assert received[-2] == BLOCKED_ERROR
    assert received[:-2] == provider_events[:released_end]

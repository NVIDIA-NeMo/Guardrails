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
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import FastAPI
from jsonschema import Draft202012Validator

from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentInspectionPolicy,
    StreamBufferingPolicy,
)
from nemoguardrails.server.experimental._guarded_proxy import create_guarded_http_operation
from nemoguardrails.server.experimental._http_kernel import create_http_proxy_router
from nemoguardrails.server.experimental._provider_transport import ProviderTransport
from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.provider.integration import UpstreamSettings
from nemoguardrails.server.experimental.provider.sse import iter_sse_events
from nemoguardrails.server.experimental.provider.stream import StreamEventRole
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.responses import LITELLM_RESPONSES_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.errors import OPENAI_ERROR_MAPPING
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT
from tests.recorded.cassette import cassette_request_jsons

pytestmark = [
    pytest.mark.recorded,
    pytest.mark.vcr(ignore_hosts=["recorded-proxy.test"], allow_playback_repeats=True),
    pytest.mark.asyncio,
]

BASE_URL = "https://inference-api.nvidia.com/v1"
MODEL = "openai/openai/gpt-5-nano"
INPUT = "Reply with exactly: Hello."
ENDPOINT = LITELLM_RESPONSES_ENDPOINT


def responses_request(stream, *, model=MODEL):
    return {
        "model": model,
        "input": INPUT,
        "stream": stream,
        "store": False,
        "max_output_tokens": 256,
        "reasoning": {"effort": "minimal"},
    }


class Checker:
    def __init__(self):
        self.calls = []

    def inspection_policy(self):
        return ContentInspectionPolicy(True, True, StreamBufferingPolicy(1, 0))

    async def check_input(self, check):
        self.calls.append(("input", check.message.content))
        return ContentAllowed()

    async def check_output(self, check):
        self.calls.append(("output", check.output_content))
        return ContentAllowed()


@asynccontextmanager
async def proxy(api_key, checker, dispatches):
    async with httpx.AsyncClient(timeout=60) as upstream:
        transport = ProviderTransport(UpstreamSettings(base_url=BASE_URL, mount_path="/v1"), upstream)

        async def buffered(request):
            dispatches.append(request)
            return await transport.dispatch(request, require_identity_response=True)

        async def streaming(request):
            dispatches.append(request)
            return await transport.stream_dispatch(request, require_identity_response=True)

        app = FastAPI()
        app.include_router(
            create_http_proxy_router(
                operations=(
                    create_guarded_http_operation(
                        ENDPOINT,
                        OPENAI_ERROR_MAPPING,
                        stream_dispatch=streaming,
                        max_stream_event_bytes=1024 * 1024,
                        max_pending_stream_bytes=1024 * 1024,
                    ),
                ),
                checker=checker,
                dispatch=buffered,
                render_outcome=OPENAI_ERROR_MAPPING.renderer,
            )
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://recorded-proxy.test",
            headers={"authorization": f"Bearer {api_key}"},
        ) as client:
            yield client


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
        assert payload.keys() - base_document["response"]["properties"].keys() == {
            "access_programs",
            "billing",
            "frequency_penalty",
            "presence_penalty",
            "store",
            "tool_usage",
        }
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
                assert payload["response"].keys() - base_document["response"]["properties"].keys() == {
                    "frequency_penalty",
                    "presence_penalty",
                    "store",
                    "tool_usage",
                }
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
    nvidia_inference_key, record_mode, recorded_cassette_path, stream
):
    request = responses_request(stream)
    upstream = await capture(nvidia_inference_key, request)
    assert upstream.status_code == 200
    if record_mode != "none":
        return

    assert cassette_request_jsons(recorded_cassette_path) == [request]
    text = await validate_inventory(upstream, request)
    assert text
    checker = Checker()
    dispatches = []
    async with proxy(nvidia_inference_key, checker, dispatches) as client:
        response = await client.post("/v1/responses", json=request)
    assert response.status_code == upstream.status_code
    assert response.content == upstream.content
    assert len(dispatches) == 1
    assert json.loads(dispatches[0].body) == request
    assert checker.calls[0] == ("input", INPUT)
    assert "".join(value for kind, value in checker.calls if kind == "output") == text

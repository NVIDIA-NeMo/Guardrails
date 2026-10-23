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
from pathlib import Path

import httpx
import pytest
import yaml

from nemoguardrails.server.experimental._content_checker import ContentInspectionPolicy, StreamBufferingPolicy
from nemoguardrails.server.experimental.provider.sse import iter_sse_events
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.responses import LITELLM_RESPONSES_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT
from tests.recorded.cassette import cassette_with_rehydrated_bodies
from tests.server.experimental.checkers import StaticChecker
from tests.server.experimental.responses_proxy import responses_proxy

ROOT = Path(__file__).resolve().parents[6]
INVENTORY_PATH = Path(__file__).parent / "fixtures/source_inventory.json"
INVENTORY = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
RECORDINGS = ROOT / "tests/recorded/server/experimental/openai/cassettes"
UNINSPECTED = "synthetic uninspected probe text"
ENDPOINTS = [pytest.param(RESPONSES_ENDPOINT, id="openai"), pytest.param(LITELLM_RESPONSES_ENDPOINT, id="litellm")]


def recording(stream, *, gateway=False):
    mode = "streaming" if stream else "buffered"
    directory = "test_litellm_responses" if gateway else "test_responses"
    name = (
        "test_gateway_responses_preserve_policy_and_wire"
        if gateway
        else "test_native_responses_preserve_policy_and_wire"
    )
    path = RECORDINGS / directory / f"{name}[{mode}].yaml"
    (interaction,) = cassette_with_rehydrated_bodies(yaml.safe_load(path.read_text(encoding="utf-8")))["interactions"]
    assert interaction["response"]["status"]["code"] == 200
    request = json.loads(interaction["request"]["body"])
    body = interaction["response"]["body"]["string"]
    return request, body.encode() if isinstance(body, str) else body


async def events(body):
    async def chunks():
        yield body

    return [event async for event in iter_sse_events(chunks(), max_event_bytes=1024 * 1024)]


def sse_bytes(payloads):
    return b"".join(
        b"event: " + payload["type"].encode() + b"\ndata: " + json.dumps(payload).encode() + b"\n\n"
        for payload in payloads
    )


async def injected_response(request, body, *, endpoint=RESPONSES_ENDPOINT, inspect=True):
    sent = []

    async def upstream(outbound):
        sent.append(outbound)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream" if request["stream"] else "application/json"},
            stream=httpx.ByteStream(body),
        )

    checker = StaticChecker(
        ContentInspectionPolicy(inspect, inspect, StreamBufferingPolicy(100, 0) if inspect else None)
    )
    dispatches = []
    async with responses_proxy(
        "synthetic-probe", checker, dispatches, http_transport=httpx.MockTransport(upstream), endpoint=endpoint
    ) as client:
        response = await client.post("/v1/responses", json=request)
    assert len(sent) == len(dispatches)
    return response, checker, sent


async def assert_rejected(response, checker, request):
    assert checker.text_calls == [("input", request["input"])]
    assert UNINSPECTED not in response.text
    if request["stream"]:
        assert response.status_code == 200
        (error,) = await events(response.content)
        assert error.event_type == b"error"
        assert json.loads(error.data)["code"] == "unsupported_responses_response_shape"
    else:
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "unsupported_responses_response_shape"

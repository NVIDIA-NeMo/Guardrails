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

"""Chat-wire HTTP harness shared by baseline and dialect conformance cases."""

import json
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from nemoguardrails.server.experimental._content_checker import ContentInspectionPolicy, StreamBufferingPolicy
from nemoguardrails.server.experimental._provider_transport import ProviderTransport
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_chat_router
from tests.server.experimental.checkers import StaticChecker
from tests.server.experimental.conformance.openai.chat_completions.support import source_validator


class ResponseBytes(httpx.AsyncByteStream):
    def __init__(self, body):
        self.body = body
        self.closed = False

    async def __aiter__(self):
        for offset in range(0, len(self.body), 7):
            yield self.body[offset : offset + 7]

    async def aclose(self):
        self.closed = True


@asynccontextmanager
async def proxy(checker, body, *, streaming=False, endpoint=None):
    """Observe requests after real dispatch construction; use no live sockets."""
    requests = []
    source = ResponseBytes(body)

    async def upstream(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream" if streaming else "application/json"},
            stream=source,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
        transport = ProviderTransport(upstream_client, base_url="https://provider.test")
        app = FastAPI()
        app.include_router(
            create_openai_chat_router(
                endpoint=endpoint,
                checker=checker,
                dispatch=transport.dispatch,
                stream_dispatch=transport.stream_dispatch,
            )
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
            yield client, requests, source


def checker_for(inspect):
    return StaticChecker(ContentInspectionPolicy(inspect, inspect, StreamBufferingPolicy(1, 0) if inspect else None))


def assert_native_error(response, status, code):
    assert response.status_code == status
    assert set(response.json()) == {"error"}
    source_validator("ErrorResponse").validate(response.json())
    assert response.json()["error"]["code"] == code


def assert_output_error(response, streaming):
    code = "unsupported_chat_completions_response_shape"
    if not streaming:
        assert_native_error(response, 502, code)
        return
    assert response.status_code == 200
    error, terminal, trailing = response.content.split(b"\n\n")
    assert terminal == b"data: [DONE]"
    assert trailing == b""
    payload = json.loads(error.removeprefix(b"data: "))
    assert set(payload) == {"error"}
    source_validator("ErrorResponse").validate(payload)
    assert payload["error"]["code"] == code

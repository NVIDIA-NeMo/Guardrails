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

from contextlib import asynccontextmanager

import anyio
import httpx
import pytest
from starlette.requests import Request

from nemoguardrails.server.experimental import _stream_cleanup
from nemoguardrails.server.experimental._provider_transport import ProviderTransport
from nemoguardrails.server.experimental.provider.integration import UpstreamSettings
from nemoguardrails.server.experimental.routers.passthrough import _passthrough_response
from tests.server.experimental.test_httpx_response_cleanup import CheckpointStream


def request_scope(method="GET"):
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": "/models",
        "raw_path": b"/models",
        "query_string": b"",
        "headers": [],
        "server": ("offline.test", 80),
        "client": ("offline-client", 1),
    }


async def receive_request():
    return {"type": "http.request", "body": b"", "more_body": False}


@asynccontextmanager
async def passthrough_response(stream):
    async def upstream(request):
        return httpx.Response(200, request=request, headers={"content-type": "application/octet-stream"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream), trust_env=False) as client:
        transport = ProviderTransport(UpstreamSettings("https://offline.test"), client, max_response_body_bytes=1)
        try:
            yield await _passthrough_response(Request(request_scope(), receive_request), transport)
        finally:
            stream.close_mode = "checkpoint"
            await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_at", ["read", "startup", "send"])
async def test_passthrough_response_owns_cleanup_through_asgi_failures(failure_at):
    stream = CheckpointStream(chunks=(b"opaque\x00bytes",), wait=True)
    async with passthrough_response(stream) as response:

        async def receive():
            await stream.read_started.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if failure_at == "startup" and message["type"] == "http.response.start":
                stream.read_started.set()
                await anyio.sleep_forever()
            if failure_at == "send" and message["type"] == "http.response.body":
                raise RuntimeError("downstream send failed")

        with anyio.fail_after(1):
            if failure_at == "send":
                with pytest.raises(RuntimeError, match="downstream send failed"):
                    await response(request_scope(), receive, send)
            else:
                await response(request_scope(), receive, send)
        assert stream.close_completed


@pytest.mark.asyncio
@pytest.mark.parametrize("close_mode", ["checkpoint", "hang", "raise"])
async def test_passthrough_eof_keeps_raw_bytes_and_finishes_response(monkeypatch, close_mode):
    monkeypatch.setattr(_stream_cleanup, "_STREAM_CLOSE_TIMEOUT_SECONDS", 0.01)
    stream = CheckpointStream(chunks=(b"not SSE\x00", b"arbitrary bytes"), close_mode=close_mode)
    messages = []
    async with passthrough_response(stream) as response:

        async def receive():
            await anyio.sleep_forever()

        async def send(message):
            messages.append(message)

        with anyio.fail_after(1):
            await response(request_scope(), receive, send)
        assert b"".join(message.get("body", b"") for message in messages) == b"not SSE\x00arbitrary bytes"
        assert messages[-1]["more_body"] is False
        assert stream.close_completed == (close_mode == "checkpoint")


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [204, 205, 304])
async def test_bodyless_passthrough_closes_under_cancellation(status_code):
    stream = CheckpointStream()
    cancel_scope = None

    async def upstream(request):
        response = httpx.Response(status_code, request=request, stream=stream)
        assert cancel_scope is not None
        cancel_scope.cancel()
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream), trust_env=False) as client:
        with anyio.CancelScope() as scope:
            cancel_scope = scope
            transport = ProviderTransport(UpstreamSettings("https://offline.test"), client)
            await _passthrough_response(Request(request_scope(), receive_request), transport)
        assert stream.close_completed

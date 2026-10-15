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
from contextlib import asynccontextmanager

import anyio
import httpx
import pytest

from nemoguardrails.server.experimental import _provider_transport, _stream_cleanup
from nemoguardrails.server.experimental._content_checker import ContentAllowed, StreamBufferingPolicy
from nemoguardrails.server.experimental._http_kernel import BufferedHttpRequest, BufferedHttpResponse
from nemoguardrails.server.experimental._httpx_response import _HttpxResponseBody
from nemoguardrails.server.experimental._provider_transport import ProviderTransport
from nemoguardrails.server.experimental._streaming_http import execute_streaming_http
from nemoguardrails.server.experimental.provider.integration import UpstreamSettings
from nemoguardrails.server.experimental.provider.types import GuardedMessage
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT


class CheckpointStream(httpx.AsyncByteStream):
    def __init__(self, *, chunks=(), wait=False, read_error=None, close_mode="checkpoint"):
        self.chunks = chunks
        self.wait = wait
        self.read_error = read_error
        self.close_mode = close_mode
        self.read_started = anyio.Event()
        self.close_calls = 0
        self.close_completed = False
        self.active_closers = 0
        self.max_active_closers = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.read_error is not None:
            raise self.read_error
        if self.wait:
            self.read_started.set()
            await anyio.sleep_forever()

    async def aclose(self):
        self.close_calls += 1
        self.active_closers += 1
        self.max_active_closers = max(self.max_active_closers, self.active_closers)
        try:
            if self.close_mode == "hang":
                await anyio.sleep_forever()
            if self.close_mode == "raise":
                raise RuntimeError("private close details")
            await anyio.sleep(0)
            self.close_completed = True
        finally:
            self.active_closers -= 1


@asynccontextmanager
async def opened(stream):
    async def upstream(request):
        return httpx.Response(200, request=request, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream), trust_env=False) as client:
        response = await client.send(client.build_request("GET", "https://offline.test"), stream=True)
        try:
            yield response, client
        finally:
            stream.close_mode = "checkpoint"
            await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", ["read", "close"])
async def test_httpx_owner_finishes_cleanup_under_cancellation(cancel_at):
    stream = CheckpointStream(wait=True)
    async with opened(stream) as (response, _client):
        body = _HttpxResponseBody(response)
        with anyio.CancelScope() as scope:
            scope.cancel()
            if cancel_at == "read":
                await body.__anext__()
            else:
                await body.aclose()
        await body.aclose()
        assert stream.close_completed
        assert response.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("close_mode", ["hang", "raise"])
async def test_httpx_owner_can_retry_incomplete_cleanup(monkeypatch, caplog, close_mode):
    monkeypatch.setattr(_stream_cleanup, "_STREAM_CLOSE_TIMEOUT_SECONDS", 0.01)
    stream = CheckpointStream(close_mode=close_mode)
    async with opened(stream) as (response, _client):
        body = _HttpxResponseBody(response)
        with anyio.fail_after(1):
            await body.aclose()
        assert response.is_closed
        assert not stream.close_completed
        previous_calls = stream.close_calls
        stream.close_mode = "checkpoint"
        await body.aclose()
        assert stream.close_completed
        assert stream.close_calls > previous_calls
        assert "Stream cleanup" in caplog.text
        assert "private close details" not in caplog.text


@pytest.mark.asyncio
async def test_httpx_owner_serializes_close_attempts():
    stream = CheckpointStream()
    async with opened(stream) as (response, _client):
        body = _HttpxResponseBody(response)
        async with anyio.create_task_group() as group:
            group.start_soon(body.aclose)
            group.start_soon(body.aclose)
        assert stream.close_completed
        assert stream.max_active_closers == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("close_mode", ["checkpoint", "hang", "raise"])
async def test_httpx_eof_retains_bytes_and_bounds_cleanup(monkeypatch, close_mode):
    monkeypatch.setattr(_stream_cleanup, "_STREAM_CLOSE_TIMEOUT_SECONDS", 0.01)
    stream = CheckpointStream(chunks=(b"one", b"two"), close_mode=close_mode)
    async with opened(stream) as (response, _client):
        body = _HttpxResponseBody(response)
        with anyio.fail_after(1):
            received = [chunk async for chunk in body]
        assert received == [b"one", b"two"]
        assert response.is_closed
        assert response.num_bytes_downloaded == 6
        assert stream.close_completed == (close_mode == "checkpoint")


@pytest.mark.asyncio
async def test_httpx_cleanup_failure_preserves_the_read_failure():
    stream = CheckpointStream(read_error=httpx.ReadError("primary read failure"), close_mode="raise")
    async with opened(stream) as (response, _client):
        with pytest.raises(httpx.ReadError, match="primary read failure"):
            await _HttpxResponseBody(response).__anext__()


@pytest.mark.asyncio
async def test_buffered_httpx_cleanup_finishes_under_cancellation():
    stream = CheckpointStream(wait=True)
    async with opened(stream) as (response, client):
        dispatcher = ProviderTransport(UpstreamSettings("https://offline.test"), client)
        with anyio.CancelScope() as scope:
            scope.cancel()
            await dispatcher._buffer(response)
        assert stream.close_completed


@pytest.mark.asyncio
async def test_dispatch_closes_acquired_response_when_construction_fails(monkeypatch):
    stream = CheckpointStream()

    async def upstream(request):
        return httpx.Response(200, request=request, stream=stream)

    def reject_headers(_headers):
        raise ValueError("response construction failed")

    monkeypatch.setattr(_provider_transport, "_response_headers", reject_headers)
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream), trust_env=False) as client:
        dispatcher = ProviderTransport(UpstreamSettings("https://offline.test"), client)
        request = BufferedHttpRequest("GET", "/models", b"/models", b"", (), b"")
        with pytest.raises(ValueError, match="response construction failed"):
            await dispatcher.stream_dispatch(request)
        assert stream.close_completed


class AllowingChecker:
    async def check_output(self, _check):
        return ContentAllowed()


@pytest.mark.asyncio
@pytest.mark.parametrize("inspect_output", [False, True])
@pytest.mark.parametrize("failure_at", ["read", "startup", "send"])
async def test_httpx_owner_cleanup_survives_asgi_disconnect_and_send_failure(inspect_output, failure_at):
    event = (
        b"data: "
        + json.dumps(
            {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "answer"}}]}
        ).encode()
        + b"\n\n"
    )
    stream = CheckpointStream(chunks=(event,), wait=True)
    async with opened(stream) as (upstream, _client):
        body = _HttpxResponseBody(upstream)

        async def dispatch(_request):
            return _provider_transport.StreamingHttpResponse(200, ((b"content-type", b"text/event-stream"),), body)

        binding = CHAT_COMPLETIONS_ENDPOINT.stream
        assert binding is not None
        response = await execute_streaming_http(
            BufferedHttpRequest("POST", "/v1/chat/completions", b"/v1/chat/completions", b"", (), b"{}"),
            dispatch=dispatch,
            checker=AllowingChecker(),
            streaming_policy=StreamBufferingPolicy(1, 0) if inspect_output else None,
            input_message=GuardedMessage("user", "question"),
            adapter=binding.create_adapter(),
            render_outcome=lambda _outcome: BufferedHttpResponse(502, (), b"error"),
            max_event_bytes=1024,
            max_pending_bytes=1024,
        )

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
                    await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
            else:
                await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
        assert stream.close_completed

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

"""Exercise the module launcher with real sockets and a synthetic upstream.

Adapted from the preserved runnable-first prototype, using the current shared
assembly and declarative manifest rather than its earlier single-provider API.
The synthetic protocol events are test fixtures, not provider recordings.
"""

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import httpx
import pytest
import yaml
from aiohttp import web

from tests.server.experimental.test_application_configuration import request_payload
from tests.server.experimental.test_openai_responses_projections import response_payload
from tests.server.experimental.test_openai_responses_streaming_projections import event_bytes

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "nemoguardrails/server/experimental/examples/basic"
LAUNCHER_TIMEOUT_SECONDS = 30


def chat_response(text="safe"):
    """Build a minimal synthetic Chat response."""
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


def stream_bytes(operation, parts, *, complete=True):
    """Encode valid text events using each operation's actual protocol."""
    if operation == "responses":
        events = [event_bytes("response.created", response={"output": []}, sequence_number=0)]
        events.extend(
            event_bytes(
                "response.output_text.delta",
                delta=text,
                item_id="msg_test",
                output_index=1,
                content_index=0,
                sequence_number=index,
            )
            for index, text in enumerate(parts, 1)
        )
        if complete:
            events.append(
                event_bytes(
                    "response.completed", response=response_payload("".join(parts)), sequence_number=len(parts) + 1
                )
            )
        return b"".join(events)
    events = [{"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}]
    events.extend({"choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}]} for part in parts)
    if complete:
        events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    wire = b"".join(
        b"data: " + json.dumps({"object": "chat.completion.chunk", **event}).encode() + b"\n\n" for event in events
    )
    return wire + (b"data: [DONE]\n\n" if complete else b"")


@asynccontextmanager
async def synthetic_upstream():
    """Serve deterministic bytes and observe upstream socket disconnection."""
    requests = []
    disconnected = asyncio.Event()

    async def generation(request):
        payload = await request.json()
        requests.append(payload)
        operation = "responses" if request.path.endswith("/responses") else "chat_completions"
        text = payload["input"] if operation == "responses" else payload["messages"][0]["content"]
        answer = "person@example.com" if text == "emit-blocked-output" else "safe"
        if not payload.get("stream"):
            return web.json_response(response_payload(answer) if operation == "responses" else chat_response(answer))
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        try:
            if text == "disconnect":
                await response.write(stream_bytes(operation, ["sa", "fe"] * 100, complete=False))
                while True:
                    await asyncio.sleep(0.01)
                    await response.write(b": keepalive\n\n")
            parts = ["person@", "example.com"] if answer == "person@example.com" else ["sa", "fe"]
            if text == "emit-blocked-window":
                parts = ["safe "] * 198 + ["person@", "example.com"]
            complete = text != "emit-blocked-window"
            for event in stream_bytes(operation, parts, complete=complete).split(b"\n\n"):
                if event:
                    await response.write(event + b"\n\n")
                    await asyncio.sleep(0.005)
            if not complete:
                while True:
                    await asyncio.sleep(0.01)
                    await response.write(b": keepalive\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            disconnected.set()
        return response

    async def models(request):
        requests.append({"path": request.path})
        response = web.StreamResponse(headers={"content-type": "application/octet-stream"})
        await response.prepare(request)
        await response.write(b"safe")
        try:
            while True:
                await asyncio.sleep(0.01)
                await response.write(b"opaque")
        except (ConnectionResetError, asyncio.CancelledError):
            disconnected.set()
        return response

    application = web.Application()
    application.router.add_post("/v1/chat/completions", generation)
    application.router.add_post("/v1/responses", generation)
    application.router.add_get("/v1/models", models)
    runner = web.AppRunner(application, shutdown_timeout=1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    try:
        await site.start()
        yield f"http://127.0.0.1:{runner.addresses[0][1]}/v1", requests, disconnected
    finally:
        await runner.cleanup()


@asynccontextmanager
async def launched_proxy(tmp_path, upstream, dialect=None, *, timeout_graceful_shutdown=None):
    """Launch the documented module, wait for readiness, and reap the child."""
    manifest = yaml.safe_load((EXAMPLE / "proxy.yaml").read_text(encoding="utf-8"))
    manifest["rails_configs"][0]["config_path"] = str(EXAMPLE / "rails")
    manifest["providers"][0]["base_url"] = upstream
    if dialect is not None:
        manifest["providers"][0]["dialect"] = dialect
    path = tmp_path / "proxy.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    environment = dict(os.environ)
    for key in (
        "OPENAI_API_KEY",
        "NVIDIA_API_KEY",
        "NVIDIA_INFERENCE_KEY",
        "LIVE_TEST",
        "LIVE_TEST_MODE",
        "TEST_LIVE_MODE",
    ):
        environment.pop(key, None)
    environment["NEMO_GUARDRAILS_NO_USAGE_STATS"] = "1"
    environment["PYTHONPATH"] = str(ROOT)
    shutdown_arguments = (
        [] if timeout_graceful_shutdown is None else ["--timeout-graceful-shutdown", str(timeout_graceful_shutdown)]
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "nemoguardrails.server.experimental",
        "serve",
        str(path),
        "--port",
        "0",
        *shutdown_arguments,
        cwd=tmp_path,
        env=environment,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    lines = []
    drain = None
    ready = False
    try:
        assert process.stdout is not None
        with anyio.fail_after(LAUNCHER_TIMEOUT_SECONDS):
            while True:
                line = await process.stdout.readline()
                assert line, b"".join(lines).decode()
                lines.append(line)
                match = re.search(rb"Uvicorn running on (http://127\.0\.0\.1:\d+)", line)
                if match:
                    ready = True
                    url = match[1].decode()
                    break
        drain = asyncio.create_task(process.stdout.read())
        yield url, process
    finally:
        shutdown_timed_out = False
        if drain is None and process.stdout is not None:
            drain = asyncio.create_task(process.stdout.read())
        if process.returncode is None:
            try:
                if sys.platform == "win32":
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    process.terminate()
                await asyncio.wait_for(process.wait(), LAUNCHER_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                shutdown_timed_out = True
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        if drain is not None:
            lines.append(await drain)
        output = b"".join(lines).decode()
        if shutdown_timed_out:
            pytest.fail(f"launcher did not shut down within {LAUNCHER_TIMEOUT_SECONDS} seconds:\n{output}")
        if ready:
            if sys.platform != "win32":
                assert process.returncode in (0, -signal.SIGTERM), output
            assert "Application shutdown complete." in output, output
            assert "cleanup timed out" not in output.lower(), output
            assert "cleanup failed" not in output.lower(), output


@pytest.mark.asyncio
@pytest.mark.parametrize("dialect", [None, "litellm"])
async def test_localhost_launcher_buffered_streaming_and_blocking(tmp_path, dialect):
    async with synthetic_upstream() as (upstream, requests, _):
        async with launched_proxy(tmp_path, upstream, dialect) as (url, _):
            async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=5) as client:
                for operation in ("chat_completions", "responses"):
                    route = "/v1/responses" if operation == "responses" else "/v1/chat/completions"
                    for stream in (False, True):
                        allowed = await client.post(route, json=request_payload(operation, streaming=stream))
                        assert allowed.status_code == 200, allowed.text
                        if stream:
                            assert allowed.content == stream_bytes(operation, ["sa", "fe"])
                        else:
                            assert allowed.json() == (
                                response_payload("safe") if operation == "responses" else chat_response()
                            )
                        count = len(requests)
                        blocked_input = await client.post(
                            route, json=request_payload(operation, "Contact person@example.com", stream)
                        )
                        assert blocked_input.status_code == 400
                        assert blocked_input.json()["error"]["code"] == "content_blocked"
                        assert len(requests) == count
                        blocked_output = await client.post(
                            route, json=request_payload(operation, "emit-blocked-output", stream)
                        )
                        assert blocked_output.status_code == (200 if stream else 400)
                        assert "content_blocked" in blocked_output.text
                        assert b"person@" not in blocked_output.content
                        assert b"example.com" not in blocked_output.content
                    blocked_window = await client.post(
                        route, json=request_payload(operation, "emit-blocked-window", True)
                    )
                    assert blocked_window.status_code == 200
                    assert "content_blocked" in blocked_window.text
                    assert b"safe " not in blocked_window.content
                    assert b"person@" not in blocked_window.content
                    assert b"example.com" not in blocked_window.content
                assert len(requests) == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["chat_completions", "responses", "passthrough"])
async def test_localhost_disconnect_closes_upstream_socket(tmp_path, operation):
    async with synthetic_upstream() as (upstream, requests, disconnected):
        async with launched_proxy(tmp_path, upstream) as (url, _):
            async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=5) as client:
                path = {
                    "chat_completions": "/v1/chat/completions",
                    "responses": "/v1/responses",
                    "passthrough": "/v1/models",
                }[operation]
                passthrough = operation == "passthrough"
                kwargs = {} if passthrough else {"json": request_payload(operation, "disconnect", True)}
                request = httpx.Request("GET" if passthrough else "POST", httpx.URL(url).join(path), **kwargs)
                reader, writer = await asyncio.open_connection(request.url.host, request.url.port)
                try:
                    writer.write(
                        request.method.encode()
                        + b" "
                        + request.url.raw_path
                        + b" HTTP/1.1\r\n"
                        + b"".join(name + b": " + value + b"\r\n" for name, value in request.headers.raw)
                        + b"\r\n"
                        + request.content
                    )
                    await writer.drain()
                    received = b""
                    with anyio.fail_after(5):
                        headers = await reader.readuntil(b"\r\n\r\n")
                        assert headers.split(b"\r\n", 1)[0].split()[1] == b"200", headers
                        while True:
                            chunk = await reader.read(4096)
                            assert chunk, "the upstream ended before the downstream disconnected"
                            received += chunk
                            if (b"safe" if passthrough else b'"sa"') in received:
                                break
                    assert b'"error"' not in received
                    writer.write_eof()
                    with anyio.fail_after(5):
                        await reader.read()
                finally:
                    writer.close()
                    with anyio.fail_after(5):
                        await writer.wait_closed()
                with anyio.fail_after(5):
                    await disconnected.wait()
                assert len(requests) == 1
                allowed = await client.post("/v1/chat/completions", json=request_payload("chat_completions"))
                assert allowed.status_code == 200
                assert allowed.json() == chat_response()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["chat_completions", "responses", "passthrough"])
async def test_shutdown_cancels_active_stream_and_closes_upstream(tmp_path, operation):
    async with synthetic_upstream() as (upstream, requests, disconnected):
        async with launched_proxy(tmp_path, upstream, timeout_graceful_shutdown=1) as (url, process):
            async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=5) as client:
                path = {
                    "chat_completions": "/v1/chat/completions",
                    "responses": "/v1/responses",
                    "passthrough": "/v1/models",
                }[operation]
                passthrough = operation == "passthrough"
                kwargs = {} if passthrough else {"json": request_payload(operation, "disconnect", True)}
                async with client.stream("GET" if passthrough else "POST", path, **kwargs) as response:
                    assert response.status_code == 200
                    chunks = response.aiter_raw()
                    assert await anext(chunks)
                    assert not disconnected.is_set()
                    assert process.returncode is None
                    if sys.platform == "win32":
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        process.terminate()
                    with anyio.fail_after(10):
                        await process.wait()
                        await disconnected.wait()
                assert len(requests) == 1

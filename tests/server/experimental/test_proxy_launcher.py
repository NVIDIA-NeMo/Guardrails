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
        answer = "BLOCK_OUTPUT" if text == "emit-blocked-output" else "safe"
        if not payload.get("stream"):
            return web.json_response(response_payload(answer) if operation == "responses" else chat_response(answer))
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        try:
            if text == "disconnect":
                await response.write(stream_bytes(operation, ["sa", "fe"], complete=False))
                while True:
                    await asyncio.sleep(0.01)
                    await response.write(b": keepalive\n\n")
            parts = ["BLOCK_", "OUTPUT"] if answer == "BLOCK_OUTPUT" else ["sa", "fe"]
            for event in stream_bytes(operation, parts).split(b"\n\n"):
                if event:
                    await response.write(event + b"\n\n")
                    await asyncio.sleep(0.005)
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
async def launched_proxy(tmp_path, upstream, dialect=None):
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
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "nemoguardrails.server.experimental",
        "serve",
        str(path),
        "--port",
        "0",
        cwd=tmp_path,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    lines = []
    drain = None
    ready = False
    try:
        assert process.stdout is not None
        with anyio.fail_after(30):
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
        yield url
    finally:
        if drain is None and process.stdout is not None:
            drain = asyncio.create_task(process.stdout.read())
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                pytest.fail("launcher did not shut down within 10 seconds")
        if drain is not None:
            lines.append(await drain)
        output = b"".join(lines).decode()
        if ready:
            assert process.returncode in (0, -signal.SIGTERM), output
            assert "Application shutdown complete." in output, output


@pytest.mark.asyncio
@pytest.mark.parametrize("dialect", [None, "litellm"])
async def test_localhost_launcher_buffered_streaming_and_blocking(tmp_path, dialect):
    async with synthetic_upstream() as (upstream, requests, _):
        async with launched_proxy(tmp_path, upstream, dialect) as url:
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
                        blocked_input = await client.post(route, json=request_payload(operation, "BLOCK_INPUT", stream))
                        assert blocked_input.status_code == 400
                        assert blocked_input.json()["error"]["code"] == "content_blocked"
                        assert len(requests) == count
                        blocked_output = await client.post(
                            route, json=request_payload(operation, "emit-blocked-output", stream)
                        )
                        assert blocked_output.status_code == (200 if stream else 400)
                        assert "content_blocked" in blocked_output.text
                        assert b"BLOCK_" not in blocked_output.content
                        assert b"OUTPUT" not in blocked_output.content
                assert len(requests) == 8


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["chat_completions", "responses", "passthrough"])
async def test_localhost_disconnect_closes_upstream_socket(tmp_path, operation):
    async with synthetic_upstream() as (upstream, requests, disconnected):
        async with launched_proxy(tmp_path, upstream) as url:
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
                    received = b""
                    with anyio.fail_after(5):
                        async for chunk in response.aiter_bytes():
                            received += chunk
                            if (b"safe" if passthrough else b'"sa"') in received:
                                break
                        else:
                            pytest.fail("the upstream ended before the downstream disconnected")
                    assert b'"error"' not in received
                with anyio.fail_after(5):
                    await disconnected.wait()
                assert len(requests) == 1
                allowed = await client.post("/v1/chat/completions", json=request_payload("chat_completions"))
                assert allowed.status_code == 200
                assert allowed.json() == chat_response()

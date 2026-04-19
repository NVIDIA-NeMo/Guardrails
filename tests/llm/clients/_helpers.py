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
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient
from nemoguardrails.llm.models.openai_chat import OpenAIChatModel

_FIXTURES_DIR = Path(__file__).parent / "fixtures"

LIVE_TEST_MODE = bool(os.environ.get("LIVE_TEST_MODE") or os.environ.get("TEST_LIVE_MODE"))

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"


def make_client(**kwargs):
    return OpenAICompatibleClient(base_url="https://api.openai.com/v1", api_key="sk-test", **kwargs)


def ok_response():
    return {
        "id": "chatcmpl-123",
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"content": "Hello", "role": "assistant"}, "finish_reason": "stop"}],
    }


def mock_httpx_post(client, responses):
    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        status, body, headers = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return httpx.Response(
            status,
            json=body if isinstance(body, dict) else None,
            text=body if isinstance(body, str) else None,
            headers=headers or {},
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

    client._client = type("MockClient", (), {"post": mock_post})()
    return lambda: call_count


def mock_stream(lines, response_headers=None):
    hdrs = response_headers or {}

    @asynccontextmanager
    async def mock(*args, **kwargs):
        class FakeResponse:
            status_code = 200
            headers = hdrs

            async def aread(self):
                pass

            async def aiter_lines(self):
                for line in lines:
                    yield line

        yield FakeResponse()

    return mock


def stream_client(lines, response_headers=None):
    client = make_client()
    client._client = type("MockClient", (), {"stream": mock_stream(lines, response_headers)})()
    return client


async def consume(client):
    chunks = []
    async for chunk in client.stream_chat_completion("gpt-4o", [{"role": "user", "content": "Hi"}]):
        chunks.append(chunk)
    return chunks


def low_retry_delay(*args, **kwargs):
    return 0.0


def tracking_mock_stream(responses):
    aread_calls = []
    call_idx = [0]

    @asynccontextmanager
    async def mock(*args, **kwargs):
        idx = call_idx[0]
        call_idx[0] += 1
        status, body_lines, headers = responses[min(idx, len(responses) - 1)]

        class FakeResponse:
            status_code = status

            def __init__(self):
                self.headers = headers or {}
                self.text = body_lines[0] if not isinstance(body_lines, list) else ""

            async def aread(self_response):
                aread_calls.append(self_response.status_code)

            async def aiter_lines(self):
                if isinstance(body_lines, list):
                    for line in body_lines:
                        yield line

        yield FakeResponse()

    return mock, aread_calls


def load_fixture(name):
    with open(_FIXTURES_DIR / name) as f:
        return json.load(f)


def fixture_exists(name):
    return (_FIXTURES_DIR / name).exists()


def _fixture_to_response(data, request):
    if isinstance(data, list):
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in data) + "data: [DONE]\n\n"
        return httpx.Response(
            200,
            content=body.encode("utf-8"),
            headers={"content-type": "text/event-stream"},
            request=request,
        )
    if isinstance(data, dict) and "status_code" in data and "body" in data:
        body = data.get("body")
        content = json.dumps(body).encode("utf-8") if body is not None else b""
        return httpx.Response(
            data["status_code"],
            content=content,
            headers=data.get("response_headers") or {},
            request=request,
        )
    if isinstance(data, dict):
        return httpx.Response(200, json=data, request=request)
    raise ValueError(f"Unknown fixture shape: {type(data).__name__}")


def _resolve_fixture(fixture_or_name):
    return load_fixture(fixture_or_name) if isinstance(fixture_or_name, str) else fixture_or_name


def fixture_transport(fixture_or_name, on_request=None):
    """Build an httpx.MockTransport that serves a fixture on each request."""
    data = _resolve_fixture(fixture_or_name)

    def handler(request):
        if on_request is not None:
            on_request(request)
        return _fixture_to_response(data, request)

    return httpx.MockTransport(handler)


def sequenced_fixture_transport(fixtures_or_names, on_request=None):
    """Serve a sequence of fixtures, one per request. For multi-turn tests."""
    data_list = [_resolve_fixture(f) for f in fixtures_or_names]
    idx = [0]

    def handler(request):
        if on_request is not None:
            on_request(request)
        if idx[0] >= len(data_list):
            raise AssertionError(
                f"sequenced_fixture_transport: request {idx[0] + 1} exceeds fixture count ({len(data_list)})"
            )
        response = _fixture_to_response(data_list[idx[0]], request)
        idx[0] += 1
        return response

    return httpx.MockTransport(handler)


@asynccontextmanager
async def simulated_model(
    fixture_or_name,
    *,
    model_name="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key="sk-test-simulated",
    on_request=None,
    transport=None,
    **client_kwargs,
):
    """Context manager yielding an OpenAIChatModel whose HTTP transport is
    backed by a fixture. Exercises the full client stack (serialization,
    httpx, SSE parsing, response parsing) without a real API call.
    """
    if transport is None:
        transport = fixture_transport(fixture_or_name, on_request=on_request)
    async with httpx.AsyncClient(transport=transport) as http:
        client = OpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            http_client=http,
            max_retries=0,
            **client_kwargs,
        )
        yield OpenAIChatModel(client=client, model=model_name)


@asynccontextmanager
async def simulated_model_sequenced(
    fixtures,
    *,
    model_name="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    on_request=None,
    **client_kwargs,
):
    """Like simulated_model but serves a sequence of fixtures (for multi-turn)."""
    transport = sequenced_fixture_transport(fixtures, on_request=on_request)
    async with simulated_model(
        None,
        model_name=model_name,
        base_url=base_url,
        on_request=on_request,
        transport=transport,
        **client_kwargs,
    ) as model:
        yield model


def live_mode_enabled(provider: str) -> bool:
    if not LIVE_TEST_MODE:
        return False
    env_var = {"openai": "OPENAI_API_KEY", "nim": "NVIDIA_API_KEY"}[provider]
    return bool(os.environ.get(env_var))


@asynccontextmanager
async def live_openai_model(*, model_name=OPENAI_DEFAULT_MODEL):
    """Yield an OpenAIChatModel backed by the real OpenAI API."""
    async with httpx.AsyncClient() as http:
        client = OpenAICompatibleClient(
            base_url=OPENAI_BASE_URL,
            api_key=os.environ.get("OPENAI_API_KEY"),
            http_client=http,
        )
        yield OpenAIChatModel(client=client, model=model_name)


@asynccontextmanager
async def live_nim_model(*, model_name=NIM_DEFAULT_MODEL):
    """Yield a NIM OpenAIChatModel backed by the real NIM API."""
    async with httpx.AsyncClient() as http:
        client = OpenAICompatibleClient(
            base_url=NIM_BASE_URL,
            api_key=os.environ.get("NVIDIA_API_KEY"),
            http_client=http,
        )
        yield OpenAIChatModel(client=client, model=model_name)

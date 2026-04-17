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

import httpx

from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient


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

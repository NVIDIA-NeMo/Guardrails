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

"""In-memory HTTP transport for canonical model integration tests."""

from unittest.mock import AsyncMock, Mock

import httpx


def mock_model_http_client(payload=None, *, raw_lines=None, error=None, headers=None):
    """Keep real client/model parsing while replacing only the HTTP transport."""

    def handle(request):
        if error is not None:
            raise error
        if raw_lines is not None:
            return httpx.Response(
                200,
                stream=httpx.ByteStream(b"".join(raw_lines)),
                headers=headers or {"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=payload, headers=headers)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    client.post = AsyncMock(wraps=client.post)
    client.stream = Mock(wraps=client.stream)
    return client


def install_model_http_mock(runtime, model_type, payload=None, *, raw_lines=None, error=None):
    """Route mock replies by model when configured models share a client pool."""
    import json

    model = runtime.llms[model_type].wrapped_model
    adapter = model._client
    client = adapter._http_client
    if client is None or not hasattr(client, "_test_model_replies"):
        replies = {}
        calls = {}

        def handle(request):
            name = json.loads(request.content)["model"]
            calls[name](request)
            reply, lines, failure = replies[name]
            if failure is not None:
                raise failure
            if lines is not None:
                return httpx.Response(
                    200, stream=httpx.ByteStream(b"".join(lines)), headers={"content-type": "text/event-stream"}
                )
            return httpx.Response(200, json=reply)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        client._test_model_replies = replies
        client._test_model_calls = calls
        client.post = AsyncMock(wraps=client.post)
        client.stream = Mock(wraps=client.stream)
        adapter._client = client
    client._test_model_replies[model.model_name] = (payload, raw_lines, error)
    client._test_model_calls[model.model_name] = Mock()
    return client

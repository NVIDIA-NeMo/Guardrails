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

from unittest import mock

import httpx
import pytest

from nemoguardrails.http import (
    ClosableHTTPClient,
    HTTPClient,
    RetryPolicy,
    create_http_client,
)


def test_default_factory_forwards_owned_client_options():
    limits = httpx.Limits(max_connections=12, max_keepalive_connections=4)

    with mock.patch("nemoguardrails.http.transport.httpx.AsyncClient") as factory:
        create_http_client(limits=limits, follow_redirects=True)

    kwargs = factory.call_args.kwargs
    assert kwargs["timeout"] is None
    assert kwargs["limits"] is limits
    assert kwargs["follow_redirects"] is True


@pytest.mark.asyncio
async def test_default_factory_composes_a_closable_client():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True}, request=request)

    injected = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = create_http_client(
        httpx_client=injected,
        retry_policy=RetryPolicy(max_attempts=1),
    )

    response = await client.request("GET", "https://example.com/check")

    assert isinstance(client, HTTPClient)
    assert isinstance(client, ClosableHTTPClient)
    assert response.json() == {"ok": True}
    assert response.extensions["retry_count"] == 0
    await client.close()
    assert not injected.is_closed
    await injected.aclose()


@pytest.mark.asyncio
async def test_default_factory_does_not_retry():
    request_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(503, request=request)

    injected = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = create_http_client(httpx_client=injected)

    response = await client.request("POST", "https://example.com/check")

    assert response.status_code == 503
    assert request_count == 1
    assert "retry_count" not in response.extensions
    await client.close()
    await injected.aclose()

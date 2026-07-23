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

import httpx
import pytest

from nemoguardrails.http import (
    HTTPClient,
    HTTPClientManager,
    HTTPResponse,
    ManagedHTTPClient,
    RetryPolicy,
    create_http_client,
    http_call,
)
from nemoguardrails.http.testing import RecordingHTTPClient


@pytest.mark.asyncio
async def test_default_factory_composes_a_managed_client():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True}, request=request)

    injected = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = create_http_client(
        httpx_client=injected,
        retry_policy=RetryPolicy(max_attempts=1),
    )

    response = await client.request("GET", "https://example.com/check")

    assert isinstance(client, HTTPClient)
    assert isinstance(client, ManagedHTTPClient)
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
    await injected.aclose()


@pytest.mark.asyncio
async def test_manager_owns_factory_client_and_closes_it_once():
    owned = RecordingHTTPClient([HTTPResponse(status_code=200)])
    manager = HTTPClientManager(factory=lambda: owned)

    with pytest.raises(RuntimeError, match="has not been started"):
        _ = manager.client

    first = await manager.start()
    second = await manager.start()
    await first.request("GET", "https://example.com/check")
    await manager.stop()
    await manager.stop()

    assert first is owned
    assert second is owned
    assert owned.close_calls == 1


@pytest.mark.asyncio
async def test_manager_never_closes_injected_client():
    injected = RecordingHTTPClient()
    manager = HTTPClientManager(injected)

    resolved = await manager.start()
    await manager.stop()

    assert resolved is injected
    assert injected.close_calls == 0


@pytest.mark.asyncio
async def test_manager_creates_a_fresh_owned_client_after_restart():
    created: list[RecordingHTTPClient] = []

    def factory() -> RecordingHTTPClient:
        client = RecordingHTTPClient()
        created.append(client)
        return client

    manager = HTTPClientManager(factory=factory)

    first = await manager.start()
    await manager.stop()
    second = await manager.start()
    await manager.stop()

    assert first is created[0]
    assert second is created[1]
    assert created[0].close_calls == 1
    assert created[1].close_calls == 1


@pytest.mark.asyncio
async def test_manager_scopes_requests_until_activated():
    created: list[RecordingHTTPClient] = []

    def factory() -> RecordingHTTPClient:
        client = RecordingHTTPClient([HTTPResponse(status_code=200)])
        created.append(client)
        return client

    manager = HTTPClientManager(factory=factory)

    await manager.request("POST", "https://example.com/check")
    await manager.request("POST", "https://example.com/check")

    assert len(created) == 2
    assert [client.close_calls for client in created] == [1, 1]


@pytest.mark.asyncio
async def test_activated_manager_lazily_reuses_client():
    owned = RecordingHTTPClient(
        [
            HTTPResponse(status_code=200),
            HTTPResponse(status_code=200),
        ]
    )
    manager = HTTPClientManager(factory=lambda: owned)

    manager.activate()
    await manager.request("POST", "https://example.com/check")
    await manager.request("POST", "https://example.com/check")

    assert len(owned.requests) == 2
    assert owned.close_calls == 0
    await manager.stop()
    assert owned.close_calls == 1


@pytest.mark.asyncio
async def test_http_call_closes_only_the_client_it_creates():
    owned = RecordingHTTPClient([HTTPResponse(status_code=200)])

    await http_call(None, "GET", "https://example.com/check", factory=lambda: owned)

    assert owned.close_calls == 1

    injected = RecordingHTTPClient([HTTPResponse(status_code=200)])
    await http_call(injected, "GET", "https://example.com/check")

    assert injected.close_calls == 0


@pytest.mark.asyncio
async def test_http_call_closes_owned_client_when_request_raises():
    owned = RecordingHTTPClient()

    with pytest.raises(RuntimeError, match="No HTTP responses available"):
        await http_call(None, "GET", "https://example.com/check", factory=lambda: owned)

    assert owned.close_calls == 1


@pytest.mark.asyncio
async def test_manager_rejects_unmanaged_factory_result():
    class UnmanagedClient:
        async def request(self, method, url, **kwargs):
            return HTTPResponse(status_code=200)

    manager = HTTPClientManager(factory=lambda: UnmanagedClient())

    with pytest.raises(TypeError, match="managed HTTP client"):
        await manager.start()

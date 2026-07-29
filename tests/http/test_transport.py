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

import asyncio
from unittest import mock

import httpx
import pytest

from nemoguardrails.http import (
    HTTPConnectionError,
    HTTPTimeoutError,
    HttpxHTTPClient,
)


@pytest.mark.asyncio
async def test_httpx_transport_forwards_request_and_returns_neutral_response():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            201,
            headers={"Content-Type": "application/json"},
            json={"created": True},
            request=request,
        )

    injected = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = HttpxHTTPClient(injected)

    response = await client.request(
        "POST",
        "https://example.com/items",
        headers={"Authorization": "Bearer secret"},
        params={"version": "1"},
        json={"name": "item"},
        timeout=4.0,
    )

    assert response.status_code == 201
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"created": True}
    assert response.extensions["http_version"] == "HTTP/1.1"
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "https://example.com/items?version=1"
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert requests[0].read() == b'{"name":"item"}'
    await client.close()
    assert not injected.is_closed
    await injected.aclose()


@pytest.mark.asyncio
async def test_httpx_transport_forwards_content():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.read() == b"payload"
        return httpx.Response(204, request=request)

    injected = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = HttpxHTTPClient(injected)

    response = await client.request("PUT", "https://example.com/items/1", content="payload")

    assert response.status_code == 204
    await injected.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport_error", "expected_error"),
    [
        (httpx.ReadTimeout("slow"), HTTPTimeoutError),
        (httpx.ConnectError("unavailable"), HTTPConnectionError),
        (httpx.DecodingError("invalid response encoding"), HTTPConnectionError),
        (httpx.TooManyRedirects("redirect limit exceeded"), HTTPConnectionError),
    ],
)
async def test_httpx_transport_translates_request_errors(transport_error, expected_error):
    async def handler(request: httpx.Request) -> httpx.Response:
        transport_error.request = request
        raise transport_error

    injected = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = HttpxHTTPClient(injected)

    with pytest.raises(expected_error) as exc_info:
        await client.request("GET", "https://user:password@example.com/items?token=secret")

    assert "password" not in str(exc_info.value)
    assert "secret" not in str(exc_info.value)
    assert exc_info.value.__cause__ is transport_error
    assert str(transport_error.request.url) == "https://example.com/items"
    await injected.aclose()


@pytest.mark.asyncio
async def test_httpx_transport_closes_owned_client_once():
    client = HttpxHTTPClient()
    owned = client._client

    await client.close()
    await client.close()

    assert owned.is_closed


@pytest.mark.asyncio
async def test_httpx_transport_requests_with_owned_client():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"owned", request=request)

    owned = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with mock.patch("nemoguardrails.http.transport.httpx.AsyncClient", return_value=owned):
        async with HttpxHTTPClient(timeout=1.0) as client:
            response = await client.request("GET", "https://example.com/items")

    assert response.content == b"owned"
    assert owned.is_closed


@pytest.mark.asyncio
async def test_httpx_transport_enforces_total_request_deadline():
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, request=request)

    injected = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = HttpxHTTPClient(injected)

    with pytest.raises(HTTPTimeoutError):
        await client.request("GET", "https://example.com/slow", timeout=0.01)

    await injected.aclose()


@pytest.mark.asyncio
async def test_httpx_transport_rejects_invalid_per_request_timeout():
    injected = mock.AsyncMock(spec=httpx.AsyncClient)
    client = HttpxHTTPClient(injected)

    with pytest.raises(ValueError, match="greater than zero"):
        await client.request("GET", "https://example.com/items", timeout=0)

    injected.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_httpx_transport_translates_request_error_without_request_context():
    transport_error = httpx.ConnectError("unavailable")
    injected = mock.AsyncMock(spec=httpx.AsyncClient)
    injected.request.side_effect = transport_error
    client = HttpxHTTPClient(injected)

    with pytest.raises(HTTPConnectionError) as exc_info:
        await client.request("GET", "https://example.com/items")

    assert exc_info.value.__cause__ is transport_error


@pytest.mark.parametrize("timeout", [0, -1])
def test_httpx_transport_rejects_invalid_total_timeout(timeout):
    with pytest.raises(ValueError, match="greater than zero"):
        HttpxHTTPClient(timeout=timeout)


def test_httpx_transport_can_enable_redirects_explicitly():
    with mock.patch("nemoguardrails.http.transport.httpx.AsyncClient") as factory:
        HttpxHTTPClient(follow_redirects=True)

    assert factory.call_args.kwargs["follow_redirects"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"timeout": 10.0},
        {"limits": httpx.Limits(max_connections=10)},
        {"follow_redirects": True},
    ],
)
async def test_httpx_transport_rejects_owned_options_with_injected_client(options):
    injected = httpx.AsyncClient()

    with pytest.raises(ValueError, match="Owned-client options"):
        HttpxHTTPClient(injected, **options)

    await injected.aclose()

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

import anyio
import httpx
import pytest

from nemoguardrails.http import (
    HTTPConnectionError,
    HTTPTimeoutError,
    HTTPTLSConfig,
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


def test_httpx_transport_configures_owned_client_tls():
    constructed = mock.MagicMock()

    with mock.patch("nemoguardrails.http.transport.httpx.AsyncClient", return_value=constructed) as factory:
        client = HttpxHTTPClient(
            timeout=12.0,
            tls=HTTPTLSConfig(
                ca_bundle="/ca.pem",
                client_certificate="/cert.pem",
                client_key="/key.pem",
            ),
        )

    assert client._client is constructed
    assert client._timeout == 12.0
    assert factory.call_args.kwargs["timeout"] is None
    assert factory.call_args.kwargs["verify"] == "/ca.pem"
    assert factory.call_args.kwargs["cert"] == ("/cert.pem", "/key.pem")


def test_httpx_transport_rejects_tls_with_injected_client():
    injected = mock.MagicMock(spec=httpx.AsyncClient)

    with pytest.raises(ValueError, match="TLS configuration"):
        HttpxHTTPClient(injected, tls=HTTPTLSConfig(verify=False))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"client_certificate": "/cert.pem"},
        {"client_key": "/key.pem"},
    ],
)
def test_http_tls_config_requires_complete_client_credentials(kwargs):
    with pytest.raises(ValueError, match="configured together"):
        HTTPTLSConfig(**kwargs)


class ClosingTransport(httpx.MockTransport):
    """Expose completion separately from HTTPX's early closed state."""

    def __init__(self):
        super().__init__(lambda request: httpx.Response(200))
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.error: Exception | None = None
        self.close_calls = 0
        self.completed = False

    async def aclose(self):
        self.close_calls += 1
        self.entered.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error
        self.completed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["task", "scope", "exception"])
async def test_owned_httpx_close_retains_completion_or_failure(failure):
    """Finish transport cleanup even after HTTPX reports its client closed."""

    transport = ClosingTransport()
    owned = httpx.AsyncClient(transport=transport, trust_env=False)
    with mock.patch("nemoguardrails.http.transport.httpx.AsyncClient", return_value=owned):
        client = HttpxHTTPClient()
    try:
        if failure == "exception":
            transport.error = RuntimeError("transport cleanup failed")
            with pytest.raises(RuntimeError, match="transport cleanup failed"):
                await client.close()
        else:
            transport.release.clear()
            if failure == "task":
                closing = asyncio.create_task(client.close())
                with anyio.fail_after(1):
                    await transport.entered.wait()
                closing.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await closing
            else:
                for _ in range(2):
                    with anyio.move_on_after(0.01) as scope:
                        await client.close()
                    assert scope.cancelled_caught
            assert not transport.completed
        assert owned.is_closed
        assert not client._closed
        transport.release.set()
        with anyio.fail_after(1):
            if failure == "exception":
                with pytest.raises(RuntimeError) as caught:
                    await client.close()
                assert caught.value is transport.error
                assert not client._closed
                assert not transport.completed
            else:
                await client.close()
                assert client._closed
                assert transport.completed
        assert transport.close_calls == 1
    finally:
        transport.error = None
        transport.release.set()
        if failure != "exception":
            await client.close()
        await owned.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_owned_httpx_concurrent_closes_wait_for_one_transport_close():
    """Share unfinished cleanup without reporting success to another caller."""

    transport = ClosingTransport()
    transport.release.clear()
    owned = httpx.AsyncClient(transport=transport, trust_env=False)
    with mock.patch("nemoguardrails.http.transport.httpx.AsyncClient", return_value=owned):
        client = HttpxHTTPClient()
    first = asyncio.create_task(client.close())
    second = None
    try:
        with anyio.fail_after(1):
            await transport.entered.wait()
        second = asyncio.create_task(client.close())
        await asyncio.sleep(0)
        assert not second.done()
        assert not client._closed
        transport.release.set()
        with anyio.fail_after(1):
            await asyncio.gather(first, second)
        assert transport.completed
        assert transport.close_calls == 1
        await client.close()
        assert transport.close_calls == 1
    finally:
        transport.release.set()
        await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)
        await client.close()
        await owned.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_failed_httpx_pool_close_cannot_be_mistaken_for_success(monkeypatch):
    """Retain failure after the real HTTPX pool discards a failing connection."""

    owned = httpx.AsyncClient(trust_env=False)
    transport = owned._transport
    assert isinstance(transport, httpx.AsyncHTTPTransport)
    pool = transport._pool
    connection = mock.Mock()
    connection.aclose = mock.AsyncMock(side_effect=RuntimeError("connection cleanup failed"))
    monkeypatch.setattr(pool, "_connections", [connection])
    with mock.patch("nemoguardrails.http.transport.httpx.AsyncClient", return_value=owned):
        client = HttpxHTTPClient()
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="connection cleanup failed"):
                await client.close()
            assert owned.is_closed
            assert pool.connections == []
            assert not client._closed
        connection.aclose.assert_awaited_once()
    finally:
        await owned.__aexit__(None, None, None)

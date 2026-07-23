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

import pytest

from nemoguardrails.http import HTTPConnectionError, HTTPResponse, HTTPStatusError, http_call
from nemoguardrails.http.testing import RecordingHTTPClient
from nemoguardrails.http.types import HTTPRequest


@pytest.mark.asyncio
async def test_http_call_forwards_the_request_and_returns_response():
    response = HTTPResponse(status_code=200, content=b"ok")
    client = RecordingHTTPClient([response])
    headers = {"Authorization": "Bearer secret"}
    params = {"version": 1}
    payload = {"text": "hello"}

    result = await http_call(
        client,
        "POST",
        "https://example.com/check",
        headers=headers,
        params=params,
        json=payload,
        timeout=3.0,
    )

    assert result is response
    assert client.requests == [
        HTTPRequest(
            method="POST",
            url="https://example.com/check",
            headers=headers,
            params=params,
            json=payload,
            timeout=3.0,
        )
    ]


@pytest.mark.asyncio
async def test_http_call_raises_status_error_with_retry_context():
    response = HTTPResponse(status_code=503, extensions={"retry_count": 2})
    client = RecordingHTTPClient([response])

    with pytest.raises(HTTPStatusError) as exc_info:
        await http_call(
            client,
            "GET",
            "https://user:password@example.com/check?token=secret",
        )

    assert exc_info.value.response is response
    assert exc_info.value.retry_count == 2
    assert "https://example.com/check" in str(exc_info.value)
    assert "password" not in str(exc_info.value)
    assert "secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_http_call_ignores_invalid_retry_metadata():
    response = HTTPResponse(status_code=503, extensions={"retry_count": "unknown"})
    client = RecordingHTTPClient([response])

    with pytest.raises(HTTPStatusError) as exc_info:
        await http_call(client, "GET", "https://example.com/check")

    assert exc_info.value.retry_count == 0


@pytest.mark.asyncio
async def test_http_call_can_return_error_response_without_raising():
    response = HTTPResponse(status_code=404)
    client = RecordingHTTPClient([response])

    result = await http_call(
        client,
        "GET",
        "https://example.com/missing",
        raise_for_status=False,
    )

    assert result is response


@pytest.mark.asyncio
async def test_http_call_preserves_client_error():
    error = HTTPConnectionError("unavailable")
    client = RecordingHTTPClient([error])

    with pytest.raises(HTTPConnectionError) as exc_info:
        await http_call(client, "GET", "https://example.com/check")

    assert exc_info.value is error


@pytest.mark.asyncio
async def test_http_call_closes_only_the_client_it_creates():
    response = HTTPResponse(status_code=200, content=b'{"ok": true}')
    owned = RecordingHTTPClient([response])

    result = await http_call(None, "GET", "https://example.com/check", factory=lambda: owned)

    assert owned.close_calls == 1
    assert result.json() == {"ok": True}

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
async def test_http_call_rejects_unmanaged_factory_result():
    class UnmanagedClient:
        async def request(self, method, url, **kwargs):
            return HTTPResponse(status_code=200)

    with pytest.raises(TypeError, match="managed HTTP client"):
        await http_call(None, "GET", "https://example.com/check", factory=lambda: UnmanagedClient())

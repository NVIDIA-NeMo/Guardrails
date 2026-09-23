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

from nemoguardrails.http import (
    ClosableHTTPClient,
    HTTPClient,
    HTTPClientError,
    HTTPConnectionError,
    HTTPRequest,
    HTTPResponse,
    HTTPResponseDecodeError,
    HTTPStatusError,
    HTTPTimeoutError,
)
from nemoguardrails.testing import RecordingHTTPClient


def test_http_response_exposes_bytes_text_and_json():
    response = HTTPResponse(
        status_code=200,
        headers={"Content-Type": "application/json; charset=utf-8"},
        content='{"message": "caf\u00e9"}'.encode(),
    )

    assert response.is_success
    assert response.text == '{"message": "caf\u00e9"}'
    assert response.json() == {"message": "caf\u00e9"}


def test_http_response_uses_declared_charset():
    response = HTTPResponse(
        status_code=200,
        headers={"content-type": "text/plain; charset=iso-8859-1"},
        content="caf\u00e9".encode("iso-8859-1"),
    )

    assert response.text == "caf\u00e9"


def test_http_response_falls_back_from_unknown_charset():
    response = HTTPResponse(
        status_code=200,
        headers={"content-type": "text/plain; charset=unknown"},
        content="caf\u00e9".encode(),
    )

    assert response.text == "caf\u00e9"


def test_http_response_decode_error_does_not_expose_body():
    response = HTTPResponse(status_code=200, content=b"secret-invalid-json")

    with pytest.raises(HTTPResponseDecodeError) as exc_info:
        response.json()

    assert exc_info.value.response is response
    assert "secret-invalid-json" not in str(exc_info.value)


def test_http_status_error_retains_context_and_redacts_url_credentials():
    request = HTTPRequest(
        method="post",
        url="https://user:password@example.com/check?api_key=secret#fragment",
    )
    response = HTTPResponse(status_code=429)

    with pytest.raises(HTTPStatusError) as exc_info:
        response.raise_for_status(request)

    assert exc_info.value.request is request
    assert exc_info.value.response is response
    assert "POST https://example.com/check" in str(exc_info.value)
    assert "password" not in str(exc_info.value)
    assert "secret" not in str(exc_info.value)


def test_http_status_error_redacts_url_credentials_without_hostname():
    request = HTTPRequest(
        method="get",
        url="https://user:password@/check?api_key=secret#fragment",
    )
    response = HTTPResponse(status_code=500)

    with pytest.raises(HTTPStatusError) as exc_info:
        response.raise_for_status(request)

    assert "GET https:///check" in str(exc_info.value)
    assert "user" not in str(exc_info.value)
    assert "password" not in str(exc_info.value)
    assert "secret" not in str(exc_info.value)


def test_http_response_accepts_success_and_redirect_statuses():
    HTTPResponse(status_code=204).raise_for_status()
    HTTPResponse(status_code=302).raise_for_status()


@pytest.mark.asyncio
async def test_recording_client_records_request_and_returns_scripted_response():
    response = HTTPResponse(status_code=201, content=b"created")
    client = RecordingHTTPClient([response])
    headers = {"Authorization": "Bearer secret"}
    params = {"version": 1}
    payload = {"text": "hello"}

    result = await client.request(
        "POST",
        "https://example.com/check",
        headers=headers,
        params=params,
        json=payload,
        content=b"body",
        timeout=4.0,
    )

    assert result is response
    assert client.requests == [
        HTTPRequest(
            method="POST",
            url="https://example.com/check",
            headers=headers,
            params=params,
            json=payload,
            content=b"body",
            timeout=4.0,
        )
    ]
    assert isinstance(client, HTTPClient)
    assert isinstance(client, ClosableHTTPClient)


@pytest.mark.asyncio
async def test_recording_client_raises_scripted_errors_and_tracks_close():
    error = HTTPTimeoutError("timed out")
    client = RecordingHTTPClient([error])

    with pytest.raises(HTTPTimeoutError) as exc_info:
        await client.request("GET", "https://example.com")

    assert exc_info.value is error
    await client.close()
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_recording_client_requires_a_scripted_response():
    client = RecordingHTTPClient()

    with pytest.raises(RuntimeError, match="No HTTP responses available"):
        await client.request("GET", "https://example.com")


@pytest.mark.asyncio
async def test_recording_client_accepts_responses_after_construction():
    response = HTTPResponse(status_code=200)
    client = RecordingHTTPClient()
    client.add_response(response)

    assert await client.request("GET", "https://example.com") is response


def test_neutral_transport_errors_have_a_shared_base():
    assert issubclass(HTTPConnectionError, HTTPClientError)
    assert issubclass(HTTPTimeoutError, HTTPClientError)
    assert issubclass(HTTPStatusError, HTTPClientError)
    assert issubclass(HTTPResponseDecodeError, HTTPClientError)

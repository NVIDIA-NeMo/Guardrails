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

"""HTTPX transport adapter for the transport-neutral HTTP client contract."""

import asyncio
from typing import Any, Mapping

import httpx

from nemoguardrails.http._url import sanitize_url
from nemoguardrails.http.errors import HTTPConnectionError, HTTPTimeoutError
from nemoguardrails.http.types import HTTPResponse


def _sanitize_request_error(error: httpx.RequestError) -> httpx.RequestError:
    try:
        request = error.request
    except RuntimeError:
        return error
    error.request = httpx.Request(request.method, sanitize_url(str(request.url)))
    return error


class HttpxHTTPClient:
    """Adapt an HTTPX asynchronous client to the neutral HTTP contract.

    A client created by this adapter is owned and closed by the adapter. An
    injected client remains owned by its caller. The configured timeout covers
    the complete request rather than each HTTPX timeout phase independently.
    """

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        timeout: float | None = 30.0,
        limits: httpx.Limits | None = None,
        follow_redirects: bool = False,
    ):
        """Initialize the HTTPX adapter.

        Args:
            client: Optional caller-owned HTTPX client.
            timeout: Default total timeout for owned clients, in seconds.
            limits: Connection-pool limits for an owned client.
            follow_redirects: Whether an owned client follows redirects.

        Raises:
            ValueError: If the configured timeout is not positive.
        """

        if timeout is not None and timeout <= 0:
            raise ValueError("HTTP timeout must be greater than zero")
        self._owns_client = client is None
        self._timeout = timeout if self._owns_client else None
        self._client = client or httpx.AsyncClient(
            timeout=None,
            limits=limits or httpx.Limits(max_connections=100, max_keepalive_connections=20),
            follow_redirects=follow_redirects,
        )
        self._closed = False

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        content: bytes | str | None = None,
        timeout: float | None = None,
    ) -> HTTPResponse:
        """Send one request and materialize the HTTPX response.

        Args:
            method: HTTP method.
            url: Absolute request URL.
            headers: Optional request headers.
            params: Optional query parameters.
            json: Optional JSON-serializable request body.
            content: Optional raw request body.
            timeout: Optional per-request total timeout in seconds.

        Returns:
            A neutral response containing copied headers and body bytes.

        Raises:
            HTTPTimeoutError: If the request exceeds its total timeout.
            HTTPConnectionError: If HTTPX reports a request failure.
            ValueError: If the per-request timeout is not positive.
        """

        kwargs: dict[str, Any] = {
            "headers": headers,
            "params": params,
            "json": json,
            "content": content,
        }
        deadline = timeout if timeout is not None else self._timeout
        if deadline is not None and deadline <= 0:
            raise ValueError("HTTP timeout must be greater than zero")
        try:
            response = await asyncio.wait_for(
                self._client.request(method, url, **kwargs),
                timeout=deadline,
            )
        except asyncio.TimeoutError as error:
            raise HTTPTimeoutError("HTTP request timed out") from error
        except httpx.TimeoutException as error:
            raise HTTPTimeoutError("HTTP request timed out") from _sanitize_request_error(error)
        except httpx.RequestError as error:
            raise HTTPConnectionError("HTTP transport failed") from _sanitize_request_error(error)
        return HTTPResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            content=response.content,
            extensions={"http_version": response.http_version},
        )

    async def close(self) -> None:
        """Close the owned HTTPX client at most once."""

        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "HttpxHTTPClient":
        """Return this client from an asynchronous context manager."""

        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close owned resources when leaving an asynchronous context."""

        await self.close()

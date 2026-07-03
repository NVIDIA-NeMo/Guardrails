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
from typing import Any, Mapping

import httpx

from nemoguardrails.http.errors import HTTPConnectionError, HTTPTimeoutError
from nemoguardrails.http.types import HTTPResponse, HTTPTLSConfig


class HttpxHTTPClient:
    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        timeout: float | None = 30.0,
        limits: httpx.Limits | None = None,
        tls: HTTPTLSConfig | None = None,
        follow_redirects: bool = False,
    ):
        if client is not None and tls is not None:
            raise ValueError("TLS configuration cannot be combined with an injected httpx client")
        if timeout is not None and timeout <= 0:
            raise ValueError("HTTP timeout must be greater than zero")
        tls_config = tls or HTTPTLSConfig()
        verify: bool | str = tls_config.verify
        if tls_config.verify and tls_config.ca_bundle is not None:
            verify = tls_config.ca_bundle
        cert = None
        if tls_config.client_certificate is not None and tls_config.client_key is not None:
            cert = (tls_config.client_certificate, tls_config.client_key)
        self._owns_client = client is None
        self._timeout = timeout if self._owns_client else None
        self._client = client or httpx.AsyncClient(
            timeout=None,
            limits=limits or httpx.Limits(max_connections=100, max_keepalive_connections=20),
            verify=verify,
            cert=cert,
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
            raise HTTPTimeoutError("HTTP request timed out") from error
        except httpx.TransportError as error:
            raise HTTPConnectionError("HTTP transport failed") from error
        return HTTPResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            content=response.content,
            extensions={"http_version": response.http_version},
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "HttpxHTTPClient":
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        await self.close()

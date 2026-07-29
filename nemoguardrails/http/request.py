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

"""Managed request lifecycle helpers for outbound HTTP calls."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Mapping

from nemoguardrails.http.client import ClosableHTTPClient, HTTPClient
from nemoguardrails.http.runtime import create_http_client
from nemoguardrails.http.types import HTTPRequest, HTTPResponse


@asynccontextmanager
async def _resolve_http_client(
    client: HTTPClient | None,
    *,
    factory: Callable[[], ClosableHTTPClient],
) -> AsyncIterator[HTTPClient]:
    """Yield an injected client or create and close an owned client."""

    if client is not None:
        yield client
        return

    owned = factory()
    if not isinstance(owned, ClosableHTTPClient):
        raise TypeError("HTTP client factory must return a closable HTTP client")
    try:
        yield owned
    finally:
        await owned.close()


async def http_call(
    client: HTTPClient | None,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, Any] | None = None,
    json: Any = None,
    content: bytes | str | None = None,
    timeout: float | None = None,
    raise_for_status: bool = True,
    factory: Callable[[], ClosableHTTPClient] = create_http_client,
) -> HTTPResponse:
    """Execute one outbound HTTP request with deterministic client ownership.

    Args:
        client: Optional caller-owned client. When omitted, ``factory`` creates
            a client that is closed after the request.
        method: HTTP method.
        url: Absolute request URL.
        headers: Optional request headers.
        params: Optional query parameters.
        json: Optional JSON-serializable request body.
        content: Optional raw request body.
        timeout: Optional total request timeout in seconds.
        raise_for_status: Whether responses with status 400 or greater raise.
        factory: Factory for an owned client when ``client`` is omitted.

    Returns:
        The materialized transport-neutral response.

    Raises:
        HTTPStatusError: If ``raise_for_status`` is enabled and the response
            status is 400 or greater.
        TypeError: If ``factory`` returns a client that cannot be closed.
    """

    async with _resolve_http_client(client, factory=factory) as resolved:
        response = await resolved.request(
            method,
            url,
            headers=headers,
            params=params,
            json=json,
            content=content,
            timeout=timeout,
        )
    if raise_for_status and response.status_code >= 400:
        response.raise_for_status(
            HTTPRequest(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json,
                content=content,
                timeout=timeout,
            )
        )
    return response

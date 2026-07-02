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

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx

from nemoguardrails.http.client import HTTPClient, ManagedHTTPClient
from nemoguardrails.http.instrumentation import HTTPBodyCapturePolicy, InstrumentedHTTPClient
from nemoguardrails.http.retry import RetryingHTTPClient, RetryPolicy
from nemoguardrails.http.transport import HttpxHTTPClient
from nemoguardrails.http.types import HTTPTLSConfig

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer


def create_http_client(
    *,
    httpx_client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
    limits: httpx.Limits | None = None,
    retry_policy: RetryPolicy | None = None,
    tracer: Tracer | None = None,
    body_capture: HTTPBodyCapturePolicy | None = None,
    tls: HTTPTLSConfig | None = None,
) -> InstrumentedHTTPClient:
    transport = HttpxHTTPClient(httpx_client, timeout=timeout, limits=limits, tls=tls)
    retrying = RetryingHTTPClient(transport, retry_policy)
    return InstrumentedHTTPClient(retrying, tracer, body_capture=body_capture)


class HTTPClientManager:
    def __init__(
        self,
        client: HTTPClient | None = None,
        *,
        factory: Callable[[], HTTPClient] = create_http_client,
    ):
        self._client = client
        self._factory = factory
        self._owns_client = False
        self._running = False

    @property
    def client(self) -> HTTPClient:
        if not self._running or self._client is None:
            raise RuntimeError("HTTP client manager has not been started")
        return self._client

    async def start(self) -> HTTPClient:
        if self._running:
            return self.client
        if self._client is None:
            client = self._factory()
            if not isinstance(client, ManagedHTTPClient):
                raise TypeError("HTTP client factory must return a managed HTTP client")
            self._client = client
            self._owns_client = True
        self._running = True
        return self.client

    async def stop(self) -> None:
        if not self._running:
            return
        client = self._client
        owns_client = self._owns_client
        self._running = False
        if owns_client:
            self._client = None
            self._owns_client = False
            if isinstance(client, ManagedHTTPClient):
                await client.close()

    async def __aenter__(self) -> HTTPClient:
        return await self.start()

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        await self.stop()


@asynccontextmanager
async def resolve_http_client(
    client: HTTPClient | None = None,
    *,
    factory: Callable[[], HTTPClient] = create_http_client,
) -> AsyncIterator[HTTPClient]:
    manager = HTTPClientManager(client, factory=factory)
    async with manager as resolved:
        yield resolved

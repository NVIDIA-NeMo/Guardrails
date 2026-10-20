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

"""Assemble the private transparent provider proxy application."""

from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager

import httpx
from fastapi import APIRouter, FastAPI

from nemoguardrails.server.experimental._content_checker import ContentChecker
from nemoguardrails.server.experimental._http_kernel import (
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    DEFAULT_MAX_RESPONSE_BODY_BYTES,
)
from nemoguardrails.server.experimental._provider_routes import include_provider_routes
from nemoguardrails.server.experimental._runtime import ProxyRuntime, _close_http_resource, get_runtime
from nemoguardrails.server.experimental.provider.integration import ProviderIntegration

HttpTransportFactory = Callable[[], httpx.AsyncBaseTransport]


@asynccontextmanager
async def _lifespan(application: FastAPI):
    factory = application.state.proxy_http_transport_factory
    transport = factory() if factory is not None else None
    if factory is not None and not isinstance(transport, httpx.AsyncBaseTransport):
        raise TypeError("The HTTP transport factory must return an httpx.AsyncBaseTransport.")
    try:
        runtime = ProxyRuntime.create(http_transport=transport)
    except BaseException:
        if transport is not None:
            await _close_http_resource(transport)
        raise
    application.state.proxy_runtime = runtime
    try:
        yield
    finally:
        await runtime.close()


def create_proxy_application(
    *,
    providers: Sequence[ProviderIntegration],
    checker: ContentChecker,
    local_routers: Sequence[APIRouter] = (),
    path_prefix: str = "",
    http_transport_factory: HttpTransportFactory | None = None,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    max_response_body_bytes: int = DEFAULT_MAX_RESPONSE_BODY_BYTES,
    timeout: float = 30.0,
) -> FastAPI:
    """Create one private proxy application with lifespan-owned HTTP dispatch."""

    if http_transport_factory is not None and not callable(http_transport_factory):
        raise TypeError("The HTTP transport factory must be callable.")
    application = FastAPI(lifespan=_lifespan)
    application.state.proxy_http_transport_factory = http_transport_factory
    for router in local_routers:
        application.include_router(router)
    include_provider_routes(
        application,
        providers,
        checker=checker,
        http_client=lambda: get_runtime(application).http_client,
        path_prefix=path_prefix,
        max_request_body_bytes=max_request_body_bytes,
        max_response_body_bytes=max_response_body_bytes,
        timeout=timeout,
    )
    return application

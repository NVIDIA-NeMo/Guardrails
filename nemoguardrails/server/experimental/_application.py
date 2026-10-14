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
from fastapi import FastAPI

from nemoguardrails.server.experimental._http_kernel import (
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    DEFAULT_MAX_RESPONSE_BODY_BYTES,
)
from nemoguardrails.server.experimental._provider_routes import include_provider_routes
from nemoguardrails.server.experimental._runtime import ServerRuntime, _close_http_resource, get_runtime
from nemoguardrails.server.experimental.provider.integration import ProviderIntegration
from nemoguardrails.server.experimental.rails import ConfigEntry, ConfigResolver
from nemoguardrails.server.experimental.rails.config import validate_config_entries

HttpTransportFactory = Callable[[], httpx.AsyncBaseTransport]


@asynccontextmanager
async def _lifespan(application: FastAPI):
    factory = application.state.experimental_http_transport_factory
    transport = factory() if factory is not None else None
    if factory is not None and not isinstance(transport, httpx.AsyncBaseTransport):
        raise TypeError("The HTTP transport factory must return an httpx.AsyncBaseTransport.")
    try:
        runtime = ServerRuntime.create(
            application.state.experimental_rails_configs,
            application.state.experimental_config_resolver,
            http_transport=transport,
        )
    except BaseException:
        if transport is not None:
            await _close_http_resource(transport)
        raise
    try:
        await runtime.content_checkers.startup()
        application.state.experimental_proxy_runtime = runtime
        yield
    finally:
        try:
            await runtime.close()
        finally:
            if getattr(application.state, "experimental_proxy_runtime", None) is runtime:
                del application.state.experimental_proxy_runtime


def create_proxy_application(
    *,
    providers: Sequence[ProviderIntegration],
    rails_configs: Sequence[ConfigEntry] = (),
    config_resolver: ConfigResolver | None = None,
    path_prefix: str = "",
    http_transport_factory: HttpTransportFactory | None = None,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    max_response_body_bytes: int = DEFAULT_MAX_RESPONSE_BODY_BYTES,
    timeout: float = 30.0,
) -> FastAPI:
    """Create one private proxy application with lifespan-owned HTTP dispatch."""

    if http_transport_factory is not None and not callable(http_transport_factory):
        raise TypeError("The HTTP transport factory must be callable.")
    resolved_configs = validate_config_entries(rails_configs)
    if config_resolver is None and len(resolved_configs) > 1:
        raise ValueError("Multiple Rails configs require a resolver.")
    if config_resolver is not None and not resolved_configs:
        raise ValueError("A config resolver requires at least one config entry.")
    resolved_resolver = config_resolver
    if resolved_resolver is None and len(resolved_configs) == 1:
        config_id = resolved_configs[0].id

        def resolve_static_config(_context):
            return config_id

        resolved_resolver = resolve_static_config

    application = FastAPI(lifespan=_lifespan)
    application.state.experimental_http_transport_factory = http_transport_factory
    application.state.experimental_rails_configs = resolved_configs
    application.state.experimental_config_resolver = resolved_resolver

    async def resolve_checker(request, provider_type, provider_instance_id, operation_name):
        return await get_runtime(application).resolve_content_checker(
            request,
            provider_type,
            provider_instance_id,
            operation_name,
        )

    include_provider_routes(
        application,
        providers,
        resolve_checker=resolve_checker,
        http_client=lambda: get_runtime(application).http_client,
        path_prefix=path_prefix,
        max_request_body_bytes=max_request_body_bytes,
        max_response_body_bytes=max_response_body_bytes,
        timeout=timeout,
    )
    return application

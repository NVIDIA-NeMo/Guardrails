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

"""Construct FastAPI routes from declarative provider integrations."""

from collections.abc import Sequence
from dataclasses import replace

from fastapi import APIRouter, FastAPI

from nemoguardrails.server.experimental._content_checker import ContentChecker, validate_content_checker
from nemoguardrails.server.experimental._guarded_proxy import create_guarded_http_operation
from nemoguardrails.server.experimental._http_kernel import (
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    DEFAULT_MAX_RESPONSE_BODY_BYTES,
    GuardedOperationPath,
    create_http_proxy_router,
)
from nemoguardrails.server.experimental._provider_transport import (
    DEFAULT_OUTBOUND_TIMEOUT,
    HTTPClientResolver,
    ProviderTransport,
)
from nemoguardrails.server.experimental.provider.integration import ProviderIntegration, _route_path_shape
from nemoguardrails.server.experimental.routers.passthrough import create_passthrough_router


def _validate_providers(providers: Sequence[ProviderIntegration]) -> tuple[ProviderIntegration, ...]:
    resolved = tuple(providers)
    if not resolved:
        raise ValueError("At least one provider integration is required.")
    mount_paths = [provider.upstream.mount_path for provider in resolved]
    if len(mount_paths) != len(set(mount_paths)):
        raise ValueError("Provider mount paths must be unique.")
    instance_ids = [provider.instance_id for provider in resolved]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("Provider instance ids must be unique.")

    operation_ids = [
        f"{provider.instance_id}.{endpoint.operation_name}" for provider in resolved for endpoint in provider.endpoints
    ]
    if len(operation_ids) != len(set(operation_ids)):
        raise ValueError("Guarded endpoint operation identifiers must be globally unique.")
    routes = [
        (method, _route_path_shape(f"{provider.upstream.mount_path}{operation_path.route_path}"))
        for provider in resolved
        for endpoint in provider.endpoints
        for operation_path in endpoint.operation_paths
        for method in operation_path.methods | operation_path.passthrough_methods
    ]
    if len(routes) != len(set(routes)):
        raise ValueError("Guarded endpoint paths must be globally unique after provider mounting.")
    return resolved


def _registered_http_routes(router: APIRouter | FastAPI, path_prefix: str = "") -> dict[str, set[str]]:
    reserved: dict[str, set[str]] = {}
    for route in router.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path is not None and methods is not None:
            reserved.setdefault(f"{path_prefix}{path}", set()).update(methods)
    return reserved


def create_provider_router(
    provider: ProviderIntegration,
    *,
    checker: ContentChecker,
    transport: ProviderTransport,
    max_request_body_bytes: int,
    max_response_body_bytes: int,
) -> APIRouter:
    """Construct the guarded routes declared by one provider."""

    checker_policy = validate_content_checker(checker).policy

    async def dispatch(request):
        return await transport.dispatch(
            request,
            require_identity_response=checker_policy.inspect_output,
        )

    async def stream_dispatch(request):
        return await transport.stream_dispatch(
            request,
            require_identity_response=checker_policy.inspect_output,
        )

    operations = tuple(
        create_guarded_http_operation(
            endpoint,
            provider.errors,
            stream_dispatch=stream_dispatch if endpoint.stream is not None else None,
            max_stream_event_bytes=max_response_body_bytes,
            max_pending_stream_bytes=max_response_body_bytes,
        )
        for endpoint in provider.endpoints
    )
    operation_ids = {
        endpoint.operation_name: f"{provider.instance_id}.{endpoint.operation_name}" for endpoint in provider.endpoints
    }
    return create_http_proxy_router(
        operations=operations,
        checker=checker,
        dispatch=dispatch,
        render_outcome=provider.errors.renderer,
        operation_ids=operation_ids,
        include_passthrough=False,
        max_request_body_bytes=max_request_body_bytes,
        max_response_body_bytes=max_response_body_bytes,
    )


def include_provider_routes(
    application: FastAPI,
    providers: Sequence[ProviderIntegration],
    *,
    checker: ContentChecker,
    http_client: HTTPClientResolver,
    path_prefix: str = "",
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    max_response_body_bytes: int = DEFAULT_MAX_RESPONSE_BODY_BYTES,
    timeout: float = DEFAULT_OUTBOUND_TIMEOUT,
) -> None:
    """Include provider instances from most-specific mount to least-specific."""

    normalized_prefix = path_prefix.rstrip("/")
    if normalized_prefix and (
        not normalized_prefix.startswith("/") or not normalized_prefix.isascii() or "//" in normalized_prefix
    ):
        raise ValueError("The provider path prefix must be empty or one canonical absolute ASCII path.")
    resolved_providers = _validate_providers(providers)
    reserved_routes = _registered_http_routes(application)
    mounted: list[tuple[str, ProviderIntegration, ProviderTransport]] = []
    for provider in resolved_providers:
        provider_prefix = f"{normalized_prefix}{provider.upstream.mount_path}"
        transport = ProviderTransport(
            replace(provider.upstream, mount_path=provider_prefix),
            http_client,
            control_headers=provider.control_headers,
            max_response_body_bytes=max_response_body_bytes,
            timeout=timeout,
        )
        guarded_router = create_provider_router(
            provider,
            checker=checker,
            transport=transport,
            max_request_body_bytes=max_request_body_bytes,
            max_response_body_bytes=max_response_body_bytes,
        )
        application.include_router(guarded_router, prefix=provider_prefix, tags=[provider.display_name])
        for path, methods in _registered_http_routes(guarded_router, provider_prefix).items():
            reserved_routes.setdefault(path, set()).update(methods)
        mounted.append((provider_prefix, provider, transport))

    for provider_prefix, provider, transport in sorted(mounted, key=lambda item: len(item[0]), reverse=True):
        guarded_paths = tuple(
            GuardedOperationPath(
                f"{provider_prefix}{operation_path.route_path}",
                operation_path.methods,
                operation_path.passthrough_methods,
            )
            for endpoint in provider.endpoints
            for operation_path in endpoint.operation_paths
        )
        application.include_router(
            create_passthrough_router(
                transport=transport,
                errors=provider.errors,
                reserved_routes=reserved_routes,
                guarded_operation_paths=guarded_paths,
            ),
            prefix=provider_prefix,
        )

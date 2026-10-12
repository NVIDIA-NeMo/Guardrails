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

from collections.abc import Collection, Mapping, Sequence

import httpx
from fastapi import APIRouter, FastAPI
from starlette.routing import compile_path

from nemoguardrails.server.experimental._content_checker import ContentChecker
from nemoguardrails.server.experimental._guarded_proxy import create_guarded_http_operation
from nemoguardrails.server.experimental._http_kernel import (
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    DEFAULT_MAX_RESPONSE_BODY_BYTES,
    create_http_proxy_router,
)
from nemoguardrails.server.experimental._provider_transport import DEFAULT_OUTBOUND_TIMEOUT, ProviderTransport
from nemoguardrails.server.experimental.provider.integration import ProviderIntegration


def _route_shape(path: str) -> str:
    _, path_format, convertors = compile_path(path)
    for name, convertor in convertors.items():
        path_format = path_format.replace(f"{{{name}}}", f"{{{type(convertor).__name__}}}")
    return path_format


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

    provider_definitions: dict[str, tuple[object, ...]] = {}
    for provider in resolved:
        definition = (
            provider.display_name,
            provider.endpoints,
            provider.errors,
            provider.control_headers,
            provider.openapi_document,
        )
        existing = provider_definitions.setdefault(provider.provider_type, definition)
        if existing != definition:
            raise ValueError("Provider instances with the same type must share one type-level definition.")

    operation_ids = [
        f"{provider.instance_id}.{endpoint.operation_name}" for provider in resolved for endpoint in provider.endpoints
    ]
    if len(operation_ids) != len(set(operation_ids)):
        raise ValueError("Guarded endpoint operation identifiers must be globally unique.")
    routes = [
        (method, _route_shape(f"{provider.upstream.mount_path}{operation_path.route_path}"))
        for provider in resolved
        for endpoint in provider.endpoints
        for operation_path in endpoint.operation_paths
        for method in operation_path.methods | operation_path.passthrough_methods
    ]
    if len(routes) != len(set(routes)):
        raise ValueError("Guarded endpoint paths must be globally unique after provider mounting.")
    return resolved


def _registered_http_routes(router: APIRouter | FastAPI) -> dict[str, set[str]]:
    reserved: dict[str, set[str]] = {}
    for route in router.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path is not None and methods is not None:
            reserved.setdefault(path, set()).update(methods)
    return reserved


def create_provider_router(
    provider: ProviderIntegration,
    *,
    checker: ContentChecker,
    dispatcher: ProviderTransport,
    reserved_routes: Mapping[str, Collection[str]],
    guarded_path_prefix: str,
    max_request_body_bytes: int,
    max_response_body_bytes: int,
) -> APIRouter:
    """Construct the guarded and transparent routes declared by one provider."""

    operations = tuple(
        create_guarded_http_operation(
            endpoint,
            provider.errors,
            stream_dispatch=dispatcher.stream_dispatch,
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
        dispatch=dispatcher.dispatch,
        render_outcome=provider.errors.renderer,
        reserved_routes=reserved_routes,
        operation_ids=operation_ids,
        guarded_path_prefix=guarded_path_prefix,
        max_request_body_bytes=max_request_body_bytes,
        max_response_body_bytes=max_response_body_bytes,
    )


def include_provider_routes(
    application: FastAPI,
    providers: Sequence[ProviderIntegration],
    *,
    checker: ContentChecker,
    http_client: httpx.AsyncClient,
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
    mounted = []
    for provider in resolved_providers:
        provider_prefix = f"{normalized_prefix}{provider.upstream.mount_path}"
        dispatcher = ProviderTransport(
            http_client,
            base_url=provider.upstream.base_url,
            mount_path=provider_prefix,
            control_headers=provider.control_headers,
            max_response_body_bytes=max_response_body_bytes,
            timeout=timeout,
        )
        mounted.append((provider_prefix, provider, dispatcher))

    for provider_prefix, provider, dispatcher in sorted(mounted, key=lambda item: len(item[0]), reverse=True):
        router = create_provider_router(
            provider,
            checker=checker,
            dispatcher=dispatcher,
            reserved_routes=reserved_routes,
            guarded_path_prefix=provider_prefix,
            max_request_body_bytes=max_request_body_bytes,
            max_response_body_bytes=max_response_body_bytes,
        )
        application.include_router(router, prefix=provider_prefix, tags=[provider.display_name])

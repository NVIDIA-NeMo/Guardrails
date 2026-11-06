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
from dataclasses import dataclass, replace
from urllib.parse import quote

from fastapi import APIRouter, FastAPI, routing

from nemoguardrails.server.experimental._configuration_error import DeploymentValidationError
from nemoguardrails.server.experimental._content_checker import ContentCheckerResolver
from nemoguardrails.server.experimental._guarded_proxy import create_guarded_http_operation
from nemoguardrails.server.experimental._http_kernel import (
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    DEFAULT_MAX_RESPONSE_BODY_BYTES,
    create_http_proxy_router,
)
from nemoguardrails.server.experimental._http_paths import HttpRouteOwnership, _route_shape
from nemoguardrails.server.experimental._provider_transport import (
    DEFAULT_OUTBOUND_TIMEOUT,
    HTTPClientResolver,
    ProviderTransport,
)
from nemoguardrails.server.experimental.provider.integration import ProviderIntegration
from nemoguardrails.server.experimental.routers.passthrough import create_passthrough_router


@dataclass(frozen=True, slots=True)
class ProviderRoutePlan:
    provider: ProviderIntegration
    mount_path: str
    dispatcher: ProviderTransport
    ownership: HttpRouteOwnership

    @classmethod
    def create(
        cls, provider: ProviderIntegration, mount_path: str, dispatcher: ProviderTransport
    ) -> "ProviderRoutePlan":
        declared_paths = tuple(path for endpoint in provider.endpoints for path in endpoint.operation_paths)
        local_paths = tuple(replace(path, route_path=f"{mount_path}{path.route_path}") for path in declared_paths)

        def map_upstream_path(path: str, raw_path: bytes) -> str:
            return dispatcher.upstream_url(path, raw_path).path

        upstream_paths = declared_paths + tuple(
            replace(
                path, route_path=map_upstream_path(path.route_path, quote(path.route_path, safe="/").encode("ascii"))
            )
            for path in local_paths
        )
        return cls(provider, mount_path, dispatcher, HttpRouteOwnership(local_paths, upstream_paths, map_upstream_path))


def _validate_provider_plans(plans: Sequence[ProviderRoutePlan]) -> None:
    if not plans:
        raise DeploymentValidationError("At least one provider integration is required.")
    mount_paths = [plan.mount_path for plan in plans]
    if len(mount_paths) != len(set(mount_paths)):
        raise DeploymentValidationError("Provider mount paths must be unique.")
    instance_ids = [plan.provider.instance_id for plan in plans]
    if len(instance_ids) != len(set(instance_ids)):
        raise DeploymentValidationError("Provider instance ids must be unique.")

    operation_ids = [
        f"{plan.provider.instance_id}.{endpoint.operation_name}"
        for plan in plans
        for endpoint in plan.provider.endpoints
    ]
    if len(operation_ids) != len(set(operation_ids)):
        raise DeploymentValidationError("Guarded endpoint operation identifiers must be globally unique.")
    routes = [
        (method, _route_shape(operation_path.route_path))
        for plan in plans
        for operation_path in plan.ownership.local_paths
        for method in operation_path.methods | operation_path.passthrough_methods
    ]
    if len(routes) != len(set(routes)):
        raise DeploymentValidationError("Guarded endpoint paths must be globally unique after provider mounting.")
    for plan in plans:
        for other in plans:
            if not other.mount_path.startswith(plan.mount_path + "/"):
                continue
            if plan.ownership.intersects_mount(other.mount_path):
                raise DeploymentValidationError("A provider mount would shadow a guarded route of another instance.")


def _registered_http_routes(router: APIRouter | FastAPI, path_prefix: str = "") -> dict[str, set[str]]:
    reserved: dict[str, set[str]] = {}
    for route in getattr(routing, "iter_route_contexts", iter)(router.routes):
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path is not None and methods is not None:
            reserved.setdefault(f"{path_prefix}{path}", set()).update(methods)
    return reserved


def create_provider_router(
    plan: ProviderRoutePlan,
    *,
    resolve_checker: ContentCheckerResolver,
    max_request_body_bytes: int,
    max_response_body_bytes: int,
) -> APIRouter:
    """Construct the guarded routes declared by one provider."""

    provider = plan.provider

    async def policy_dispatch(request, inspect_output):
        return await plan.dispatcher.dispatch(request, require_identity_response=inspect_output)

    async def policy_stream_dispatch(request, inspect_output):
        return await plan.dispatcher.stream_dispatch(request, require_identity_response=inspect_output)

    operations = tuple(
        create_guarded_http_operation(
            endpoint,
            provider.errors,
            policy_stream_dispatch=policy_stream_dispatch if endpoint.stream is not None else None,
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
        resolve_checker=resolve_checker,
        provider_type=provider.provider_type,
        provider_instance_id=provider.instance_id,
        dispatch=plan.dispatcher.dispatch,
        policy_dispatch=policy_dispatch,
        render_outcome=provider.errors.renderer,
        operation_ids=operation_ids,
        guarded_path_prefix=plan.mount_path,
        route_ownership=plan.ownership,
        include_passthrough=False,
        max_request_body_bytes=max_request_body_bytes,
        max_response_body_bytes=max_response_body_bytes,
    )


def include_provider_routes(
    application: FastAPI,
    providers: Sequence[ProviderIntegration],
    *,
    resolve_checker: ContentCheckerResolver,
    http_client: HTTPClientResolver,
    path_prefix: str = "",
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    max_response_body_bytes: int = DEFAULT_MAX_RESPONSE_BODY_BYTES,
    timeout: float = DEFAULT_OUTBOUND_TIMEOUT,
) -> None:
    """Include provider instances from most-specific mount to least-specific.

    Reject mounts that would hide another instance's guarded route before
    modifying the application's routing table.
    """

    normalized_prefix = path_prefix.rstrip("/")
    if normalized_prefix and (
        not normalized_prefix.startswith("/") or not normalized_prefix.isascii() or "//" in normalized_prefix
    ):
        raise DeploymentValidationError("The provider path prefix must be empty or one canonical absolute ASCII path.")
    reserved_routes = _registered_http_routes(application)
    plans = []
    for provider in providers:
        provider_prefix = f"{normalized_prefix}{provider.upstream.mount_path}"
        dispatcher = ProviderTransport(
            replace(provider.upstream, mount_path=provider_prefix),
            http_client,
            control_headers=provider.control_headers,
            max_response_body_bytes=max_response_body_bytes,
            timeout=timeout,
        )
        plans.append(ProviderRoutePlan.create(provider, provider_prefix, dispatcher))

    _validate_provider_plans(plans)
    guarded_routers = [
        (
            plan,
            create_provider_router(
                plan,
                resolve_checker=resolve_checker,
                max_request_body_bytes=max_request_body_bytes,
                max_response_body_bytes=max_response_body_bytes,
            ),
        )
        for plan in sorted(plans, key=lambda plan: len(plan.mount_path), reverse=True)
    ]
    for plan, router in guarded_routers:
        for path, methods in _registered_http_routes(router, plan.mount_path).items():
            reserved_routes.setdefault(path, set()).update(methods)
    passthrough_routers = [
        (
            plan,
            create_passthrough_router(
                transport=plan.dispatcher,
                errors=plan.provider.errors,
                reserved_routes=reserved_routes,
                route_ownership=plan.ownership,
            ),
        )
        for plan in sorted(plans, key=lambda plan: len(plan.mount_path), reverse=True)
    ]
    for plan, router in guarded_routers:
        application.include_router(router, prefix=plan.mount_path, tags=[plan.provider.display_name])
    for plan, router in passthrough_routers:
        application.include_router(router, prefix=plan.mount_path)

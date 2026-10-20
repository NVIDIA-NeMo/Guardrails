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

"""Describe provider-specific behavior consumed by the server application."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from nemoguardrails.server.experimental._http_paths import _route_shape
from nemoguardrails.server.experimental.provider.endpoint import GuardedJsonEndpoint
from nemoguardrails.server.experimental.provider.errors import ProviderErrorMapping

ProviderOpenApiDocument: TypeAlias = Callable[[], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class UpstreamSettings:
    """Configure one provider upstream and its local mount."""

    base_url: str
    mount_path: str = ""


@dataclass(frozen=True, slots=True)
class ProviderIntegration:
    """Declare one mounted provider instance and its reviewed endpoint policy.

    ``provider_type`` identifies the protocol family, not a globally registered
    definition. Instances of the same family may select different endpoints or
    dialects, error mappings, and control headers. ``instance_id`` and mount
    paths are unique within the application; policy is never selected from
    untrusted request data or inferred from the upstream URL.
    """

    provider_type: str
    instance_id: str
    upstream: UpstreamSettings
    display_name: str
    endpoints: tuple[GuardedJsonEndpoint, ...]
    errors: ProviderErrorMapping
    control_headers: frozenset[str] = frozenset()
    openapi_document: ProviderOpenApiDocument | None = None

    def __post_init__(self) -> None:
        if not self.provider_type or not self.provider_type.replace("-", "_").isidentifier():
            raise ValueError("The provider type must be a non-empty identifier.")
        if not self.instance_id or not self.instance_id.replace("-", "_").isidentifier():
            raise ValueError("The provider instance id must be a non-empty identifier.")
        if not isinstance(self.upstream.base_url, str) or not self.upstream.base_url.strip():
            raise ValueError("The provider upstream base URL must not be empty.")
        if not self.display_name.strip():
            raise ValueError("The provider display name must not be empty.")
        if not self.endpoints:
            raise ValueError("A provider integration must declare at least one guarded endpoint.")
        if any(
            not isinstance(name, str) or not name or not name.isascii() or name != name.strip().lower()
            for name in self.control_headers
        ):
            raise ValueError("Provider control header names must be non-empty, lowercase ASCII strings.")

        mount_path = self.upstream.mount_path
        if mount_path and (
            not mount_path.startswith("/") or mount_path.endswith("/") or not mount_path.isascii() or "//" in mount_path
        ):
            raise ValueError("A provider mount path must be empty or one canonical absolute ASCII path.")
        if "{" in mount_path or "}" in mount_path:
            raise ValueError("A provider mount path must not contain dynamic parameters.")

        routes = [
            (method, _route_shape(operation_path.route_path))
            for endpoint in self.endpoints
            for operation_path in endpoint.operation_paths
            for method in operation_path.methods | operation_path.passthrough_methods
        ]
        if len(routes) != len(set(routes)):
            raise ValueError("Guarded endpoint paths must be unique within a provider.")
        operation_names = [endpoint.operation_name for endpoint in self.endpoints]
        if len(operation_names) != len(set(operation_names)):
            raise ValueError("Guarded endpoint operation names must be unique within a provider.")

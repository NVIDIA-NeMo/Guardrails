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

"""Compose transparent Guardrails dispatch with trusted model-routing policy."""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from nemoguardrails.server.experimental._application import create_proxy_application
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_chat_integration
from nemoguardrails.server.experimental.rails import ConfigEntry, ConfigResolver


@dataclass(frozen=True, slots=True)
class ModelRoute:
    """Describe one trusted OpenAI-compatible model destination."""

    base_url: httpx.URL
    upstream_model: str
    api_key_environment_variable: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, httpx.URL):
            raise TypeError("A model route base URL must be an httpx.URL.")
        if not self.base_url.is_absolute_url or self.base_url.scheme not in {"http", "https"}:
            raise ValueError("A model route base URL must be absolute HTTP or HTTPS.")
        if self.base_url.userinfo or self.base_url.query or self.base_url.fragment:
            raise ValueError("A model route base URL must not contain credentials, a query, or a fragment.")
        if not isinstance(self.upstream_model, str) or not self.upstream_model:
            raise ValueError("A provider-facing model must be a non-empty string.")
        if self.api_key_environment_variable is not None and (
            not isinstance(self.api_key_environment_variable, str) or not self.api_key_environment_variable
        ):
            raise ValueError("A model route credential environment variable must be a non-empty string.")


MODEL_ROUTES = {
    "model-2": ModelRoute(
        base_url=httpx.URL(os.getenv("MODEL_2_BASE_URL", "http://model-2:8000/v1")),
        upstream_model="model-2",
        api_key_environment_variable="MODEL_2_API_KEY",
    ),
    "model-3": ModelRoute(
        base_url=httpx.URL(os.getenv("MODEL_3_BASE_URL", "http://model-3:8000/v1")),
        upstream_model="deployment-3",
        api_key_environment_variable="MODEL_3_API_KEY",
    ),
    "model-4": ModelRoute(
        base_url=httpx.URL(os.getenv("MODEL_4_BASE_URL", "http://model-4:8000/v1")),
        upstream_model="vendor/model-4",
        api_key_environment_variable="MODEL_4_API_KEY",
    ),
}

MODEL_ALIASES = {
    "model-1": "model-2",
    "fast": "model-3",
}


class ModelRoutingTransport(httpx.AsyncBaseTransport):
    """Route final outbound HTTPX requests using trusted application policy."""

    def __init__(
        self,
        routes: Mapping[str, ModelRoute],
        aliases: Mapping[str, str],
        *,
        default_model: str,
        network_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not routes or any(not isinstance(name, str) or not name for name in routes):
            raise ValueError("Model routes must use non-empty string names.")
        if any(not isinstance(route, ModelRoute) for route in routes.values()):
            raise TypeError("Every model route must be a ModelRoute.")
        if default_model not in routes:
            raise ValueError("The default model must identify a configured route.")
        if any(
            not isinstance(alias, str) or not alias or not isinstance(target, str) or not target
            for alias, target in aliases.items()
        ):
            raise ValueError("Model aliases and their targets must be non-empty strings.")
        unknown_alias_targets = set(aliases.values()).difference(routes)
        if unknown_alias_targets:
            raise ValueError(f"Aliases reference unknown models: {sorted(unknown_alias_targets)}")
        if network_transport is not None and not isinstance(network_transport, httpx.AsyncBaseTransport):
            raise TypeError("The model router network transport must be an httpx.AsyncBaseTransport.")
        self._routes = dict(routes)
        self._aliases = dict(aliases)
        self._default_model = default_model
        self._network_transport = network_transport or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        original_body = await request.aread()
        payload = self._parse_json_object(request, original_body)
        requested_model = payload.get("model") if payload is not None else None
        if requested_model is not None and not isinstance(requested_model, str):
            return self._routing_error(request, 400, "The model field must be a string.")

        canonical_model = self._aliases.get(requested_model, requested_model)
        if canonical_model is None:
            canonical_model = self._default_model
        route = self._routes.get(canonical_model)
        if route is None:
            return self._routing_error(request, 404, f"No route is configured for model {requested_model!r}.")

        api_key = None
        if route.api_key_environment_variable is not None:
            api_key = os.getenv(route.api_key_environment_variable)
            if not api_key:
                return self._routing_error(
                    request,
                    502,
                    f"The selected route requires {route.api_key_environment_variable}.",
                )

        body = original_body
        body_modified = payload is not None and requested_model is not None and requested_model != route.upstream_model
        if body_modified:
            payload["model"] = route.upstream_model
            body = json.dumps(payload, separators=(",", ":")).encode()

        routed_request = httpx.Request(
            method=request.method,
            url=self._upstream_url(route.base_url, request.url),
            headers=self._upstream_headers(
                request.headers,
                body_modified=body_modified,
                api_key=api_key,
            ),
            content=body,
            extensions=request.extensions,
        )
        return await self._network_transport.handle_async_request(routed_request)

    async def aclose(self) -> None:
        await self._network_transport.aclose()

    @staticmethod
    def _parse_json_object(request: httpx.Request, body: bytes) -> dict[str, Any] | None:
        content_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
        if content_type != "application/json" or not body:
            return None
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _upstream_url(base_url: httpx.URL, original_url: httpx.URL) -> httpx.URL:
        original_path, separator, query = original_url.raw_path.partition(b"?")
        base_path = base_url.raw_path.rstrip(b"/")
        if base_path and (original_path == base_path or original_path.startswith(base_path + b"/")):
            routed_path = original_path
        else:
            routed_path = base_path + b"/" + original_path.lstrip(b"/")
        if separator:
            routed_path += b"?" + query
        return base_url.copy_with(raw_path=routed_path)

    @staticmethod
    def _upstream_headers(
        headers: httpx.Headers,
        *,
        body_modified: bool,
        api_key: str | None,
    ) -> httpx.Headers:
        result = httpx.Headers(headers)
        for name in ("host", "content-length", "authorization", "api-key", "x-api-key", "cookie"):
            result.pop(name, None)
        if api_key is not None:
            result["authorization"] = f"Bearer {api_key}"
        if body_modified:
            for name in ("content-digest", "content-md5", "digest", "signature", "signature-input"):
                result.pop(name, None)
        return result

    @staticmethod
    def _routing_error(request: httpx.Request, status_code: int, message: str) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"error": {"type": "model_routing_error", "message": message}},
            request=request,
        )


def create_model_routing_transport() -> ModelRoutingTransport:
    """Create one routing transport for one application lifespan."""

    return ModelRoutingTransport(
        routes=MODEL_ROUTES,
        aliases=MODEL_ALIASES,
        default_model="model-2",
    )


def create_model_routed_application(
    rails_configs: tuple[ConfigEntry, ...],
    config_resolver: ConfigResolver | None = None,
):
    """Bind the routing transport to the current private proxy composition."""

    openai = create_openai_chat_integration(base_url="http://model-router.internal/v1")
    return create_proxy_application(
        providers=(openai,),
        rails_configs=rails_configs,
        config_resolver=config_resolver,
        http_transport_factory=create_model_routing_transport,
    )

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

from nemoguardrails.server.experimental._http_paths import _strip_mount_prefix


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


class ModelRoutingTransport(httpx.AsyncBaseTransport):
    """Route final outbound HTTPX requests using trusted application policy."""

    def __init__(
        self,
        routes: Mapping[str, ModelRoute],
        aliases: Mapping[str, str],
        *,
        default_model: str,
        network_transport: httpx.AsyncBaseTransport | None = None,
        logical_base_path: str = "/v1",
        max_json_body_bytes: int = 1024 * 1024,
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
        if (
            not isinstance(logical_base_path, str)
            or not logical_base_path.startswith("/")
            or not logical_base_path.isascii()
            or any(character in logical_base_path for character in "? #%")
        ):
            raise ValueError("The logical base path must be an absolute, unencoded ASCII path.")
        if type(max_json_body_bytes) is not int or max_json_body_bytes <= 0:
            raise ValueError("The JSON body limit must be a positive integer.")
        self._routes = dict(routes)
        self._aliases = dict(aliases)
        self._default_model = default_model
        self._logical_base_path = logical_base_path.rstrip("/")
        self._max_json_body_bytes = max_json_body_bytes
        self._network_transport = network_transport or httpx.AsyncHTTPTransport(trust_env=False)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Buffer bounded JSON for model selection; relay other uploads incrementally."""

        assert isinstance(request.stream, httpx.AsyncByteStream)
        content_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
        original_body = None
        if content_type == "application/json":
            buffered = bytearray()
            async for chunk in request.stream:
                if len(chunk) > self._max_json_body_bytes - len(buffered):
                    return self._routing_error(request, 413, "The JSON body exceeds the model-routing limit.")
                buffered.extend(chunk)
            original_body = bytes(buffered)
        payload = self._parse_json_object(original_body) if original_body is not None else None
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

        body = original_body if original_body is not None else request.stream
        body_modified = payload is not None and requested_model is not None and requested_model != route.upstream_model
        if body_modified:
            payload["model"] = route.upstream_model
            body = json.dumps(payload, separators=(",", ":")).encode()

        # Passing an existing stream as content would make HTTPX encode it again.
        routed_request = httpx.Request(
            method=request.method,
            url=self._upstream_url(route.base_url, request.url),
            headers=self._upstream_headers(
                request.headers,
                body_modified=body_modified,
                body_buffered=original_body is not None,
                api_key=api_key,
            ),
            content=body if original_body is not None else None,
            stream=request.stream if original_body is None else None,
            extensions=request.extensions,
        )
        routed_request.headers.setdefault("host", routed_request.url.netloc.decode("ascii"))
        return await self._network_transport.handle_async_request(routed_request)

    async def aclose(self) -> None:
        await self._network_transport.aclose()

    @staticmethod
    def _parse_json_object(body: bytes) -> dict[str, Any] | None:
        """Read model metadata without interpreting unrelated provider payloads."""

        if not body:
            return None
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _upstream_url(self, base_url: httpx.URL, original_url: httpx.URL) -> httpx.URL:
        """Replace the logical base while retaining encoded suffix and query bytes."""

        original_path, separator, query = original_url.raw_path.partition(b"?")
        base_path = base_url.raw_path.rstrip(b"/")
        suffix = _strip_mount_prefix(original_url.path, original_path, self._logical_base_path)
        routed_path = base_path + suffix
        if separator:
            routed_path += b"?" + query
        return base_url.copy_with(raw_path=routed_path)

    @staticmethod
    def _upstream_headers(
        headers: httpx.Headers,
        *,
        body_modified: bool,
        body_buffered: bool,
        api_key: str | None,
    ) -> httpx.Headers:
        """Use destination credentials and rebuild framing for buffered JSON."""

        result = httpx.Headers(headers)
        for name in ("host", "authorization", "api-key", "x-api-key", "cookie"):
            result.pop(name, None)
        if api_key is not None:
            result["authorization"] = f"Bearer {api_key}"
        if body_buffered:
            result.pop("content-length", None)
            result.pop("transfer-encoding", None)
        if body_modified:
            for name in ("content-digest", "content-md5", "digest", "signature", "signature-input"):
                result.pop(name, None)
        return result

    @staticmethod
    def _routing_error(request: httpx.Request, status_code: int, message: str) -> httpx.Response:
        """Return an unread async body for the proxy's raw response reader."""

        body = json.dumps({"error": {"type": "model_routing_error", "message": message}}).encode()
        return httpx.Response(
            status_code,
            headers={"content-type": "application/json", "content-length": str(len(body))},
            stream=httpx.ByteStream(body),
            request=request,
        )

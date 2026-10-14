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

"""Own application-lifespan resources for the private provider proxy."""

import inspect
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from http.cookiejar import Cookie, CookieJar, DefaultCookiePolicy
from typing import Any

import anyio
import httpx
from fastapi import FastAPI, Request

from nemoguardrails.server.experimental._content_checker import ContentChecker
from nemoguardrails.server.experimental.rails import ConfigContext, ConfigEntry, ConfigResolver
from nemoguardrails.server.experimental.rails._repository import GuardrailsRepository

log = logging.getLogger(__name__)
_HTTP_SHUTDOWN_TIMEOUT_SECONDS = 5.0


async def _close_http_resource(resource: httpx.AsyncClient | httpx.AsyncBaseTransport) -> None:
    """Bound cooperative HTTP cleanup without replacing a primary failure."""

    try:
        with anyio.move_on_after(_HTTP_SHUTDOWN_TIMEOUT_SECONDS, shield=True) as scope:
            await resource.aclose()
        if scope.cancelled_caught:
            log.warning("HTTP client cleanup timed out; resources may remain open.")
    except Exception:
        log.warning("HTTP client cleanup failed; resources may remain open.")


class _NoCookiePersistence(DefaultCookiePolicy):
    """Keep pooled proxy connections without retaining an upstream session."""

    def set_ok(self, cookie: Cookie, request: Any) -> bool:
        return False


@dataclass(slots=True)
class ServerRuntime:
    """Group outbound resources shared by one application instance."""

    http_client: httpx.AsyncClient
    content_checkers: GuardrailsRepository
    config_resolver: ConfigResolver | None

    @classmethod
    def create(
        cls,
        entries: Sequence[ConfigEntry],
        resolver: ConfigResolver | None,
        *,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ) -> "ServerRuntime":
        return cls(
            http_client=httpx.AsyncClient(
                follow_redirects=False,
                trust_env=False,
                transport=http_transport,
                cookies=CookieJar(policy=_NoCookiePersistence()),
            ),
            content_checkers=GuardrailsRepository(entries),
            config_resolver=resolver,
        )

    async def resolve_content_checker(
        self,
        request: Request,
        provider_type: str,
        provider_instance_id: str,
        operation_name: str,
    ) -> ContentChecker:
        """Resolve a declared config and return its application-owned checker."""

        if self.config_resolver is None:
            raise ValueError("No Rails config is configured.")
        context = ConfigContext(
            request=request,
            provider_type=provider_type,
            provider_instance_id=provider_instance_id,
            operation_name=operation_name,
        )
        config_id = self.config_resolver(context)
        if inspect.isawaitable(config_id):
            config_id = await config_id
        if not isinstance(config_id, str) or not config_id:
            raise TypeError("The config resolver must return a non-empty config id.")
        return self.content_checkers.get(config_id)

    async def close(self) -> None:
        """Attempt checker and HTTP cleanup independently during shutdown."""

        try:
            await self.content_checkers.close()
        finally:
            await _close_http_resource(self.http_client)


def get_runtime(application: FastAPI) -> ServerRuntime:
    """Return the runtime initialized by the application's active lifespan."""

    runtime = getattr(application.state, "experimental_proxy_runtime", None)
    if not isinstance(runtime, ServerRuntime):
        raise RuntimeError("The experimental proxy runtime is not available.")
    return runtime

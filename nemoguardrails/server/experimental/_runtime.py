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

import logging
from dataclasses import dataclass
from http.cookiejar import Cookie, CookieJar, DefaultCookiePolicy
from typing import Any

import anyio
import httpx
from fastapi import FastAPI

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
class ProxyRuntime:
    """Group outbound resources shared by one application instance."""

    http_client: httpx.AsyncClient

    @classmethod
    def create(
        cls,
        *,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ) -> "ProxyRuntime":
        return cls(
            http_client=httpx.AsyncClient(
                follow_redirects=False,
                trust_env=False,
                transport=http_transport,
                cookies=CookieJar(policy=_NoCookiePersistence()),
            )
        )

    async def close(self) -> None:
        await _close_http_resource(self.http_client)


def get_runtime(application: FastAPI) -> ProxyRuntime:
    """Return the runtime initialized by the application's active lifespan."""

    runtime = getattr(application.state, "experimental_proxy_runtime", None)
    if not isinstance(runtime, ProxyRuntime):
        raise RuntimeError("The experimental proxy runtime is not available.")
    return runtime

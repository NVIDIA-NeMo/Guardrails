# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Generic API engine for IORails, calling arbitrary REST endpoints via aiohttp with retry."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional, cast

import aiohttp
from aiohttp_retry import RetryClient

from nemoguardrails.guardrails._http import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT_CONNECT,
    DEFAULT_TIMEOUT_TOTAL,
    safe_read_body,
)
from nemoguardrails.guardrails.base_engine import BaseEngine
from nemoguardrails.guardrails.guardrails_types import get_request_id, truncate

if TYPE_CHECKING:
    from nemoguardrails.rails.llm.config import JailbreakDetectionConfig

log = logging.getLogger(__name__)


class APIEngineError(Exception):
    """Raised when an API engine call fails."""

    def __init__(self, message: str, endpoint: str, status: int | None = None) -> None:
        self.endpoint = endpoint
        self.status = status
        super().__init__(message)


class APIEngine(BaseEngine):
    """Wraps a single API endpoint and makes HTTP calls with retry support."""

    def __init__(
        self,
        *,
        base_url: str,
        endpoint: str,
        api_key: Optional[str] = None,
        timeout_total: float = DEFAULT_TIMEOUT_TOTAL,
        timeout_connect: float = DEFAULT_TIMEOUT_CONNECT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        """Configure the API endpoint URL, auth key, and retry settings."""
        self.base_url = base_url
        self.endpoint = endpoint
        self.api_key = api_key

        super().__init__(
            timeout_total=timeout_total,
            timeout_connect=timeout_connect,
            max_attempts=max_attempts,
        )

    @property
    def url(self) -> str:
        """Full URL for the API endpoint."""
        return self.base_url.rstrip("/") + "/" + self.endpoint.lstrip("/")

    @classmethod
    def from_jailbreak_config(cls, jailbreak_config: JailbreakDetectionConfig) -> APIEngine:
        """Create an APIEngine from a JailbreakDetectionConfig."""
        if not jailbreak_config.nim_base_url:
            raise ValueError("jailbreak_detection.nim_base_url is required for IORails jailbreak detection")
        if not jailbreak_config.nim_server_endpoint:
            raise ValueError("jailbreak_detection.nim_server_endpoint is required for IORails jailbreak detection")

        return cls(
            base_url=jailbreak_config.nim_base_url,
            endpoint=jailbreak_config.nim_server_endpoint,
            api_key=jailbreak_config.get_api_key(),
        )

    async def call(self, body: dict[str, Any], **kwargs) -> dict:
        """POST the JSON body to the configured endpoint and return the parsed response."""
        if not self._running:
            raise APIEngineError("APIEngine has not been started. Call start() first.", endpoint=self.url)

        client = cast(RetryClient, self._client)
        url = self.url
        request_body: dict[str, Any] = {**body, **kwargs}
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req_id = get_request_id()
        log.info("[%s] HTTP POST %s", req_id, url)
        log.debug("[%s] HTTP request body: %s", req_id, truncate(request_body))

        t0 = time.monotonic()
        try:
            async with client.post(url, json=request_body, headers=headers) as response:
                elapsed_ms = (time.monotonic() - t0) * 1000

                if response.status >= 400:
                    error_body = await safe_read_body(response)
                    log.warning("[%s] HTTP %s from endpoint '%s' time=%.1fms", req_id, response.status, url, elapsed_ms)
                    raise APIEngineError(
                        f"HTTP {response.status} from endpoint '{url}': {error_body}",
                        endpoint=url,
                        status=response.status,
                    )

                result = await response.json()
                log.debug(
                    "[%s] HTTP response status=%s time=%.1fms body: %s",
                    req_id,
                    response.status,
                    elapsed_ms,
                    truncate(result),
                )
                return result

        except aiohttp.ContentTypeError as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            log.warning("[%s] Failed to parse response as JSON time=%.1fms", req_id, elapsed_ms)
            raise APIEngineError(f"Failed to parse response as JSON: {exc}", endpoint=url, status=exc.status) from exc

        except APIEngineError:
            raise
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            log.warning("[%s] Request to endpoint '%s' failed time=%.1fms", req_id, url, elapsed_ms)
            raise APIEngineError(
                f"Request to endpoint '{url}' failed: {exc}",
                endpoint=url,
            ) from exc

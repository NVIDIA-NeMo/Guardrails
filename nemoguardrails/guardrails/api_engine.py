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
from typing import TYPE_CHECKING, Any, Optional, cast

import aiohttp
from aiohttp_retry import ExponentialRetry, RetryClient

if TYPE_CHECKING:
    from nemoguardrails.rails.llm.config import JailbreakDetectionConfig

log = logging.getLogger(__name__)

_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_TIMEOUT_TOTAL = 30
_DEFAULT_TIMEOUT_CONNECT = 5
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class APIEngineError(Exception):
    """Raised when an API engine call fails."""

    def __init__(self, message: str, endpoint: str, status: int | None = None) -> None:
        self.endpoint = endpoint
        self.status = status
        super().__init__(message)


class APIEngine:
    """Wraps a single API endpoint and makes HTTP calls with retry support."""

    def __init__(
        self,
        *,
        base_url: str,
        endpoint: str,
        api_key: Optional[str] = None,
        timeout_total: float = _DEFAULT_TIMEOUT_TOTAL,
        timeout_connect: float = _DEFAULT_TIMEOUT_CONNECT,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.base_url = base_url
        self.endpoint = endpoint
        self.api_key = api_key

        self._timeout = aiohttp.ClientTimeout(
            total=timeout_total,
            connect=timeout_connect,
        )
        self._retry_options = ExponentialRetry(
            attempts=max_attempts,
            statuses=set(_RETRYABLE_STATUS_CODES),
            exceptions={aiohttp.ClientConnectionError},
        )
        self._client: Optional[RetryClient] = None
        self._running = False

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

    async def start(self) -> None:
        """Create this engine's RetryClient. Call this during service startup."""
        if self._running:
            return

        self._client = RetryClient(
            retry_options=self._retry_options,
            client_session=aiohttp.ClientSession(timeout=self._timeout),
        )
        self._running = True

    async def stop(self) -> None:
        """Close this engine's RetryClient. Call this during service shutdown."""
        if not self._running:
            return

        try:
            if self._client:
                await self._client.close()
                self._client = None
        finally:
            self._running = False

    async def call(self, body: dict[str, Any], **kwargs) -> dict:
        """POST the JSON body to the configured endpoint and return the parsed response."""
        if not self._running:
            await self.start()

        client = cast(RetryClient, self._client)
        url = self.url
        request_body: dict[str, Any] = {**body, **kwargs}
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with client.post(url, json=request_body, headers=headers) as response:
                if response.status >= 400:
                    error_body = await _safe_read_body(response)
                    raise APIEngineError(
                        f"HTTP {response.status} from endpoint '{url}': {error_body}",
                        endpoint=url,
                        status=response.status,
                    )
                return await response.json()

        except APIEngineError:
            raise
        except Exception as exc:
            raise APIEngineError(
                f"Request to endpoint '{url}' failed: {exc}",
                endpoint=url,
            ) from exc

    async def __aenter__(self):
        """Context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.stop()


async def _safe_read_body(response: aiohttp.ClientResponse, max_chars: int = 500) -> str:
    """Read response body for error messages, truncating if too large."""
    try:
        text = await response.text()
        return text[:max_chars] if len(text) > max_chars else text
    except Exception:
        return "<could not read response body>"

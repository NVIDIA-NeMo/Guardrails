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

"""Model engine for IORails.

Wraps a single Model config and makes raw HTTP calls to its
OpenAI-compatible /v1/chat/completions endpoint via aiohttp.
Retries are handled by aiohttp-retry (ExponentialRetry).
"""

import logging
import os
import time
from typing import Any, Optional, cast

import aiohttp
from aiohttp_retry import ExponentialRetry, RetryClient

from nemoguardrails.guardrails._http import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT_CONNECT,
    DEFAULT_TIMEOUT_TOTAL,
    RETRYABLE_STATUS_CODES,
    safe_read_body,
)
from nemoguardrails.guardrails.guardrails_types import LLMMessages, get_request_id, truncate
from nemoguardrails.rails.llm.config import Model

log = logging.getLogger(__name__)

# Default base URLs by engine type
_ENGINE_BASE_URLS = {
    "nim": "https://integrate.api.nvidia.com",
    "openai": "https://api.openai.com",
}

_CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"


class ModelEngineError(Exception):
    """Raised when a model engine call fails."""

    def __init__(self, message: str, model_name: str, status: int | None = None) -> None:
        self.model_name = model_name
        self.status = status
        super().__init__(message)


class ModelEngine:
    """Wraps a single Model config and makes HTTP calls to its endpoint.

    Each ModelEngine owns its own RetryClient with per-model timeout,
    retry, and connection pool settings.
    """

    def __init__(self, model_config: Model) -> None:
        self.model_config = model_config
        self.model_name: str = model_config.model or ""
        self.base_url: str = self._resolve_base_url()
        self.api_key: Optional[str] = self._resolve_api_key(model_config.engine)

        # Configurable from model parameters
        params = model_config.parameters or {}
        self._timeout = aiohttp.ClientTimeout(
            total=float(params.get("timeout", DEFAULT_TIMEOUT_TOTAL)),
            connect=float(params.get("timeout_connect", DEFAULT_TIMEOUT_CONNECT)),
        )
        self._retry_options = ExponentialRetry(
            attempts=int(params.get("max_attempts", DEFAULT_MAX_ATTEMPTS)),
            statuses=set(RETRYABLE_STATUS_CODES),
            exceptions={aiohttp.ClientConnectionError},
        )
        self._client: Optional[RetryClient] = None
        self._running = False

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

    def _resolve_base_url(self) -> str:
        """Resolve the base URL from model parameters or engine type."""
        params = self.model_config.parameters or {}

        if params.get("base_url"):
            return params["base_url"]

        engine = self.model_config.engine
        if engine in _ENGINE_BASE_URLS:
            return _ENGINE_BASE_URLS[engine]

        raise ValueError(
            f"No base_url in parameters and cannot infer from engine '{engine}' for model '{self.model_name}'"
        )

    def _get_environment_variable(self, variable_name: str) -> str | None:
        """Return the value stored in environment variable `variable_name`."""
        env_value = os.environ.get(variable_name)
        return env_value

    def _resolve_api_key(self, engine: str | None) -> Optional[str]:
        """Resolve the API key from model config or environment."""
        if self.model_config.api_key_env_var:
            env_value = self._get_environment_variable(self.model_config.api_key_env_var)

            # Only raise an exception if the user provided an API Key env var and it isn't set
            if not env_value:
                raise RuntimeError(f"Environment variable '{self.model_config.api_key_env_var}' not set")
            return env_value

        if engine == "nim":
            env_value = self._get_environment_variable("NVIDIA_API_KEY")
            return env_value

        if engine == "openai":
            env_value = self._get_environment_variable("OPENAI_API_KEY")
            return env_value

        # If no key is available, assume it's a local model that doesn't need one
        return None

    async def call(
        self,
        messages: LLMMessages,
        **kwargs: Any,
    ) -> dict:
        """Make a POST request to the /v1/chat/completions endpoint.

        Retries on transient failures (429, 5xx, connection errors) are
        handled automatically by the RetryClient with exponential backoff.

        Args:
            messages: List of message dicts in OpenAI format.
            **kwargs: Additional parameters for the request body (temperature, max_tokens, etc.)

        Returns:
            The parsed JSON response dict from the API.

        Raises:
            ModelEngineError: If the request fails after all retries.
        """

        if not self._running:
            raise ModelEngineError(
                f"ModelEngine for '{self.model_name}' has not been started. Call start() first.",
                model_name=self.model_name,
            )

        # Cast as RetryClient so type-checking knows it isn't None
        client = cast(RetryClient, self._client)

        url = self.base_url.rstrip("/") + _CHAT_COMPLETIONS_ENDPOINT

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            **kwargs,
        }

        req_id = get_request_id()
        log.info("[%s] HTTP POST %s model='%s'", req_id, url, self.model_name)
        log.debug("[%s] HTTP request body: %s", req_id, truncate(body))

        t0 = time.monotonic()
        try:
            async with client.post(url, json=body, headers=headers) as response:
                elapsed_ms = (time.monotonic() - t0) * 1000

                if response.status >= 400:
                    error_body = await safe_read_body(response)
                    log.warning(
                        "[%s] HTTP %s from model '%s' time=%.1fms", req_id, response.status, self.model_name, elapsed_ms
                    )
                    raise ModelEngineError(
                        f"HTTP {response.status} from model '{self.model_name}': {error_body}",
                        model_name=self.model_name,
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

        except ModelEngineError:
            raise
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            log.warning("[%s] Request to model '%s' failed time=%.1fms", req_id, self.model_name, elapsed_ms)
            raise ModelEngineError(
                f"Request to model '{self.model_name}' failed: {exc}",
                model_name=self.model_name,
            ) from exc

    async def __aenter__(self):
        """Context manager (used for testing rather than long-lived instance)"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager (used for testing rather than long-lived instance)"""
        await self.stop()

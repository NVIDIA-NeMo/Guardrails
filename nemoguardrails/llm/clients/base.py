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

import asyncio
import json
import logging
import random
import warnings
from typing import Any, AsyncGenerator, Dict, Optional
from urllib.parse import urlparse

import httpx

from nemoguardrails.exceptions import LLMConnectionError, LLMResponseValidationError, LLMTimeoutError
from nemoguardrails.llm.clients._errors import (
    ErrorContext,
    _parse_retry_after_value,
    raise_for_sse_error,
    raise_for_status,
)
from nemoguardrails.llm.clients._sse import SSEDecoder
from nemoguardrails.llm.clients.constants import (
    DEFAULT_CONNECTION_LIMITS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    INITIAL_RETRY_DELAY,
    MAX_RETRY_AFTER,
    MAX_RETRY_DELAY,
    RETRYABLE_STATUS_CODES,
)

log = logging.getLogger(__name__)


class BaseClient:
    _client: httpx.AsyncClient

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        connect_timeout: Optional[float] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        custom_headers: Optional[Dict[str, str]] = None,
        custom_query: Optional[Dict[str, Any]] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        """Initialize the HTTP-backed LLM client.

        custom_headers takes precedence over api_key-derived headers.
        Passing custom_headers={"Authorization": "..."} overrides the
        Bearer token built from api_key, allowing custom auth schemes
        (Basic, raw JWT, pre-signed tokens, multi-scheme) to be used
        alongside the env-var-driven api_key default.
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_retries = max_retries
        self._custom_headers = custom_headers or {}
        self._custom_query = custom_query or {}

        if api_key and base_url.startswith("http://"):
            host = (urlparse(base_url).hostname or "").lower()
            if host not in ("localhost", "127.0.0.1", "::1") and not host.endswith(".local"):
                warnings.warn(
                    f"API key will be sent over plaintext HTTP to {base_url}; use https:// for production deployments.",
                    UserWarning,
                    stacklevel=2,
                )

        if http_client is not None and not isinstance(http_client, httpx.AsyncClient):
            raise TypeError(f"Invalid http_client argument; expected httpx.AsyncClient but got {type(http_client)}")

        _timeout = timeout if timeout is not None else DEFAULT_TIMEOUT.read
        _connect_timeout = connect_timeout if connect_timeout is not None else DEFAULT_TIMEOUT.connect
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(_timeout, connect=_connect_timeout),
            limits=DEFAULT_CONNECTION_LIMITS,
        )

    @property
    def provider_name(self) -> Optional[str]:
        return None

    @property
    def provider_url(self) -> Optional[str]:
        return None

    def _error_context(self) -> ErrorContext:
        return ErrorContext(
            model_name=None,
            provider_name=None,
            base_url=self.provider_url,
        )

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        for name, value in self._custom_headers.items():
            existing = next((k for k in headers if k.lower() == name.lower()), None)
            if existing is not None:
                del headers[existing]
            headers[name] = value
        return headers

    @staticmethod
    def _should_retry(status_code: int, headers: Any) -> bool:
        should = (headers.get("x-should-retry") or "").lower()
        if should == "false":
            return False
        if should == "true":
            return True
        return status_code in RETRYABLE_STATUS_CODES

    @staticmethod
    def _calculate_retry_delay(headers: Any, retries_attempted: int) -> float:
        retry_after = headers.get("retry-after")
        if retry_after:
            delay = _parse_retry_after_value(retry_after)
            if delay is not None and 0 < delay <= MAX_RETRY_AFTER:
                return delay
            if delay is None:
                log.debug("Ignoring unparseable Retry-After=%r", retry_after)
            else:
                log.debug("Ignoring Retry-After=%r (out of range, max=%s)", retry_after, MAX_RETRY_AFTER)

        sleep_cap = min(INITIAL_RETRY_DELAY * (2.0**retries_attempted), MAX_RETRY_DELAY)
        return random.uniform(0, sleep_cap)

    async def _sleep_for_retry(self, retries_attempted: int, headers: Any = None) -> None:
        delay = self._calculate_retry_delay(headers or {}, retries_attempted)
        log.info("Retrying (delay=%.1fs, attempted=%d)", delay, retries_attempted)
        await asyncio.sleep(delay)

    async def _apost(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        retries_remaining = self._max_retries
        retries_attempted = 0
        ctx = self._error_context()

        while True:
            try:
                response = await self._client.post(
                    f"{self._base_url}{path}",
                    json=payload,
                    headers=self._build_headers(),
                    params=self._custom_query or None,
                )
            except httpx.TimeoutException as err:
                if retries_remaining > 0:
                    await self._sleep_for_retry(retries_attempted)
                    retries_remaining -= 1
                    retries_attempted += 1
                    continue
                raise LLMTimeoutError(0, f"Request timed out: {err}", **ctx.as_kwargs()) from err
            except httpx.NetworkError as err:
                if retries_remaining > 0:
                    await self._sleep_for_retry(retries_attempted)
                    retries_remaining -= 1
                    retries_attempted += 1
                    continue
                raise LLMConnectionError(0, f"Connection error: {err}", **ctx.as_kwargs()) from err

            if self._should_retry(response.status_code, response.headers) and retries_remaining > 0:
                await self._sleep_for_retry(retries_attempted, response.headers)
                retries_remaining -= 1
                retries_attempted += 1
                continue

            if response.status_code >= 400:
                raise_for_status(response.status_code, response.text, response.headers, ctx)
            try:
                data = response.json()
            except json.JSONDecodeError as err:
                raise LLMResponseValidationError(
                    f"Provider returned non-JSON response (status={response.status_code}, "
                    f"content-type={response.headers.get('content-type')!r}): {err}",
                    response_data=None,
                    **ctx.as_kwargs(),
                ) from err
            data["_response_headers"] = dict(response.headers)
            return data

    async def _apost_stream(self, path: str, payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        retries_remaining = self._max_retries
        retries_attempted = 0
        ctx = self._error_context()

        while True:
            first_yielded = False
            try:
                async with self._client.stream(
                    "POST",
                    f"{self._base_url}{path}",
                    json=payload,
                    headers=self._build_headers(),
                    params=self._custom_query or None,
                ) as response:
                    if self._should_retry(response.status_code, response.headers) and retries_remaining > 0:
                        await response.aread()
                        await self._sleep_for_retry(retries_attempted, response.headers)
                        retries_remaining -= 1
                        retries_attempted += 1
                        continue

                    if response.status_code >= 400:
                        await response.aread()
                        raise_for_status(response.status_code, response.text, response.headers, ctx)

                    response_headers = dict(response.headers)
                    decoder = SSEDecoder()
                    async for line in response.aiter_lines():
                        sse = decoder.decode(line)
                        if sse is None:
                            continue
                        if sse.data.startswith("[DONE]"):
                            return
                        try:
                            parsed = sse.json()
                        except json.JSONDecodeError as err:
                            raise LLMResponseValidationError(
                                f"Malformed SSE chunk: {sse.data[:200]!r}: {err}",
                                response_data=None,
                                **ctx.as_kwargs(),
                            ) from err
                        parsed["_response_headers"] = response_headers
                        self._check_sse_error(parsed, response.headers, ctx)
                        first_yielded = True
                        yield parsed

                    sse = decoder.decode("")
                    if sse is not None and not sse.data.startswith("[DONE]"):
                        try:
                            parsed = sse.json()
                        except json.JSONDecodeError as err:
                            raise LLMResponseValidationError(
                                f"Malformed trailing SSE chunk: {sse.data[:200]!r}: {err}",
                                response_data=None,
                                **ctx.as_kwargs(),
                            ) from err
                        parsed["_response_headers"] = response_headers
                        self._check_sse_error(parsed, response.headers, ctx)
                        yield parsed
                    return
            except httpx.TimeoutException as err:
                if first_yielded or retries_remaining <= 0:
                    raise LLMTimeoutError(0, f"Request timed out: {err}", **ctx.as_kwargs()) from err
                await self._sleep_for_retry(retries_attempted)
                retries_remaining -= 1
                retries_attempted += 1
                continue
            except httpx.NetworkError as err:
                if first_yielded or retries_remaining <= 0:
                    raise LLMConnectionError(0, f"Connection error: {err}", **ctx.as_kwargs()) from err
                await self._sleep_for_retry(retries_attempted)
                retries_remaining -= 1
                retries_attempted += 1
                continue

    def _check_sse_error(self, parsed: Any, headers: Any, ctx: Optional[ErrorContext] = None) -> None:
        if not isinstance(parsed, dict) or "error" not in parsed:
            return
        raise_for_sse_error(parsed, headers, ctx)

    def is_closed(self) -> bool:
        return self._client.is_closed

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

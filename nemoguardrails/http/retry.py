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
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping

from nemoguardrails.http.client import HTTPClient, ManagedHTTPClient
from nemoguardrails.http.errors import HTTPConnectionError, HTTPTimeoutError
from nemoguardrails.http.types import HTTPResponse


def _header(headers: Mapping[str, str], name: str) -> str | None:
    lowered_name = name.lower()
    return next((value for key, value in headers.items() if key.lower() == lowered_name), None)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    retryable_methods: frozenset[str] | None = None
    retryable_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({408, 409, 429, 500, 502, 503, 504})
    )
    initial_delay: float = 0.5
    max_delay: float = 8.0
    max_retry_after: float = 60.0
    retry_transport_errors: bool = True
    honor_retry_override_header: bool = False
    clamp_retry_after: bool = False

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_delay < 0:
            raise ValueError("initial_delay must not be negative")
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay must be greater than or equal to initial_delay")
        if self.max_retry_after < 0:
            raise ValueError("max_retry_after must not be negative")
        if self.retryable_methods is not None:
            object.__setattr__(
                self, "retryable_methods", frozenset(method.upper() for method in self.retryable_methods)
            )

    def should_retry(self, response: HTTPResponse) -> bool:
        if self.honor_retry_override_header:
            override = _header(response.headers, "x-should-retry")
            if override is not None:
                if override.lower() == "true":
                    return True
                if override.lower() == "false":
                    return False
        return response.status_code in self.retryable_status_codes

    def can_retry_method(self, method: str) -> bool:
        return self.retryable_methods is None or method.upper() in self.retryable_methods

    def retry_after(self, response: HTTPResponse, *, now: datetime) -> float | None:
        value = _header(response.headers, "retry-after")
        if value is None:
            return None
        try:
            delay = float(value)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            delay = (parsed - now).total_seconds()
        if self.clamp_retry_after:
            return min(max(delay, 0.0), self.max_retry_after)
        if 0 < delay <= self.max_retry_after:
            return delay
        return None


class RetryingHTTPClient:
    def __init__(
        self,
        client: HTTPClient,
        policy: RetryPolicy | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random_value: Callable[[], float] = random.random,
        now: Callable[[], datetime] | None = None,
    ):
        self._client = client
        self._policy = policy or RetryPolicy()
        self._sleep = sleep
        self._random_value = random_value
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._closed = False

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        content: bytes | str | None = None,
        timeout: float | None = None,
    ) -> HTTPResponse:
        retries = 0
        can_retry_method = self._policy.can_retry_method(method)
        while True:
            try:
                response = await self._client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json,
                    content=content,
                    timeout=timeout,
                )
            except (HTTPConnectionError, HTTPTimeoutError) as error:
                if (
                    not can_retry_method
                    or not self._policy.retry_transport_errors
                    or retries + 1 >= self._policy.max_attempts
                ):
                    error.retry_count = retries
                    raise
                await self._sleep(self._backoff(retries))
                retries += 1
                continue

            if (
                not can_retry_method
                or not self._policy.should_retry(response)
                or retries + 1 >= self._policy.max_attempts
            ):
                extensions = dict(response.extensions)
                extensions["retry_count"] = retries
                return replace(response, extensions=extensions)

            delay = self._policy.retry_after(response, now=self._now())
            await self._sleep(delay if delay is not None else self._backoff(retries))
            retries += 1

    def _backoff(self, retries: int) -> float:
        cap = min(self._policy.initial_delay * (2**retries), self._policy.max_delay)
        return cap * self._random_value()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if isinstance(self._client, ManagedHTTPClient):
            await self._client.close()

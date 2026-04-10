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

"""Base class for IORails HTTP engines with aiohttp retry client lifecycle."""

import asyncio
from typing import Optional

import aiohttp
from aiohttp_retry import ExponentialRetry, RetryClient

from nemoguardrails.guardrails._http import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT_CONNECT,
    DEFAULT_TIMEOUT_TOTAL,
    RETRYABLE_STATUS_CODES,
)


class BaseEngine:
    """HTTP engine base with aiohttp RetryClient lifecycle.

    Manages a single RetryClient with configurable timeout and retry settings.
    Subclasses add endpoint-specific call logic.
    """

    def __init__(
        self,
        *,
        timeout_total: float = DEFAULT_TIMEOUT_TOTAL,
        timeout_connect: float = DEFAULT_TIMEOUT_CONNECT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        """Initialize timeout, retry options, and client state."""
        self._timeout = aiohttp.ClientTimeout(
            total=timeout_total,
            connect=timeout_connect,
        )
        self._retry_options = ExponentialRetry(
            attempts=max_attempts,
            statuses=set(RETRYABLE_STATUS_CODES),
            exceptions={aiohttp.ClientConnectionError},
        )
        self._client: Optional[RetryClient] = None
        self._running = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Create this engine's RetryClient. Call during service startup."""
        async with self._lock:
            if self._running:
                return

            self._client = RetryClient(
                retry_options=self._retry_options,
                client_session=aiohttp.ClientSession(timeout=self._timeout),
            )
            self._running = True

    async def stop(self) -> None:
        """Close this engine's RetryClient. Call during service shutdown."""
        async with self._lock:
            if not self._running:
                return

            try:
                if self._client:
                    await self._client.close()
                    self._client = None
            finally:
                self._running = False

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

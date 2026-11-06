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

"""Best-effort, bounded cleanup independent of stream iteration state."""

import inspect
import logging
from collections.abc import Awaitable, Callable

import anyio

log = logging.getLogger(__name__)

_STREAM_CLOSE_TIMEOUT_SECONDS = 5.0
StreamClose = Callable[[], Awaitable[None]]


async def close_stream(source: object, *, close: StreamClose | None = None) -> None:
    """Close an iterator and, independently, its resource owner.

    Each attempt has a shielded five-second budget. Timeouts and ordinary close
    failures are logged without replacing the stream's outcome. Resource owners
    must make closing idempotent and cancellation-cooperative. Iterator cleanup
    alone cannot recover resources from an interrupted async-generator finally.
    This does not forcibly stop blocking code or cancellation-suppressing code.
    """

    async def attempt(closer: Callable[[], object] | None) -> None:
        """Bound one closer without swallowing cancellation from its caller."""
        if not callable(closer):
            return
        try:
            with anyio.move_on_after(_STREAM_CLOSE_TIMEOUT_SECONDS, shield=True) as scope:
                result = closer()
                if inspect.isawaitable(result):
                    await result
            if scope.cancelled_caught:
                log.warning("Stream cleanup timed out; resources may remain open.")
        except Exception:
            log.warning("Stream cleanup failed; resources may remain open.")

    try:
        await attempt(getattr(source, "aclose", None))
    finally:
        await attempt(close)

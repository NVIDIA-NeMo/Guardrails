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

"""Shared async test helpers.

Primitives for tests that coordinate with asyncio-backed components.
Centralised here so the polling / timeout patterns are consistent across
``test_async_work_queue.py``, ``test_iorails_telemetry.py``, and other
tests that need to observe state transitions mid-flight.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator
from unittest.mock import patch

from nemoguardrails.guardrails.async_work_queue import AsyncWorkQueue
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.rails.llm.config import RailsConfig


@asynccontextmanager
async def started_iorails(config_dict: dict) -> AsyncIterator[IORails]:
    """Build, start, yield, then stop an IORails instance for tests.

    Centralises the ``async with iorails`` lifecycle pattern duplicated across
    ``test_iorails_streaming.py``, ``test_iorails_telemetry.py``, and
    ``test_iorails_reasoning.py``.  Each fixture becomes a one-liner::

        @pytest_asyncio.fixture
        async def iorails():
            async with started_iorails(NEMOGUARDS_CONFIG) as iorails:
                yield iorails

    The ``NVIDIA_API_KEY`` env patch covers config loading; the ``async with``
    on the IORails instance starts the worker queue and stops it on teardown
    so no asyncio tasks leak past the test's event loop.
    """
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
        iorails = IORails(RailsConfig.from_content(config=config_dict))
    async with iorails:
        yield iorails


async def wait_for_queue_state(
    queue: AsyncWorkQueue,
    busy: int,
    pending: int,
    timeout: float = 1.0,
) -> None:
    """Poll ``queue`` until it reaches ``(busy, pending)`` or time out.

    Replaces fixed ``asyncio.sleep(<magic>)`` spins in tests that need the
    worker pool to reach a known state before asserting on it.  Yielding
    via ``sleep(0)`` re-runs the scheduler each iteration, so the helper
    returns the moment the state is correct rather than after a full
    fixed delay.

    Raises ``AssertionError`` on timeout, carrying the last observed
    state in the message for faster debugging of flaky setups.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while (queue.num_busy_workers(), queue.num_pending()) != (busy, pending):
        if loop.time() > deadline:
            raise AssertionError(
                f"timed out waiting for queue state busy={busy} pending={pending}; "
                f"last seen busy={queue.num_busy_workers()} pending={queue.num_pending()}"
            )
        await asyncio.sleep(0)


def saturate_stream_semaphore(iorails: IORails) -> None:
    """Force ``iorails._stream_semaphore`` into a fully-occupied state by
    swapping in a zero-permit semaphore.  Any subsequent
    ``stream_async()`` call trips ``Semaphore.locked() == True`` and is
    rejected with ``asyncio.QueueFull``.

    Cheaper and more future-proof than draining all
    ``STREAM_MAX_CONCURRENCY`` permits one-by-one — the test no longer
    breaks silently if that constant grows.
    """
    iorails._stream_semaphore = asyncio.Semaphore(0)

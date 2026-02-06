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
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Generic, List, Tuple, TypeVar

T = TypeVar("T")

log = logging.getLogger(__name__)


# --- IMMUTABLE WORK ITEM ---
@dataclass(slots=True, frozen=True)
class WorkItem(Generic[T]):
    """Dataclass with async function, args, kwargs, and a future to return result"""

    func: Callable[..., Awaitable[T]]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    future: asyncio.Future[T]


class AsyncWorkQueue(Generic[T]):
    """Async Work Queue with static concurrency and queue size"""

    # Time (seconds) to wait before restarting a crashed worker loop.
    # Prevents CPU spin if the error is persistent.
    WORKER_ERROR_BACKOFF_SECONDS: float = 0.1

    def __init__(self, name: str, max_queue_size: int, max_concurrency: int, reject_on_full: bool = False) -> None:
        self._name = name  # Used in logging to identify overflows
        self._max_queue_size = max_queue_size
        self._max_concurrency = max_concurrency
        self._reject_on_full = reject_on_full

        # Internal state
        self._queue: asyncio.Queue[WorkItem[T]] = asyncio.Queue(maxsize=max_queue_size)
        self._workers: List[asyncio.Task] = []
        self._busy_count = 0
        self._running = False

    def num_busy_workers(self) -> int:
        """Returns the number of workers currently executing a task."""
        return self._busy_count

    def is_queue_full(self) -> bool:
        """Returns True if the queue is currently full."""
        return self._queue.full()

    def is_queue_empty(self) -> bool:
        """Returns True if the queue has zero pending items."""
        return self._queue.empty()

    def is_busy(self) -> bool:
        """Returns True if any worker is currently executing a task."""
        return self._busy_count > 0

    async def start(self) -> None:
        """Starts the worker pool. Call this during service startup."""
        if self._running:
            return
        self._running = True
        self._busy_count = 0
        self._workers = []
        for i in range(self._max_concurrency):
            task = asyncio.create_task(self._worker_loop(), name=f"{self._name}_worker_id{i}")
            self._workers.append(task)

    async def stop(self, wait_for_completion: bool = True) -> None:
        """Stops the worker pool. Call this during service shutdown."""
        if not self._running:
            return
        self._running = False

        if wait_for_completion:
            await self._queue.join()

        for task in self._workers:
            task.cancel()

        # Swallow cancellations to prevent noise during shutdown
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        self._busy_count = 0

    async def submit(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """
        Submit a task.
        If queue is full:
            - self._reject_on_full=True  -> Raises asyncio.QueueFull
            - self._reject_on_full=False -> Blocks caller until slot opens

        Note: Automatically starts the queue on first submission (lazy initialization).
        """
        # Lazy initialization: auto-start on first use
        if not self._running:
            log.info("AsyncWorkQueue %s not running on first task submit, starting now", self._name)
            await self.start()

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        item = WorkItem(func, args, kwargs, future)

        # If the queue is full, this will raise an asyncio.QueueFull exception to be caught
        # at a higher-level object (e.g. FastAPI to return a 429 or 503 error)
        if self._reject_on_full:
            self._queue.put_nowait(item)
        else:
            await self._queue.put(item)

        return await future

    async def _worker_loop(self) -> None:
        while True:
            try:
                item: WorkItem[T] = await self._queue.get()

                if item.future.cancelled():
                    self._queue.task_done()
                    continue

                try:
                    self._busy_count += 1  # No race condition since it's not await'ed
                    result = await item.func(*item.args, **item.kwargs)
                    if not item.future.cancelled():
                        item.future.set_result(result)
                except Exception as e:
                    if not item.future.cancelled():
                        item.future.set_exception(e)
                finally:
                    self._busy_count -= 1  # No race condition since it's not await'ed
                    self._queue.task_done()

            except asyncio.CancelledError:
                break

            except Exception as e:
                task = asyncio.current_task()
                worker_name = task.get_name() if task else "unknown-worker"
                log.critical(f"{worker_name} crashed due to {type(e).__name__}: {e}", exc_info=True)

                # Wait before the worker goes back to waiting for new work queue items
                await asyncio.sleep(self.WORKER_ERROR_BACKOFF_SECONDS)

    async def __aenter__(self):
        """Context manager (used for testing rather than long-lived instance)"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager (used for testing rather than long-lived instance)"""
        await self.stop()

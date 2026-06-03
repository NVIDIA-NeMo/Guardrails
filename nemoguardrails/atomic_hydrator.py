#
# Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module providing atomic state hydration mechanisms for concurrent evaluation pipelines."""

import asyncio
from typing import Any, Dict, Optional


class AtomicStateHydrator:
    """Manages sharded, thread-safe asynchronous locks per conversation id.

    Prevents state mutation race conditions under high throughput.
    """

    def __init__(self, backend_client: Any) -> None:
        """Initializes the hydrator component with an abstract storage backend.

        Args:
            backend_client: The high-level caching or database engine client.
        """
        self.backend = backend_client
        self._locks: Dict[str, asyncio.Lock] = {}
        self._lock_creation_mutex: Optional[asyncio.Lock] = None
        self._ref_counts: Dict[str, int] = {}

    def _ensure_mutex(self) -> asyncio.Lock:
        """Lazily instantiates the underlying memory barrier inside the active event loop.

        Returns:
            asyncio.Lock: The active context lock instance.
        """
        if self._lock_creation_mutex is None:
            self._lock_creation_mutex = asyncio.Lock()
        return self._lock_creation_mutex

    async def _acquire_session_lock(self, conversation_id: str) -> asyncio.Lock:
        """Tracks in-flight utilization and builds a local memory barrier for the session.

        Args:
            conversation_id: Unique string identifier for the active tracking sequence.

        Returns:
            asyncio.Lock: The localized session block primitive.
        """
        mutex = self._ensure_mutex()
        async with mutex:
            if conversation_id not in self._locks:
                self._locks[conversation_id] = asyncio.Lock()
                self._ref_counts[conversation_id] = 0
            self._ref_counts[conversation_id] += 1
            return self._locks[conversation_id]

    async def _release_session_lock(self, conversation_id: str) -> None:
        """Decrements reference counter and releases resources if execution queue drops to zero.

        Args:
            conversation_id: Unique string identifier for the active tracking sequence.
        """
        mutex = self._ensure_mutex()
        async with mutex:
            if conversation_id in self._ref_counts:
                self._ref_counts[conversation_id] -= 1
                if self._ref_counts[conversation_id] <= 0:
                    self._locks.pop(conversation_id, None)
                    self._ref_counts.pop(conversation_id, None)

    async def execute_atomic_pipeline(
        self, conversation_id: str, evaluation_coro: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Enforces a strict serialize linearizability lifecycle around state extraction.

        Args:
            conversation_id: Unique session hash.
            evaluation_coro: Target coroutine task evaluating actual guardrail parameters.
            *args: Variable length argument list forwarded to the target routine.
            **kwargs: Arbitrary keyword arguments forwarded to the target routine.

        Returns:
            Any: The raw execution return context resolved out of the evaluation task.
        """
        lock = await self._acquire_session_lock(conversation_id)
        async with lock:
            try:
                current_state = await self.backend.fetch_state(conversation_id)
                result, updated_state = await evaluation_coro(current_state, *args, **kwargs)
                await self.backend.save_state(conversation_id, updated_state)
                return result
            finally:
                await self._release_session_lock(conversation_id)

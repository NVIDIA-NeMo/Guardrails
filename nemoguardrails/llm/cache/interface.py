# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Cache interface for NeMo Guardrails caching system.

This module defines the abstract base class for cache implementations
that can be used interchangeably throughout the guardrails system.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class CacheInterface(ABC):
    """
    Abstract base class defining the interface for cache implementations.

    All cache implementations must inherit from this class and implement
    the required methods to ensure compatibility with the caching system.
    """

    @abstractmethod
    def get(self, key: Any, default: Any = None) -> Any:
        """
        Retrieve an item from the cache.

        Args:
            key: The key to look up in the cache.
            default: Value to return if key is not found (default: None).

        Returns:
            The value associated with the key, or default if not found.
        """
        pass

    @abstractmethod
    def put(self, key: Any, value: Any) -> None:
        """
        Store an item in the cache.

        If the cache is at maxsize, this method should evict an item
        according to the cache's eviction policy (e.g., LFU, LRU, etc.).

        Args:
            key: The key to store.
            value: The value to associate with the key.
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """
        Get the current number of items in the cache.

        Returns:
            The number of items currently stored in the cache.
        """
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """
        Check if the cache is empty.

        Returns:
            True if the cache contains no items, False otherwise.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Remove all items from the cache.

        After calling this method, the cache should be empty.
        """
        pass

    def contains(self, key: Any) -> bool:
        """
        Check if a key exists in the cache.

        This is an optional method that can be overridden for efficiency.
        The default implementation uses get() to check existence.

        Args:
            key: The key to check.

        Returns:
            True if the key exists in the cache, False otherwise.
        """
        # Default implementation - can be overridden for efficiency
        sentinel = object()
        return self.get(key, sentinel) is not sentinel

    @property
    @abstractmethod
    def maxsize(self) -> int:
        """
        Get the maximum size of the cache.

        Returns:
            The maximum number of items the cache can hold.
        """
        pass

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics. The format and contents
            may vary by implementation. Common fields include:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - evictions: Number of items evicted
            - hit_rate: Percentage of requests that were hits
            - current_size: Current number of items in cache
            - maxsize: Maximum size of the cache

        The default implementation returns a message indicating that
        statistics tracking is not supported.
        """
        return {
            "message": "Statistics tracking is not supported by this cache implementation"
        }

    def reset_stats(self) -> None:
        """
        Reset cache statistics.

        This is an optional method that cache implementations can override
        if they support statistics tracking. The default implementation does nothing.
        """
        # Default no-op implementation
        pass

    def log_stats_now(self) -> None:
        """
        Force immediate logging of cache statistics.

        This is an optional method that cache implementations can override
        if they support statistics logging. The default implementation does nothing.

        Implementations that support statistics logging should output the
        current cache statistics to their configured logging backend.
        """
        # Default no-op implementation
        pass

    def supports_stats_logging(self) -> bool:
        """
        Check if this cache implementation supports statistics logging.

        Returns:
            True if the cache supports statistics logging, False otherwise.

        The default implementation returns False. Cache implementations
        that support statistics logging should override this to return True
        when logging is enabled.
        """
        return False

    async def get_or_compute(
        self, key: Any, compute_fn: Callable[[], Any], default: Any = None
    ) -> Any:
        """
        Atomically get a value from the cache or compute it if not present.

        This method ensures that the compute function is called at most once
        even in the presence of concurrent requests for the same key.

        Args:
            key: The key to look up
            compute_fn: Async function to compute the value if key is not found
            default: Value to return if compute_fn raises an exception

        Returns:
            The cached value or the computed value

        This is an optional method with a default implementation. Cache
        implementations should override this for better thread-safety guarantees.
        """
        # Default implementation - not thread-safe for computation
        value = self.get(key)
        if value is not None:
            return value

        try:
            computed_value = await compute_fn()
            self.put(key, computed_value)
            return computed_value
        except Exception:
            return default

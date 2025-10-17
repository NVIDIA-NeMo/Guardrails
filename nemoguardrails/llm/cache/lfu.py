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

"""Least Frequently Used (LFU) cache implementation."""

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Optional

from nemoguardrails.llm.cache.interface import CacheInterface

log = logging.getLogger(__name__)


class LFUNode:
    """Node for the LFU cache doubly linked list."""

    def __init__(self, key: Any, value: Any) -> None:
        self.key = key
        self.value = value
        self.freq = 1
        self.prev: Optional["LFUNode"] = None
        self.next: Optional["LFUNode"] = None
        self.created_at = time.time()
        self.accessed_at = self.created_at


class DoublyLinkedList:
    """Doubly linked list to maintain nodes with the same frequency."""

    def __init__(self) -> None:
        # Create dummy head and tail nodes
        self.head = LFUNode(None, None)
        self.tail = LFUNode(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def append(self, node: LFUNode) -> None:
        """Add node to the end of the list (before tail)."""
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node
        self.size += 1

    def pop(self, node: Optional[LFUNode] = None) -> Optional[LFUNode]:
        """Remove and return a node. If no node specified, removes the first node."""
        if self.size == 0:
            return None

        if node is None:
            node = self.head.next

        # Remove node from the list
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1

        return node


class LFUCache(CacheInterface):
    """
    Least Frequently Used (LFU) Cache implementation.

    When the cache reaches maxsize, it evicts the least frequently used item.
    If there are ties in frequency, it evicts the least recently used among them.
    """

    def __init__(
        self,
        maxsize: int,
        track_stats: bool = False,
        stats_logging_interval: Optional[float] = None,
    ) -> None:
        """
        Initialize the LFU cache.

        Args:
            maxsize: Maximum number of items the cache can hold
            track_stats: Enable tracking of cache statistics
            stats_logging_interval: Seconds between periodic stats logging (None disables logging)
        """
        if maxsize < 0:
            raise ValueError("Capacity must be non-negative")

        self._maxsize = maxsize
        self.track_stats = track_stats
        self._lock = threading.RLock()  # Thread-safe access
        self._computing: dict[Any, asyncio.Future] = {}  # Track keys being computed

        self.key_map: dict[Any, LFUNode] = {}  # key -> node mapping
        self.freq_map: dict[int, DoublyLinkedList] = {}  # frequency -> list of nodes
        self.min_freq = 0  # Track minimum frequency for eviction

        # Stats logging configuration
        self.stats_logging_interval = stats_logging_interval
        # Initialize to None to ensure first check doesn't trigger immediately
        self.last_stats_log_time = None

        # Statistics tracking
        if self.track_stats:
            self.stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "puts": 0,
                "updates": 0,
            }

    def _update_node_freq(self, node: LFUNode) -> None:
        """Update the frequency of a node and move it to the appropriate frequency list."""
        old_freq = node.freq
        old_list = self.freq_map[old_freq]

        # Remove node from current frequency list
        old_list.pop(node)

        # Update min_freq if necessary
        if self.min_freq == old_freq and old_list.size == 0:
            self.min_freq += 1
            # Clean up empty frequency lists
            del self.freq_map[old_freq]

        # Increment frequency and add to new list
        node.freq += 1
        new_freq = node.freq
        node.accessed_at = time.time()  # Update access time

        if new_freq not in self.freq_map:
            self.freq_map[new_freq] = DoublyLinkedList()

        self.freq_map[new_freq].append(node)

    def get(self, key: Any, default: Any = None) -> Any:
        """
        Get an item from the cache.

        Args:
            key: The key to look up
            default: Value to return if key is not found

        Returns:
            The value associated with the key, or default if not found
        """
        with self._lock:
            # Check if we should log stats
            self._check_and_log_stats()

            if key not in self.key_map:
                if self.track_stats:
                    self.stats["misses"] += 1
                return default

            node = self.key_map[key]

            if self.track_stats:
                self.stats["hits"] += 1

            self._update_node_freq(node)
            return node.value

    def put(self, key: Any, value: Any) -> None:
        """
        Put an item into the cache.

        Args:
            key: The key to store
            value: The value to associate with the key
        """
        with self._lock:
            # Check if we should log stats
            self._check_and_log_stats()

            if self._maxsize == 0:
                return

            if key in self.key_map:
                # Update existing key
                node = self.key_map[key]
                node.value = value
                node.created_at = time.time()  # Reset creation time on update
                self._update_node_freq(node)
                if self.track_stats:
                    self.stats["updates"] += 1
            else:
                # Add new key
                if len(self.key_map) >= self._maxsize:
                    # Need to evict least frequently used item
                    self._evict_lfu()

                # Create new node and add to cache
                new_node = LFUNode(key, value)
                self.key_map[key] = new_node

                # Add to frequency 1 list
                if 1 not in self.freq_map:
                    self.freq_map[1] = DoublyLinkedList()

                self.freq_map[1].append(new_node)
                self.min_freq = 1

                if self.track_stats:
                    self.stats["puts"] += 1

    def _evict_lfu(self) -> None:
        """Evict the least frequently used item from the cache."""
        if self.min_freq in self.freq_map:
            lfu_list = self.freq_map[self.min_freq]
            node_to_evict = lfu_list.pop()  # Remove least recently used among LFU

            if node_to_evict:
                del self.key_map[node_to_evict.key]

                if self.track_stats:
                    self.stats["evictions"] += 1

                # Clean up empty frequency list
                if lfu_list.size == 0:
                    del self.freq_map[self.min_freq]

    def size(self) -> int:
        """Return the current size of the cache."""
        with self._lock:
            return len(self.key_map)

    def is_empty(self) -> bool:
        """Check if the cache is empty."""
        with self._lock:
            return len(self.key_map) == 0

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            if self.track_stats:
                # Track number of items evicted
                self.stats["evictions"] += len(self.key_map)

            self.key_map.clear()
            self.freq_map.clear()
            self.min_freq = 0

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics (if tracking is enabled)
        """
        with self._lock:
            if not self.track_stats:
                return {"message": "Statistics tracking is disabled"}

            stats = self.stats.copy()
            stats["current_size"] = len(self.key_map)  # Direct access within lock
            stats["maxsize"] = self._maxsize

            # Calculate hit rate
            total_requests = stats["hits"] + stats["misses"]
            stats["hit_rate"] = (
                stats["hits"] / total_requests if total_requests > 0 else 0.0
            )

            return stats

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            if self.track_stats:
                self.stats = {
                    "hits": 0,
                    "misses": 0,
                    "evictions": 0,
                    "puts": 0,
                    "updates": 0,
                }

    def _check_and_log_stats(self) -> None:
        """Check if enough time has passed and log stats if needed."""
        if not self.track_stats or self.stats_logging_interval is None:
            return

        current_time = time.time()

        # Initialize timestamp on first check
        if self.last_stats_log_time is None:
            self.last_stats_log_time = current_time
            return

        if current_time - self.last_stats_log_time >= self.stats_logging_interval:
            self._log_stats()
            self.last_stats_log_time = current_time

    def _log_stats(self) -> None:
        """Log current cache statistics."""
        stats = self.get_stats()

        # Format the log message
        log_msg = (
            f"LFU Cache Statistics - "
            f"Size: {stats['current_size']}/{stats['maxsize']} | "
            f"Hits: {stats['hits']} | "
            f"Misses: {stats['misses']} | "
            f"Hit Rate: {stats['hit_rate']:.2%} | "
            f"Evictions: {stats['evictions']} | "
            f"Puts: {stats['puts']} | "
            f"Updates: {stats['updates']}"
        )

        log.info(log_msg)

    def log_stats_now(self) -> None:
        """Force immediate logging of cache statistics."""
        if self.track_stats:
            self._log_stats()
            self.last_stats_log_time = time.time()

    def supports_stats_logging(self) -> bool:
        """Check if this cache instance supports stats logging."""
        return self.track_stats and self.stats_logging_interval is not None

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
        """
        # First check if the value is already in cache
        future = None
        with self._lock:
            if key in self.key_map:
                node = self.key_map[key]
                if self.track_stats:
                    self.stats["hits"] += 1
                self._update_node_freq(node)
                return node.value

            # Check if this key is already being computed
            if key in self._computing:
                future = self._computing[key]

        # If the key is being computed, wait for it outside the lock
        if future is not None:
            try:
                return await future
            except Exception:
                return default

        # Create a future for this computation
        future = asyncio.Future()
        with self._lock:
            # Double-check the cache and computing dict
            if key in self.key_map:
                node = self.key_map[key]
                if self.track_stats:
                    self.stats["hits"] += 1
                self._update_node_freq(node)
                return node.value

            if key in self._computing:
                # Another thread started computing while we were waiting
                future = self._computing[key]
            else:
                # We'll be the ones computing
                self._computing[key] = future

        # If another thread is computing, wait for it
        if not future.done() and self._computing.get(key) is not future:
            try:
                return await self._computing[key]
            except Exception:
                return default

        # We're responsible for computing the value
        try:
            computed_value = await compute_fn()

            # Store the computed value in cache
            with self._lock:
                # Remove from computing dict
                self._computing.pop(key, None)

                # Check one more time if someone else added it
                if key in self.key_map:
                    node = self.key_map[key]
                    if self.track_stats:
                        self.stats["hits"] += 1
                    self._update_node_freq(node)
                    future.set_result(node.value)
                    return node.value

                # Now add to cache using internal logic
                if self._maxsize == 0:
                    future.set_result(computed_value)
                    return computed_value

                # Add new key
                if len(self.key_map) >= self._maxsize:
                    self._evict_lfu()

                # Create new node and add to cache
                new_node = LFUNode(key, computed_value)
                self.key_map[key] = new_node

                # Add to frequency 1 list
                if 1 not in self.freq_map:
                    self.freq_map[1] = DoublyLinkedList()

                self.freq_map[1].append(new_node)
                self.min_freq = 1

                if self.track_stats:
                    self.stats["puts"] += 1

                # Set the result in the future
                future.set_result(computed_value)
                return computed_value

        except Exception as e:
            with self._lock:
                self._computing.pop(key, None)
            future.set_exception(e)
            return default

    def contains(self, key: Any) -> bool:
        """
        Check if a key exists in the cache without updating its frequency.

        This is more efficient than the default implementation which uses get()
        and has the side effect of updating frequency counts.

        Args:
            key: The key to check

        Returns:
            True if the key exists in the cache, False otherwise
        """
        with self._lock:
            return key in self.key_map

    @property
    def maxsize(self) -> int:
        """Get the maximum size of the cache."""
        return self._maxsize

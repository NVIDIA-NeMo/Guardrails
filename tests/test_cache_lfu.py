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
Comprehensive test suite for LFU Cache implementation.

Tests all functionality including basic operations, eviction policies,
maxsize management, and edge cases.
"""

import asyncio
import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock, patch

from nemoguardrails.llm.cache.lfu import LFUCache


class TestLFUCache(unittest.TestCase):
    """Test cases for LFU Cache implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = LFUCache(3)

    def test_initialization(self):
        """Test cache initialization with various capacities."""
        # Normal maxsize
        cache = LFUCache(5)
        self.assertEqual(cache.size(), 0)
        self.assertTrue(cache.is_empty())

        # Zero maxsize
        cache_zero = LFUCache(0)
        self.assertEqual(cache_zero.size(), 0)

        # Negative maxsize should raise error
        with self.assertRaises(ValueError):
            LFUCache(-1)

    def test_basic_put_get(self):
        """Test basic put and get operations."""
        # Put and get single item
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertEqual(self.cache.size(), 1)

        # Put and get multiple items
        self.cache.put("key2", "value2")
        self.cache.put("key3", "value3")

        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertEqual(self.cache.get("key2"), "value2")
        self.assertEqual(self.cache.get("key3"), "value3")
        self.assertEqual(self.cache.size(), 3)

    def test_get_nonexistent_key(self):
        """Test getting non-existent keys."""
        # Default behavior (returns None)
        self.assertIsNone(self.cache.get("nonexistent"))

        # With custom default
        self.assertEqual(self.cache.get("nonexistent", "default"), "default")

        # After adding some items
        self.cache.put("key1", "value1")
        self.assertIsNone(self.cache.get("key2"))
        self.assertEqual(self.cache.get("key2", 42), 42)

    def test_update_existing_key(self):
        """Test updating values for existing keys."""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")

        # Update existing key
        self.cache.put("key1", "new_value1")
        self.assertEqual(self.cache.get("key1"), "new_value1")

        # Size should not change
        self.assertEqual(self.cache.size(), 2)

    def test_lfu_eviction_basic(self):
        """Test basic LFU eviction when cache is full."""
        # Fill cache
        self.cache.put("a", 1)
        self.cache.put("b", 2)
        self.cache.put("c", 3)

        # Access 'a' and 'b' to increase their frequency
        self.cache.get("a")  # freq: 2
        self.cache.get("b")  # freq: 2
        # 'c' remains at freq: 1

        # Add new item - should evict 'c' (lowest frequency)
        self.cache.put("d", 4)

        self.assertEqual(self.cache.get("a"), 1)
        self.assertEqual(self.cache.get("b"), 2)
        self.assertEqual(self.cache.get("d"), 4)
        self.assertIsNone(self.cache.get("c"))  # Should be evicted

    def test_lfu_with_lru_tiebreaker(self):
        """Test LRU eviction among items with same frequency."""
        # Fill cache - all items have frequency 1
        self.cache.put("a", 1)
        self.cache.put("b", 2)
        self.cache.put("c", 3)

        # Add new item - should evict 'a' (least recently used among freq 1)
        self.cache.put("d", 4)

        self.assertIsNone(self.cache.get("a"))  # Should be evicted
        self.assertEqual(self.cache.get("b"), 2)
        self.assertEqual(self.cache.get("c"), 3)
        self.assertEqual(self.cache.get("d"), 4)

    def test_complex_eviction_scenario(self):
        """Test complex eviction scenario with multiple frequency levels."""
        # Create a new cache for this test
        cache = LFUCache(4)

        # Add items and create different frequency levels
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)

        # Create frequency pattern:
        # a: freq 3 (accessed 2 more times)
        # b: freq 2 (accessed 1 more time)
        # c: freq 2 (accessed 1 more time)
        # d: freq 1 (not accessed)

        cache.get("a")
        cache.get("a")
        cache.get("b")
        cache.get("c")

        # Add new item - should evict 'd' (freq 1)
        cache.put("e", 5)
        self.assertIsNone(cache.get("d"))

        # Add another item - should evict one of the least frequently used
        cache.put("f", 6)

        # After eviction, we should have:
        # - 'a' (freq 3) - definitely kept
        # - 'b' (freq 2) and 'c' (freq 2) - higher frequency, both kept
        # - 'f' (freq 1) - just added
        # - 'e' (freq 1) was evicted as it was least recently used among freq 1 items

        # Check that we're at maxsize
        self.assertEqual(cache.size(), 4)

        # 'a' should definitely still be there (highest frequency)
        self.assertEqual(cache.get("a"), 1)

        # 'b' and 'c' should both be there (freq 2)
        self.assertEqual(cache.get("b"), 2)
        self.assertEqual(cache.get("c"), 3)

        # 'f' should be there (just added)
        self.assertEqual(cache.get("f"), 6)

        # 'e' should have been evicted (freq 1, LRU among freq 1 items)
        self.assertIsNone(cache.get("e"))

    def test_zero_maxsize_cache(self):
        """Test cache with zero maxsize."""
        cache = LFUCache(0)

        # Put should not store anything
        cache.put("key", "value")
        self.assertEqual(cache.size(), 0)
        self.assertIsNone(cache.get("key"))

        # Multiple puts
        for i in range(10):
            cache.put(f"key{i}", f"value{i}")

        self.assertEqual(cache.size(), 0)
        self.assertTrue(cache.is_empty())

    def test_clear_method(self):
        """Test clearing the cache."""
        # Add items
        self.cache.put("a", 1)
        self.cache.put("b", 2)
        self.cache.put("c", 3)

        # Verify items exist
        self.assertEqual(self.cache.size(), 3)
        self.assertFalse(self.cache.is_empty())

        # Clear cache
        self.cache.clear()

        # Verify cache is empty
        self.assertEqual(self.cache.size(), 0)
        self.assertTrue(self.cache.is_empty())

        # Verify items are gone
        self.assertIsNone(self.cache.get("a"))
        self.assertIsNone(self.cache.get("b"))
        self.assertIsNone(self.cache.get("c"))

        # Can still use cache after clear
        self.cache.put("new_key", "new_value")
        self.assertEqual(self.cache.get("new_key"), "new_value")

    def test_various_data_types(self):
        """Test cache with various data types as keys and values."""
        # Integer keys
        self.cache.put(1, "one")
        self.cache.put(2, "two")
        self.assertEqual(self.cache.get(1), "one")
        self.assertEqual(self.cache.get(2), "two")

        # Tuple keys
        self.cache.put((1, 2), "tuple_value")
        self.assertEqual(self.cache.get((1, 2)), "tuple_value")

        # Clear for more tests
        self.cache.clear()

        # Complex values
        self.cache.put("list", [1, 2, 3])
        self.cache.put("dict", {"a": 1, "b": 2})
        self.cache.put("set", {1, 2, 3})

        self.assertEqual(self.cache.get("list"), [1, 2, 3])
        self.assertEqual(self.cache.get("dict"), {"a": 1, "b": 2})
        self.assertEqual(self.cache.get("set"), {1, 2, 3})

    def test_none_values(self):
        """Test storing None as a value."""
        self.cache.put("key", None)
        # get should return None for the value, not the default
        self.assertIsNone(self.cache.get("key"))
        self.assertEqual(self.cache.get("key", "default"), None)

        # Verify key exists
        self.assertEqual(self.cache.size(), 1)

    def test_size_and_maxsize(self):
        """Test size tracking and maxsize limits."""
        # Start empty
        self.assertEqual(self.cache.size(), 0)

        # Add items up to maxsize
        for i in range(3):
            self.cache.put(f"key{i}", f"value{i}")
            self.assertEqual(self.cache.size(), i + 1)

        # Add more items - size should stay at maxsize
        for i in range(3, 10):
            self.cache.put(f"key{i}", f"value{i}")
            self.assertEqual(self.cache.size(), 3)

    def test_is_empty(self):
        """Test is_empty method in various states."""
        # Initially empty
        self.assertTrue(self.cache.is_empty())

        # After adding item
        self.cache.put("key", "value")
        self.assertFalse(self.cache.is_empty())

        # After clearing
        self.cache.clear()
        self.assertTrue(self.cache.is_empty())

    def test_repeated_puts_same_key(self):
        """Test repeated puts with the same key maintain size=1 and update frequency."""
        self.cache.put("key", "value1")
        self.assertEqual(self.cache.size(), 1)

        # Track initial state
        initial_stats = self.cache.get_stats() if self.cache.track_stats else None

        # Update same key multiple times
        for i in range(10):
            self.cache.put("key", f"value{i}")
            self.assertEqual(self.cache.size(), 1)

        # Final value should be the last one
        self.assertEqual(self.cache.get("key"), "value9")

        # Verify stats if tracking enabled
        if self.cache.track_stats:
            final_stats = self.cache.get_stats()
            # Should have 10 updates (after initial put)
            self.assertEqual(final_stats["updates"], 10)

    def test_access_pattern_preserves_frequently_used(self):
        """Test that frequently accessed items are preserved during evictions."""
        # Create specific access pattern
        cache = LFUCache(3)

        # Add three items
        cache.put("rarely_used", 1)
        cache.put("sometimes_used", 2)
        cache.put("frequently_used", 3)

        # Create access pattern
        # frequently_used: access 10 times
        for _ in range(10):
            cache.get("frequently_used")

        # sometimes_used: access 3 times
        for _ in range(3):
            cache.get("sometimes_used")

        # rarely_used: no additional access (freq = 1)

        # Add new items to trigger evictions
        cache.put("new1", 4)  # Should evict rarely_used
        cache.put("new2", 5)  # Should evict new1 (freq = 1)

        # frequently_used and sometimes_used should still be there
        self.assertEqual(cache.get("frequently_used"), 3)
        self.assertEqual(cache.get("sometimes_used"), 2)

        # rarely_used and new1 should be evicted
        self.assertIsNone(cache.get("rarely_used"))
        self.assertIsNone(cache.get("new1"))

        # new2 should be there
        self.assertEqual(cache.get("new2"), 5)


class TestLFUCacheInterface(unittest.TestCase):
    """Test that LFUCache properly implements CacheInterface."""

    def test_interface_methods_exist(self):
        """Verify all interface methods are implemented."""
        cache = LFUCache(5)

        # Check all required methods exist and are callable
        self.assertTrue(callable(getattr(cache, "get", None)))
        self.assertTrue(callable(getattr(cache, "put", None)))
        self.assertTrue(callable(getattr(cache, "size", None)))
        self.assertTrue(callable(getattr(cache, "is_empty", None)))
        self.assertTrue(callable(getattr(cache, "clear", None)))

        # Check property
        self.assertEqual(cache.maxsize, 5)


class TestLFUCacheStatsLogging(unittest.TestCase):
    """Test cases for LFU Cache statistics logging functionality."""

    def test_stats_logging_disabled_by_default(self):
        """Test that stats logging is disabled when not configured."""
        cache = LFUCache(5, track_stats=True)
        self.assertFalse(cache.supports_stats_logging())

    def test_stats_logging_requires_tracking(self):
        """Test that stats logging requires stats tracking to be enabled."""
        # Logging without tracking
        cache = LFUCache(5, track_stats=False, stats_logging_interval=1.0)
        self.assertFalse(cache.supports_stats_logging())

        # Both enabled
        cache = LFUCache(5, track_stats=True, stats_logging_interval=1.0)
        self.assertTrue(cache.supports_stats_logging())

    def test_log_stats_now(self):
        """Test immediate stats logging."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(5, track_stats=True, stats_logging_interval=60.0)

        # Add some data
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.get("key1")
        cache.get("nonexistent")

        with patch.object(
            logging.getLogger("nemoguardrails.llm.cache.lfu"), "info"
        ) as mock_log:
            cache.log_stats_now()

            # Verify log was called
            self.assertEqual(mock_log.call_count, 1)
            log_message = mock_log.call_args[0][0]

            # Check log format
            self.assertIn("LFU Cache Statistics", log_message)
            self.assertIn("Size: 2/5", log_message)
            self.assertIn("Hits: 1", log_message)
            self.assertIn("Misses: 1", log_message)
            self.assertIn("Hit Rate: 50.00%", log_message)
            self.assertIn("Evictions: 0", log_message)
            self.assertIn("Puts: 2", log_message)
            self.assertIn("Updates: 0", log_message)

    def test_periodic_stats_logging(self):
        """Test automatic periodic stats logging."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(5, track_stats=True, stats_logging_interval=0.5)

        # Add some data
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        with patch.object(
            logging.getLogger("nemoguardrails.llm.cache.lfu"), "info"
        ) as mock_log:
            # Initial operations shouldn't trigger logging
            cache.get("key1")
            self.assertEqual(mock_log.call_count, 0)

            # Wait for interval to pass
            time.sleep(0.6)

            # Next operation should trigger logging
            cache.get("key1")
            self.assertEqual(mock_log.call_count, 1)

            # Another operation without waiting shouldn't trigger
            cache.get("key2")
            self.assertEqual(mock_log.call_count, 1)

            # Wait again
            time.sleep(0.6)
            cache.put("key3", "value3")
            self.assertEqual(mock_log.call_count, 2)

    def test_stats_logging_with_empty_cache(self):
        """Test stats logging with empty cache."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(5, track_stats=True, stats_logging_interval=0.1)

        # Generate a miss first
        cache.get("nonexistent")

        # Wait for interval to pass
        time.sleep(0.2)

        with patch.object(
            logging.getLogger("nemoguardrails.llm.cache.lfu"), "info"
        ) as mock_log:
            # This will trigger stats logging with the previous miss already counted
            cache.get("another_nonexistent")  # Trigger check

            self.assertEqual(mock_log.call_count, 1)
            log_message = mock_log.call_args[0][0]

            self.assertIn("Size: 0/5", log_message)
            self.assertIn("Hits: 0", log_message)
            self.assertIn("Misses: 1", log_message)  # The first miss is logged
            self.assertIn("Hit Rate: 0.00%", log_message)

    def test_stats_logging_with_full_cache(self):
        """Test stats logging when cache is at maxsize."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(3, track_stats=True, stats_logging_interval=0.1)

        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Cause eviction
        cache.put("key4", "value4")

        with patch.object(
            logging.getLogger("nemoguardrails.llm.cache.lfu"), "info"
        ) as mock_log:
            time.sleep(0.2)
            cache.get("key4")  # Trigger check

            log_message = mock_log.call_args[0][0]
            self.assertIn("Size: 3/3", log_message)
            self.assertIn("Evictions: 1", log_message)
            self.assertIn("Puts: 4", log_message)

    def test_stats_logging_high_hit_rate(self):
        """Test stats logging with high hit rate."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(5, track_stats=True, stats_logging_interval=0.1)

        cache.put("key1", "value1")

        # Many hits
        for _ in range(99):
            cache.get("key1")

        # One miss
        cache.get("nonexistent")

        with patch.object(
            logging.getLogger("nemoguardrails.llm.cache.lfu"), "info"
        ) as mock_log:
            cache.log_stats_now()

            log_message = mock_log.call_args[0][0]
            self.assertIn("Hit Rate: 99.00%", log_message)
            self.assertIn("Hits: 99", log_message)
            self.assertIn("Misses: 1", log_message)

    def test_stats_logging_without_tracking(self):
        """Test that log_stats_now does nothing when tracking is disabled."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(5, track_stats=False)

        cache.put("key1", "value1")
        cache.get("key1")

        with patch.object(
            logging.getLogger("nemoguardrails.llm.cache.lfu"), "info"
        ) as mock_log:
            cache.log_stats_now()

            # Should not log anything
            self.assertEqual(mock_log.call_count, 0)

    def test_stats_logging_interval_timing(self):
        """Test that stats logging respects the interval timing."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(5, track_stats=True, stats_logging_interval=1.0)

        with patch.object(
            logging.getLogger("nemoguardrails.llm.cache.lfu"), "info"
        ) as mock_log:
            # Multiple operations within interval
            for i in range(10):
                cache.put(f"key{i}", f"value{i}")
                cache.get(f"key{i}")
                time.sleep(0.05)  # Total time < 1.0

            # Should not have logged yet
            self.assertEqual(mock_log.call_count, 0)

            # Wait for interval to pass
            time.sleep(0.6)
            cache.get("key1")  # Trigger check

            # Now should have logged once
            self.assertEqual(mock_log.call_count, 1)

    def test_stats_logging_with_updates(self):
        """Test stats logging includes update counts."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(5, track_stats=True, stats_logging_interval=0.1)

        cache.put("key1", "value1")
        cache.put("key1", "updated_value1")  # Update
        cache.put("key1", "updated_again")  # Another update

        with patch.object(
            logging.getLogger("nemoguardrails.llm.cache.lfu"), "info"
        ) as mock_log:
            cache.log_stats_now()

            log_message = mock_log.call_args[0][0]
            self.assertIn("Updates: 2", log_message)
            self.assertIn("Puts: 1", log_message)

    def test_stats_log_format_percentages(self):
        """Test that percentages in stats log are formatted correctly."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(5, track_stats=True, stats_logging_interval=0.1)

        # Test various hit rates
        test_cases = [
            (0, 0, "0.00%"),  # No requests
            (1, 0, "100.00%"),  # All hits
            (0, 1, "0.00%"),  # All misses
            (1, 1, "50.00%"),  # 50/50
            (2, 1, "66.67%"),  # 2/3
            (99, 1, "99.00%"),  # High hit rate
        ]

        for hits, misses, expected_rate in test_cases:
            cache.reset_stats()

            # Generate hits
            if hits > 0:
                cache.put("hit_key", "value")
                for _ in range(hits):
                    cache.get("hit_key")

            # Generate misses
            for i in range(misses):
                cache.get(f"miss_key_{i}")

            with patch.object(
                logging.getLogger("nemoguardrails.llm.cache.lfu"), "info"
            ) as mock_log:
                cache.log_stats_now()

                if hits > 0 or misses > 0:
                    log_message = mock_log.call_args[0][0]
                    self.assertIn(f"Hit Rate: {expected_rate}", log_message)


class TestContentSafetyCacheStatsConfig(unittest.TestCase):
    """Test cache stats configuration in content safety context."""

    def test_cache_config_with_stats_disabled(self):
        """Test cache configuration with stats disabled."""
        from nemoguardrails.rails.llm.config import CacheStatsConfig, ModelCacheConfig

        cache_config = ModelCacheConfig(
            enabled=True, maxsize=1000, stats=CacheStatsConfig(enabled=False)
        )

        cache = LFUCache(
            maxsize=cache_config.maxsize,
            track_stats=cache_config.stats.enabled,
            stats_logging_interval=None,
        )

        self.assertIsNotNone(cache)
        self.assertFalse(cache.track_stats)
        self.assertFalse(cache.supports_stats_logging())

    def test_cache_config_with_stats_tracking_only(self):
        """Test cache configuration with stats tracking but no logging."""
        from nemoguardrails.rails.llm.config import CacheStatsConfig, ModelCacheConfig

        cache_config = ModelCacheConfig(
            enabled=True,
            maxsize=1000,
            stats=CacheStatsConfig(enabled=True, log_interval=None),
        )

        cache = LFUCache(
            maxsize=cache_config.maxsize,
            track_stats=cache_config.stats.enabled,
            stats_logging_interval=cache_config.stats.log_interval,
        )

        self.assertIsNotNone(cache)
        self.assertTrue(cache.track_stats)
        self.assertFalse(cache.supports_stats_logging())
        self.assertIsNone(cache.stats_logging_interval)

    def test_cache_config_with_stats_logging(self):
        """Test cache configuration with stats tracking and logging."""
        from nemoguardrails.rails.llm.config import CacheStatsConfig, ModelCacheConfig

        cache_config = ModelCacheConfig(
            enabled=True,
            maxsize=1000,
            stats=CacheStatsConfig(enabled=True, log_interval=60.0),
        )

        cache = LFUCache(
            maxsize=cache_config.maxsize,
            track_stats=cache_config.stats.enabled,
            stats_logging_interval=cache_config.stats.log_interval,
        )

        self.assertIsNotNone(cache)
        self.assertTrue(cache.track_stats)
        self.assertTrue(cache.supports_stats_logging())
        self.assertEqual(cache.stats_logging_interval, 60.0)

    def test_cache_config_default_stats(self):
        """Test cache configuration with default stats settings."""
        from nemoguardrails.rails.llm.config import ModelCacheConfig

        cache_config = ModelCacheConfig(enabled=True)

        cache = LFUCache(
            maxsize=cache_config.maxsize,
            track_stats=cache_config.stats.enabled,
            stats_logging_interval=None,
        )

        self.assertIsNotNone(cache)
        self.assertFalse(cache.track_stats)  # Default is disabled
        self.assertFalse(cache.supports_stats_logging())

    def test_cache_config_from_dict(self):
        """Test cache configuration creation from dictionary."""
        from nemoguardrails.rails.llm.config import ModelCacheConfig

        config_dict = {
            "enabled": True,
            "maxsize": 5000,
            "stats": {"enabled": True, "log_interval": 120.0},
        }

        cache_config = ModelCacheConfig(**config_dict)
        self.assertTrue(cache_config.enabled)
        self.assertEqual(cache_config.maxsize, 5000)
        self.assertTrue(cache_config.stats.enabled)
        self.assertEqual(cache_config.stats.log_interval, 120.0)

    def test_cache_config_stats_validation(self):
        """Test cache configuration validation for stats settings."""
        from nemoguardrails.rails.llm.config import CacheStatsConfig

        # Valid configurations
        stats1 = CacheStatsConfig(enabled=True, log_interval=60.0)
        self.assertTrue(stats1.enabled)
        self.assertEqual(stats1.log_interval, 60.0)

        stats2 = CacheStatsConfig(enabled=True, log_interval=None)
        self.assertTrue(stats2.enabled)
        self.assertIsNone(stats2.log_interval)

        stats3 = CacheStatsConfig(enabled=False, log_interval=60.0)
        self.assertFalse(stats3.enabled)
        self.assertEqual(stats3.log_interval, 60.0)


class TestLFUCacheThreadSafety(unittest.TestCase):
    """Test thread safety of LFU Cache implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = LFUCache(100, track_stats=True)

    def test_concurrent_reads_writes(self):
        """Test that concurrent reads and writes don't corrupt the cache."""
        num_threads = 10
        operations_per_thread = 100
        # Use a larger cache to avoid evictions during the test
        large_cache = LFUCache(2000, track_stats=True)
        errors = []

        def worker(thread_id):
            """Worker function that performs cache operations."""
            for i in range(operations_per_thread):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"

                # Put operation
                large_cache.put(key, value)

                # Get operation - should always succeed with large cache
                retrieved = large_cache.get(key)

                # Verify data integrity
                if retrieved != value:
                    errors.append(
                        f"Data corruption for {key}: expected {value}, got {retrieved}"
                    )

                # Access some shared keys
                shared_key = f"shared_key_{i % 10}"
                large_cache.put(shared_key, f"shared_value_{thread_id}_{i}")
                large_cache.get(shared_key)

        # Run threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in futures:
                future.result()  # Wait for completion and raise any exceptions

        # Check for any errors
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors[:5]}...")

        # Verify cache is still functional
        test_key = "test_after_concurrent"
        test_value = "test_value"
        large_cache.put(test_key, test_value)
        self.assertEqual(large_cache.get(test_key), test_value)

        # Check statistics are reasonable
        stats = large_cache.get_stats()
        self.assertGreater(stats["hits"], 0)
        self.assertGreater(stats["puts"], 0)

    def test_concurrent_evictions(self):
        """Test that concurrent operations during evictions don't corrupt the cache."""
        # Use a small cache to trigger frequent evictions
        small_cache = LFUCache(10)
        num_threads = 5
        operations_per_thread = 50

        def worker(thread_id):
            """Worker that adds many items to trigger evictions."""
            for i in range(operations_per_thread):
                key = f"t{thread_id}_k{i}"
                value = f"t{thread_id}_v{i}"
                small_cache.put(key, value)

                # Try to get recently added items
                if i > 0:
                    prev_key = f"t{thread_id}_k{i-1}"
                    small_cache.get(prev_key)  # May or may not exist

        # Run threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in futures:
                future.result()

        # Cache should still be at maxsize
        self.assertEqual(small_cache.size(), 10)

    def test_concurrent_clear_operations(self):
        """Test concurrent clear operations with other operations."""

        def writer():
            """Continuously write to cache."""
            for i in range(100):
                self.cache.put(f"key_{i}", f"value_{i}")
                time.sleep(0.001)  # Small delay

        def clearer():
            """Periodically clear the cache."""
            for _ in range(5):
                time.sleep(0.01)
                self.cache.clear()

        def reader():
            """Continuously read from cache."""
            for i in range(100):
                self.cache.get(f"key_{i}")
                time.sleep(0.001)

        # Run operations concurrently
        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=clearer),
            threading.Thread(target=reader),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Cache should still be functional
        self.cache.put("final_key", "final_value")
        self.assertEqual(self.cache.get("final_key"), "final_value")

    def test_concurrent_stats_operations(self):
        """Test that concurrent operations don't corrupt statistics."""

        def worker(thread_id):
            """Worker that performs operations and checks stats."""
            for i in range(50):
                key = f"stats_key_{thread_id}_{i}"
                self.cache.put(key, i)
                self.cache.get(key)  # Hit
                self.cache.get(f"nonexistent_{thread_id}_{i}")  # Miss

                # Periodically check stats
                if i % 10 == 0:
                    stats = self.cache.get_stats()
                    # Just verify we can get stats without error
                    self.assertIsInstance(stats, dict)
                    self.assertIn("hits", stats)
                    self.assertIn("misses", stats)

        # Run threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(5)]
            for future in futures:
                future.result()

        # Final stats check
        final_stats = self.cache.get_stats()
        self.assertGreater(final_stats["hits"], 0)
        self.assertGreater(final_stats["misses"], 0)
        self.assertGreater(final_stats["puts"], 0)

    def test_get_or_compute_thread_safety(self):
        """Test thread safety of get_or_compute method."""
        compute_count = threading.local()
        compute_count.value = 0
        total_computes = []
        lock = threading.Lock()

        async def expensive_compute():
            """Simulate expensive computation that should only run once."""
            # Track how many times this is called
            if not hasattr(compute_count, "value"):
                compute_count.value = 0
            compute_count.value += 1

            with lock:
                total_computes.append(1)

            # Simulate expensive operation
            await asyncio.sleep(0.1)
            return f"computed_value_{len(total_computes)}"

        async def worker(thread_id):
            """Worker that tries to get or compute the same key."""
            result = await self.cache.get_or_compute(
                "shared_compute_key", expensive_compute, default="default"
            )
            return result

        async def run_test():
            """Run the async test."""
            # Run multiple workers concurrently
            tasks = [worker(i) for i in range(10)]
            results = await asyncio.gather(*tasks)

            # All should get the same value
            self.assertTrue(
                all(r == results[0] for r in results),
                f"All threads should get same value, got: {results}",
            )

            # Compute should have been called only once
            self.assertEqual(
                len(total_computes),
                1,
                f"Compute should be called once, called {len(total_computes)} times",
            )

            return results[0]

        # Run the async test
        result = asyncio.run(run_test())
        self.assertEqual(result, "computed_value_1")

    def test_get_or_compute_exception_handling(self):
        """Test get_or_compute handles exceptions properly.

        NOTE: This test will produce "ValueError: Computation failed" messages in the test output.
        These are EXPECTED and NORMAL - the test intentionally triggers failures to verify
        that the cache handles exceptions correctly. Each of the 5 workers will generate one
        error message, but all workers should receive the fallback value successfully.
        """
        # Optional: Uncomment to see a message before the expected errors
        # print("\n[test_get_or_compute_exception_handling] Note: The following 5 'ValueError: Computation failed' messages are expected...")

        call_count = [0]

        async def failing_compute():
            """Compute function that fails."""
            call_count[0] += 1
            raise ValueError("Computation failed")

        async def worker():
            """Worker that tries to compute."""
            result = await self.cache.get_or_compute(
                "failing_key", failing_compute, default="fallback"
            )
            return result

        async def run_test():
            """Run the async test."""
            # Multiple workers should all get the default value
            tasks = [worker() for _ in range(5)]
            results = await asyncio.gather(*tasks)

            # All should get the default value
            self.assertTrue(all(r == "fallback" for r in results))

            # The compute function might be called multiple times
            # since failed computations aren't cached
            self.assertGreaterEqual(call_count[0], 1)

        asyncio.run(run_test())

    def test_thread_safe_size_operations(self):
        """Test that size-related operations are thread-safe."""
        results = []

        def worker(thread_id):
            """Worker that checks size consistency."""
            for i in range(100):
                # Add item
                self.cache.put(f"size_key_{thread_id}_{i}", i)

                # Check size
                size = self.cache.size()
                is_empty = self.cache.is_empty()

                # Size should never be negative or exceed maxsize
                if size < 0 or size > 100:
                    results.append(f"Invalid size: {size}")

                # is_empty should match size
                if (size == 0) != is_empty:
                    results.append(f"Size {size} but is_empty={is_empty}")

        # Run workers
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            for future in futures:
                future.result()

        # Check for any inconsistencies
        self.assertEqual(len(results), 0, f"Inconsistencies found: {results}")

    def test_concurrent_contains_operations(self):
        """Test thread safety of contains method."""
        # Use a larger cache to avoid evictions during the test
        # Need maxsize for: 50 existing + (5 threads × 100 new keys) = 550+
        large_cache = LFUCache(1000, track_stats=True)

        # Pre-populate cache
        for i in range(50):
            large_cache.put(f"existing_key_{i}", f"value_{i}")

        results = []
        eviction_warnings = []

        def worker(thread_id):
            """Worker that checks contains and manipulates cache."""
            for i in range(100):
                # Check existing keys
                key = f"existing_key_{i % 50}"
                if not large_cache.contains(key):
                    results.append(f"Thread {thread_id}: Missing key {key}")

                # Add new keys
                new_key = f"new_key_{thread_id}_{i}"
                large_cache.put(new_key, f"value_{thread_id}_{i}")

                # Check new key immediately
                if not large_cache.contains(new_key):
                    # This could happen if cache is full and eviction occurred
                    # Track it separately as it's not a thread safety issue
                    eviction_warnings.append(
                        f"Thread {thread_id}: Key {new_key} possibly evicted"
                    )

                # Check non-existent keys
                if large_cache.contains(f"non_existent_{thread_id}_{i}"):
                    results.append(f"Thread {thread_id}: Found non-existent key")

        # Run workers
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(5)]
            for future in futures:
                future.result()

        # Check for any errors (not counting eviction warnings)
        self.assertEqual(len(results), 0, f"Errors found: {results}")

        # Eviction warnings should be minimal with large cache
        if eviction_warnings:
            print(f"Note: {len(eviction_warnings)} keys were evicted during test")

    def test_concurrent_reset_stats(self):
        """Test thread safety of reset_stats operations."""
        errors = []

        def worker(thread_id):
            """Worker that performs operations and resets stats."""
            for i in range(50):
                # Perform operations
                self.cache.put(f"key_{thread_id}_{i}", i)
                self.cache.get(f"key_{thread_id}_{i}")
                self.cache.get("non_existent")

                # Periodically reset stats
                if i % 10 == 0:
                    self.cache.reset_stats()

                # Check stats integrity
                stats = self.cache.get_stats()
                if any(v < 0 for v in stats.values() if isinstance(v, (int, float))):
                    errors.append(f"Thread {thread_id}: Negative stat value: {stats}")

        # Run workers
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(5)]
            for future in futures:
                future.result()

        # Verify no errors
        self.assertEqual(len(errors), 0, f"Stats errors: {errors[:5]}")

    def test_get_or_compute_concurrent_different_keys(self):
        """Test get_or_compute with different keys being computed concurrently."""
        compute_counts = {}
        lock = threading.Lock()

        async def compute_for_key(key):
            """Compute function that tracks calls per key."""
            with lock:
                compute_counts[key] = compute_counts.get(key, 0) + 1
            await asyncio.sleep(0.05)  # Simulate work
            return f"value_for_{key}"

        async def worker(thread_id, key_id):
            """Worker that computes values for specific keys."""
            key = f"key_{key_id}"
            result = await self.cache.get_or_compute(
                key, lambda: compute_for_key(key), default="error"
            )
            return key, result

        async def run_test():
            """Run concurrent computations for different keys."""
            # Create tasks for multiple keys, with some overlap
            tasks = []
            for key_id in range(5):
                for thread_id in range(3):  # 3 threads per key
                    tasks.append(worker(thread_id, key_id))

            results = await asyncio.gather(*tasks)

            # Verify each key was computed exactly once
            for key_id in range(5):
                key = f"key_{key_id}"
                self.assertEqual(
                    compute_counts.get(key, 0),
                    1,
                    f"{key} should be computed exactly once",
                )

            # Verify all threads got correct values
            for key, value in results:
                expected = f"value_for_{key}"
                self.assertEqual(value, expected)

        asyncio.run(run_test())

    def test_concurrent_operations_with_evictions(self):
        """Test thread safety when cache is at maxsize and evictions occur."""
        # Small cache to force evictions
        small_cache = LFUCache(50, track_stats=True)
        data_integrity_errors = []

        def worker(thread_id):
            """Worker that handles potential evictions gracefully."""
            for i in range(100):
                key = f"t{thread_id}_k{i}"
                value = f"t{thread_id}_v{i}"

                # Put value
                small_cache.put(key, value)

                # Immediately access to increase frequency
                retrieved = small_cache.get(key)

                # Value might be None if evicted immediately (unlikely but possible)
                if retrieved is not None and retrieved != value:
                    # This would indicate actual data corruption
                    data_integrity_errors.append(
                        f"Wrong value for {key}: expected {value}, got {retrieved}"
                    )

                # Also work with some high-frequency keys (access multiple times)
                high_freq_key = f"high_freq_{thread_id % 5}"
                for _ in range(3):  # Access 3 times to increase frequency
                    small_cache.put(high_freq_key, f"high_freq_value_{thread_id}")
                    small_cache.get(high_freq_key)

        # Run workers
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            for future in futures:
                future.result()

        # Should have no data integrity errors (wrong values)
        self.assertEqual(
            len(data_integrity_errors),
            0,
            f"Data integrity errors: {data_integrity_errors}",
        )

        # Cache should be at maxsize
        self.assertEqual(small_cache.size(), 50)

        # Stats should show many evictions
        stats = small_cache.get_stats()
        self.assertGreater(stats["evictions"], 0)
        self.assertGreater(stats["puts"], 0)


if __name__ == "__main__":
    unittest.main()

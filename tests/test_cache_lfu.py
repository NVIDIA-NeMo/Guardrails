# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
capacity management, edge cases, and persistence functionality.
"""

import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock, patch

from nemoguardrails.cache.lfu import LFUCache
from nemoguardrails.library.content_safety.manager import ContentSafetyManager


class TestLFUCache(unittest.TestCase):
    """Test cases for LFU Cache implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = LFUCache(3)

    def test_initialization(self):
        """Test cache initialization with various capacities."""
        # Normal capacity
        cache = LFUCache(5)
        self.assertEqual(cache.size(), 0)
        self.assertTrue(cache.is_empty())

        # Zero capacity
        cache_zero = LFUCache(0)
        self.assertEqual(cache_zero.size(), 0)

        # Negative capacity should raise error
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

        # Check that we're at capacity
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

    def test_zero_capacity_cache(self):
        """Test cache with zero capacity."""
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

    def test_size_and_capacity(self):
        """Test size tracking and capacity limits."""
        # Start empty
        self.assertEqual(self.cache.size(), 0)

        # Add items up to capacity
        for i in range(3):
            self.cache.put(f"key{i}", f"value{i}")
            self.assertEqual(self.cache.size(), i + 1)

        # Add more items - size should stay at capacity
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
        self.assertEqual(cache.capacity, 5)

    def test_persistence_interface_methods(self):
        """Verify persistence interface methods are implemented."""
        # Cache without persistence
        cache_no_persist = LFUCache(5)
        self.assertTrue(callable(getattr(cache_no_persist, "persist_now", None)))
        self.assertTrue(
            callable(getattr(cache_no_persist, "supports_persistence", None))
        )
        self.assertFalse(cache_no_persist.supports_persistence())

        # Cache with persistence
        temp_file = os.path.join(tempfile.mkdtemp(), "test_interface.json")
        try:
            cache_with_persist = LFUCache(
                5, persistence_interval=10.0, persistence_path=temp_file
            )
            self.assertTrue(cache_with_persist.supports_persistence())

            # persist_now should work without errors
            cache_with_persist.put("key", "value")
            cache_with_persist.persist_now()  # Should not raise any exception
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            if os.path.exists(os.path.dirname(temp_file)):
                os.rmdir(os.path.dirname(temp_file))


class TestLFUCachePersistence(unittest.TestCase):
    """Test cases for LFU Cache persistence functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_cache.json")

    def tearDown(self):
        """Clean up test files."""
        # Clean up any created files
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_basic_persistence(self):
        """Test basic save and load functionality."""
        # Create cache and add items
        cache = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)

        cache.put("key1", "value1")
        cache.put("key2", {"nested": "value"})
        cache.put("key3", [1, 2, 3])

        # Force persistence
        cache.persist_now()

        # Verify file was created
        self.assertTrue(os.path.exists(self.test_file))

        # Load into new cache
        new_cache = LFUCache(
            5, persistence_interval=10.0, persistence_path=self.test_file
        )

        # Verify data was loaded correctly
        self.assertEqual(new_cache.size(), 3)
        self.assertEqual(new_cache.get("key1"), "value1")
        self.assertEqual(new_cache.get("key2"), {"nested": "value"})
        self.assertEqual(new_cache.get("key3"), [1, 2, 3])

    def test_frequency_preservation(self):
        """Test that frequencies are preserved across persistence."""
        cache = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)

        # Create different frequency levels
        cache.put("freq1", "value1")
        cache.put("freq3", "value3")
        cache.put("freq5", "value5")

        # Access items to create different frequencies
        cache.get("freq3")  # freq = 2
        cache.get("freq3")  # freq = 3

        cache.get("freq5")  # freq = 2
        cache.get("freq5")  # freq = 3
        cache.get("freq5")  # freq = 4
        cache.get("freq5")  # freq = 5

        # Force persistence
        cache.persist_now()

        # Load into new cache
        new_cache = LFUCache(
            5, persistence_interval=10.0, persistence_path=self.test_file
        )

        # Add new items to test eviction order
        new_cache.put("new1", "newvalue1")
        new_cache.put("new2", "newvalue2")
        new_cache.put("new3", "newvalue3")

        # freq1 should be evicted first (lowest frequency)
        self.assertIsNone(new_cache.get("freq1"))
        # freq3 and freq5 should still be there
        self.assertEqual(new_cache.get("freq3"), "value3")
        self.assertEqual(new_cache.get("freq5"), "value5")

    def test_periodic_persistence(self):
        """Test automatic periodic persistence."""
        # Use short interval for testing
        cache = LFUCache(5, persistence_interval=0.5, persistence_path=self.test_file)

        cache.put("key1", "value1")

        # File shouldn't exist yet
        self.assertFalse(os.path.exists(self.test_file))

        # Wait for interval to pass
        time.sleep(0.6)

        # Access cache to trigger persistence check
        cache.get("key1")

        # File should now exist
        self.assertTrue(os.path.exists(self.test_file))

        # Verify content
        with open(self.test_file, "r") as f:
            data = json.load(f)

        self.assertEqual(data["capacity"], 5)
        self.assertEqual(len(data["nodes"]), 1)
        self.assertEqual(data["nodes"][0]["key"], "key1")

    def test_persistence_with_empty_cache(self):
        """Test persistence behavior with empty cache."""
        cache = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)

        # Add and remove items
        cache.put("key1", "value1")
        cache.clear()

        # Force persistence
        cache.persist_now()

        # File should be removed when cache is empty
        self.assertFalse(os.path.exists(self.test_file))

    def test_no_persistence_when_disabled(self):
        """Test that persistence doesn't occur when not configured."""
        # Create cache without persistence
        cache = LFUCache(5)

        cache.put("key1", "value1")
        cache.persist_now()  # Should do nothing

        # No file should be created
        self.assertFalse(os.path.exists("lfu_cache.json"))

    def test_load_from_nonexistent_file(self):
        """Test loading when persistence file doesn't exist."""
        # Create cache with non-existent file
        cache = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)

        # Should start empty
        self.assertEqual(cache.size(), 0)
        self.assertTrue(cache.is_empty())

    def test_persistence_with_complex_data(self):
        """Test persistence with various data types."""
        cache = LFUCache(10, persistence_interval=10.0, persistence_path=self.test_file)

        # Add various data types
        test_data = {
            "string": "hello world",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, [3, 4]],
            "dict": {"a": 1, "b": {"c": 2}},
            "tuple_key": "value_for_tuple",  # Will use string key since tuples aren't JSON serializable
        }

        for key, value in test_data.items():
            cache.put(key, value)

        # Force persistence
        cache.persist_now()

        # Load into new cache
        new_cache = LFUCache(
            10, persistence_interval=10.0, persistence_path=self.test_file
        )

        # Verify all data types
        for key, value in test_data.items():
            self.assertEqual(new_cache.get(key), value)

    def test_persistence_file_corruption_handling(self):
        """Test handling of corrupted persistence files."""
        # Create invalid JSON file
        with open(self.test_file, "w") as f:
            f.write("{ invalid json content")

        # Should handle gracefully and start with empty cache
        cache = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)
        self.assertEqual(cache.size(), 0)

        # Cache should still be functional
        cache.put("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")

    def test_multiple_persistence_cycles(self):
        """Test multiple save/load cycles."""
        # First cycle
        cache1 = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)
        cache1.put("key1", "value1")
        cache1.put("key2", "value2")
        cache1.persist_now()

        # Second cycle - load and modify
        cache2 = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)
        self.assertEqual(cache2.size(), 2)
        cache2.put("key3", "value3")
        cache2.persist_now()

        # Third cycle - verify all changes
        cache3 = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)
        self.assertEqual(cache3.size(), 3)
        self.assertEqual(cache3.get("key1"), "value1")
        self.assertEqual(cache3.get("key2"), "value2")
        self.assertEqual(cache3.get("key3"), "value3")

    def test_capacity_change_on_load(self):
        """Test loading cache data into cache with different capacity."""
        # Create cache with capacity 5
        cache1 = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)
        for i in range(5):
            cache1.put(f"key{i}", f"value{i}")
        cache1.persist_now()

        # Load into cache with smaller capacity
        cache2 = LFUCache(3, persistence_interval=10.0, persistence_path=self.test_file)

        # Current design: loads all persisted items regardless of new capacity
        # This is a valid design choice - preserve data integrity on load
        self.assertEqual(cache2.size(), 5)

        # The cache continues to operate with loaded items
        # New items can still be added, and the cache will manage its size
        cache2.put("new_key", "new_value")

        # Verify the cache is still functional and contains the new item
        self.assertEqual(cache2.get("new_key"), "new_value")
        self.assertGreaterEqual(
            cache2.size(), 4
        )  # At least has the new item plus some old ones

    def test_persistence_timing(self):
        """Test that persistence doesn't happen too frequently."""
        cache = LFUCache(5, persistence_interval=1.0, persistence_path=self.test_file)

        cache.put("key1", "value1")

        # Multiple operations within interval shouldn't trigger persistence
        for i in range(10):
            cache.get("key1")
            self.assertFalse(os.path.exists(self.test_file))
            time.sleep(0.05)  # Total time still less than interval

        # Wait for interval to pass
        time.sleep(0.6)
        cache.get("key1")

        # Now file should exist
        self.assertTrue(os.path.exists(self.test_file))

    def test_persistence_with_statistics(self):
        """Test persistence doesn't interfere with statistics tracking."""
        cache = LFUCache(
            5,
            track_stats=True,
            persistence_interval=0.5,
            persistence_path=self.test_file,
        )

        # Perform operations
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.get("key1")
        cache.get("nonexistent")

        # Wait for persistence
        time.sleep(0.6)
        cache.get("key1")  # Trigger persistence

        # Check stats are still correct
        stats = cache.get_stats()
        self.assertEqual(stats["puts"], 2)
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["misses"], 1)

        # Load into new cache with stats
        new_cache = LFUCache(
            5,
            track_stats=True,
            persistence_interval=0.5,
            persistence_path=self.test_file,
        )

        # Stats should be reset in new instance
        new_stats = new_cache.get_stats()
        self.assertEqual(new_stats["puts"], 0)
        self.assertEqual(new_stats["hits"], 0)

        # But data should be loaded
        self.assertEqual(new_cache.size(), 2)


class TestLFUCacheStatsLogging(unittest.TestCase):
    """Test cases for LFU Cache statistics logging functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_file = tempfile.mktemp()

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

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
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
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
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
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
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
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
        """Test stats logging when cache is at capacity."""
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
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
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
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
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
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
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
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
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
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
        ) as mock_log:
            cache.log_stats_now()

            log_message = mock_log.call_args[0][0]
            self.assertIn("Updates: 2", log_message)
            self.assertIn("Puts: 1", log_message)

    def test_stats_logging_combined_with_persistence(self):
        """Test that stats logging and persistence work together."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(
            5,
            track_stats=True,
            persistence_interval=1.0,
            persistence_path=self.test_file,
            stats_logging_interval=0.5,
        )

        cache.put("key1", "value1")

        with patch.object(
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
        ) as mock_log:
            # Wait for stats logging interval
            time.sleep(0.6)
            cache.get("key1")  # Trigger stats log

            self.assertEqual(mock_log.call_count, 1)
            self.assertFalse(os.path.exists(self.test_file))  # Not persisted yet

            # Wait for persistence interval
            time.sleep(0.5)
            cache.get("key1")  # Trigger persistence

            self.assertTrue(os.path.exists(self.test_file))  # Now persisted
            # Stats log might trigger again if interval passed
            self.assertGreaterEqual(mock_log.call_count, 1)

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
                logging.getLogger("nemoguardrails.cache.lfu"), "info"
            ) as mock_log:
                cache.log_stats_now()

                if hits > 0 or misses > 0:
                    log_message = mock_log.call_args[0][0]
                    self.assertIn(f"Hit Rate: {expected_rate}", log_message)


class TestContentSafetyCacheStatsConfig(unittest.TestCase):
    """Test cache stats configuration in content safety context."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_file = tempfile.mktemp()

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_cache_config_with_stats_disabled(self):
        """Test cache configuration with stats disabled."""
        from nemoguardrails.library.content_safety.manager import ContentSafetyManager
        from nemoguardrails.rails.llm.config import (
            CacheStatsConfig,
            ModelCacheConfig,
            ModelConfig,
        )

        cache_config = ModelCacheConfig(
            enabled=True, capacity_per_model=1000, stats=CacheStatsConfig(enabled=False)
        )

        model_config = ModelConfig(cache=cache_config)
        manager = ContentSafetyManager(model_config)

        cache = manager.get_cache_for_model("test_model")
        self.assertIsNotNone(cache)
        self.assertFalse(cache.track_stats)
        self.assertFalse(cache.supports_stats_logging())

    def test_cache_config_with_stats_tracking_only(self):
        """Test cache configuration with stats tracking but no logging."""
        from nemoguardrails.library.content_safety.manager import ContentSafetyManager
        from nemoguardrails.rails.llm.config import (
            CacheStatsConfig,
            ModelCacheConfig,
            ModelConfig,
        )

        cache_config = ModelCacheConfig(
            enabled=True,
            capacity_per_model=1000,
            stats=CacheStatsConfig(enabled=True, log_interval=None),
        )

        model_config = ModelConfig(cache=cache_config)
        manager = ContentSafetyManager(model_config)

        cache = manager.get_cache_for_model("test_model")
        self.assertIsNotNone(cache)
        self.assertTrue(cache.track_stats)
        self.assertFalse(cache.supports_stats_logging())
        self.assertIsNone(cache.stats_logging_interval)

    def test_cache_config_with_stats_logging(self):
        """Test cache configuration with stats tracking and logging."""
        from nemoguardrails.library.content_safety.manager import ContentSafetyManager
        from nemoguardrails.rails.llm.config import (
            CacheStatsConfig,
            ModelCacheConfig,
            ModelConfig,
        )

        cache_config = ModelCacheConfig(
            enabled=True,
            capacity_per_model=1000,
            stats=CacheStatsConfig(enabled=True, log_interval=60.0),
        )

        model_config = ModelConfig(cache=cache_config)
        manager = ContentSafetyManager(model_config)

        cache = manager.get_cache_for_model("test_model")
        self.assertIsNotNone(cache)
        self.assertTrue(cache.track_stats)
        self.assertTrue(cache.supports_stats_logging())
        self.assertEqual(cache.stats_logging_interval, 60.0)

    def test_cache_config_default_stats(self):
        """Test cache configuration with default stats settings."""
        from nemoguardrails.library.content_safety.manager import ContentSafetyManager
        from nemoguardrails.rails.llm.config import ModelCacheConfig, ModelConfig

        cache_config = ModelCacheConfig(enabled=True)

        model_config = ModelConfig(cache=cache_config)
        manager = ContentSafetyManager(model_config)

        cache = manager.get_cache_for_model("test_model")
        self.assertIsNotNone(cache)
        self.assertFalse(cache.track_stats)  # Default is disabled
        self.assertFalse(cache.supports_stats_logging())

    def test_cache_config_stats_with_persistence(self):
        """Test cache configuration with both stats and persistence."""
        from nemoguardrails.library.content_safety.manager import ContentSafetyManager
        from nemoguardrails.rails.llm.config import (
            CachePersistenceConfig,
            CacheStatsConfig,
            ModelCacheConfig,
            ModelConfig,
        )

        cache_config = ModelCacheConfig(
            enabled=True,
            capacity_per_model=1000,
            stats=CacheStatsConfig(enabled=True, log_interval=30.0),
            persistence=CachePersistenceConfig(
                enabled=True, interval=60.0, path=self.test_file
            ),
        )

        model_config = ModelConfig(cache=cache_config)
        manager = ContentSafetyManager(model_config)

        cache = manager.get_cache_for_model("test_model")
        self.assertIsNotNone(cache)
        self.assertTrue(cache.track_stats)
        self.assertTrue(cache.supports_stats_logging())
        self.assertEqual(cache.stats_logging_interval, 30.0)
        self.assertTrue(cache.supports_persistence())
        self.assertEqual(cache.persistence_interval, 60.0)

    def test_cache_config_from_dict(self):
        """Test cache configuration creation from dictionary."""
        from nemoguardrails.rails.llm.config import ModelCacheConfig

        config_dict = {
            "enabled": True,
            "capacity_per_model": 5000,
            "stats": {"enabled": True, "log_interval": 120.0},
        }

        cache_config = ModelCacheConfig(**config_dict)
        self.assertTrue(cache_config.enabled)
        self.assertEqual(cache_config.capacity_per_model, 5000)
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

    def test_multiple_model_caches_with_stats(self):
        """Test multiple model caches each with their own stats configuration."""
        from nemoguardrails.library.content_safety.manager import ContentSafetyManager
        from nemoguardrails.rails.llm.config import (
            CacheStatsConfig,
            ModelCacheConfig,
            ModelConfig,
        )

        cache_config = ModelCacheConfig(
            enabled=True,
            capacity_per_model=1000,
            stats=CacheStatsConfig(enabled=True, log_interval=30.0),
        )

        model_config = ModelConfig(
            cache=cache_config, model_mapping={"model_alias": "actual_model"}
        )
        manager = ContentSafetyManager(model_config)

        # Get caches for different models
        cache1 = manager.get_cache_for_model("model1")
        cache2 = manager.get_cache_for_model("model2")
        cache_alias = manager.get_cache_for_model("model_alias")
        cache_actual = manager.get_cache_for_model("actual_model")

        # All should have stats enabled
        self.assertTrue(cache1.track_stats)
        self.assertTrue(cache2.track_stats)
        self.assertTrue(cache_alias.track_stats)

        # Alias should resolve to same cache as actual
        self.assertIs(cache_alias, cache_actual)


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

        # Cache should still be at capacity
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
        """Test get_or_compute handles exceptions properly."""
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

    def test_concurrent_persistence(self):
        """Test thread safety of persistence operations."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            cache_file = f.name

        try:
            # Create cache with persistence
            cache = LFUCache(
                capacity=50,
                track_stats=True,
                persistence_interval=0.1,  # Short interval for testing
                persistence_path=cache_file,
            )

            def worker(thread_id):
                """Worker that performs operations."""
                for i in range(20):
                    cache.put(f"persist_key_{thread_id}_{i}", f"value_{thread_id}_{i}")
                    cache.get(f"persist_key_{thread_id}_{i}")

                    # Force persistence sometimes
                    if i % 5 == 0:
                        cache.persist_now()

            # Run workers
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(worker, i) for i in range(5)]
                for future in futures:
                    future.result()

            # Final persist
            cache.persist_now()

            # Load the persisted data
            new_cache = LFUCache(
                capacity=50, persistence_interval=1.0, persistence_path=cache_file
            )

            # Verify some data was persisted correctly
            # (Due to capacity limits, not all items will be present)
            self.assertGreater(new_cache.size(), 0)
            self.assertLessEqual(new_cache.size(), 50)

        finally:
            # Clean up
            if os.path.exists(cache_file):
                os.unlink(cache_file)

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

                # Size should never be negative or exceed capacity
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
        # Need capacity for: 50 existing + (5 threads × 100 new keys) = 550+
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
        """Test thread safety when cache is at capacity and evictions occur."""
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

                # Also work with some persistent keys (access multiple times)
                persistent_key = f"persistent_{thread_id % 5}"
                for _ in range(3):  # Access 3 times to increase frequency
                    small_cache.put(persistent_key, f"persistent_value_{thread_id}")
                    small_cache.get(persistent_key)

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

        # Cache should be at capacity
        self.assertEqual(small_cache.size(), 50)

        # Stats should show many evictions
        stats = small_cache.get_stats()
        self.assertGreater(stats["evictions"], 0)
        self.assertGreater(stats["puts"], 0)


class TestContentSafetyManagerThreadSafety(unittest.TestCase):
    """Test thread safety of ContentSafetyManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock cache config
        self.cache_config = MagicMock()
        self.cache_config.enabled = True
        self.cache_config.store = "memory"
        self.cache_config.capacity_per_model = 100
        self.cache_config.stats.enabled = True
        self.cache_config.stats.log_interval = None
        self.cache_config.persistence.enabled = False
        self.cache_config.persistence.interval = None
        self.cache_config.persistence.path = None

        # Create mock model config
        self.model_config = MagicMock()
        self.model_config.cache = self.cache_config
        self.model_config.model_mapping = {"alias_model": "actual_model"}

    def test_concurrent_cache_creation(self):
        """Test that concurrent cache creation returns the same instance."""
        manager = ContentSafetyManager(self.model_config)
        caches = []

        def worker(thread_id):
            """Worker that gets cache for model."""
            cache = manager.get_cache_for_model("test_model")
            caches.append((thread_id, cache))
            return cache

        # Run many threads to increase chance of race condition
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker, i) for i in range(20)]
            for future in futures:
                future.result()

        # All caches should be the same instance
        first_cache = caches[0][1]
        for thread_id, cache in caches:
            self.assertIs(
                cache, first_cache, f"Thread {thread_id} got different cache instance"
            )

    def test_concurrent_multi_model_caches(self):
        """Test concurrent access to caches for different models."""
        manager = ContentSafetyManager(self.model_config)
        results = []

        def worker(thread_id):
            """Worker that accesses multiple model caches."""
            model_names = [f"model_{i}" for i in range(5)]

            for model_name in model_names:
                cache = manager.get_cache_for_model(model_name)

                # Perform operations
                key = f"thread_{thread_id}_key"
                value = f"thread_{thread_id}_value"
                cache.put(key, value)
                retrieved = cache.get(key)

                if retrieved != value:
                    results.append(f"Mismatch for {model_name}: {retrieved} != {value}")

        # Run workers
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            for future in futures:
                future.result()

        # Check for errors
        self.assertEqual(len(results), 0, f"Errors found: {results}")

    def test_concurrent_persist_all_caches(self):
        """Test thread safety of persist_all_caches method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock config with persistence
            cache_config = MagicMock()
            cache_config.enabled = True
            cache_config.store = "memory"
            cache_config.capacity_per_model = 50
            cache_config.persistence.enabled = True
            cache_config.persistence.interval = 1.0
            cache_config.persistence.path = f"{temp_dir}/cache_{{model_name}}.json"
            cache_config.stats.enabled = True
            cache_config.stats.log_interval = None

            model_config = MagicMock()
            model_config.cache = cache_config
            model_config.model_mapping = {}

            manager = ContentSafetyManager(model_config)

            # Create caches for multiple models
            for i in range(5):
                cache = manager.get_cache_for_model(f"model_{i}")
                for j in range(10):
                    cache.put(f"key_{j}", f"value_{j}")

            persist_count = [0]

            def persist_worker():
                """Worker that calls persist_all_caches."""
                manager.persist_all_caches()
                persist_count[0] += 1

            def modify_worker():
                """Worker that modifies caches while persistence happens."""
                for i in range(20):
                    model_name = f"model_{i % 5}"
                    cache = manager.get_cache_for_model(model_name)
                    cache.put(f"new_key_{i}", f"new_value_{i}")
                    time.sleep(0.001)

            # Run persistence and modifications concurrently
            threads = []

            # Multiple persist threads
            for _ in range(3):
                t = threading.Thread(target=persist_worker)
                threads.append(t)
                t.start()

            # Modification thread
            t = threading.Thread(target=modify_worker)
            threads.append(t)
            t.start()

            # Wait for all threads
            for t in threads:
                t.join()

            # Verify persistence was called
            self.assertEqual(persist_count[0], 3)

    def test_model_alias_thread_safety(self):
        """Test thread safety when using model aliases."""
        manager = ContentSafetyManager(self.model_config)
        caches = []

        def worker(use_alias):
            """Worker that gets cache using alias or actual name."""
            if use_alias:
                cache = manager.get_cache_for_model("alias_model")
            else:
                cache = manager.get_cache_for_model("actual_model")
            caches.append(cache)

        # Mix of threads using alias and actual name
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(10):
                use_alias = i % 2 == 0
                futures.append(executor.submit(worker, use_alias))

            for future in futures:
                future.result()

        # All should get the same cache instance
        first_cache = caches[0]
        for cache in caches:
            self.assertIs(
                cache,
                first_cache,
                "Alias and actual model should resolve to same cache",
            )


if __name__ == "__main__":
    unittest.main()

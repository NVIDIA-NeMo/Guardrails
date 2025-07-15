#!/usr/bin/env python3
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
Test script for translation caching functionality.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from nemoguardrails.evaluate.utils_translate import (
    TranslationCache,
    get_translation_cache,
    load_dataset,
)


def test_translation_cache():
    """Test the translation caching functionality."""

    # Set a dummy API key for testing
    os.environ["DEEPL_API_KEY"] = "test_key"

    # Create a simple test dataset
    test_data = [
        "Hello, how are you?",
        "This is a test message.",
        "Hello, how are you?",  # Duplicate to test cache
        "Another test message.",
    ]

    # Save test data to a temporary file
    with open("test_data.txt", "w") as f:
        for line in test_data:
            f.write(line + "\n")

    print("Testing translation caching...")
    print("=" * 50)

    # Create a temporary translation config file
    with open("translation_config.yaml", "w") as f:
        translation_config = {
            "langproviders": [
                {"language": "en,ja", "model_type": "remote.DeeplTranslator"}
            ]
        }
        import yaml

        yaml.dump(translation_config, f)

    # First run - should create cache entries
    print("First run (creating cache):")
    try:
        translated_data = load_dataset(
            "test_data.txt", translation_config="translation_config.yaml"
        )
        print(f"Translated {len(translated_data)} items")
        for i, item in enumerate(translated_data):
            print(f"  {i+1}: {item}")
    except Exception as e:
        print(f"Translation failed (expected with test key): {e}")

    # Check cache stats - use service name for DeeplTranslator
    cache = get_translation_cache("DeeplTranslator")
    stats = cache.get_cache_stats()
    print(f"\nCache stats after first run: {stats}")
    print(f"Cache file: {stats.get('cache_file', 'N/A')}")

    # Second run - should use cache
    print("\nSecond run (using cache):")
    try:
        translated_data2 = load_dataset(
            "test_data.txt", translation_config="translation_config.yaml"
        )
        print(f"Translated {len(translated_data2)} items")
        for i, item in enumerate(translated_data2):
            print(f"  {i+1}: {item}")
    except Exception as e:
        print(f"Translation failed (expected with test key): {e}")

    # Check cache stats again
    stats2 = cache.get_cache_stats()
    print(f"\nCache stats after second run: {stats2}")
    print(f"Cache file: {stats2.get('cache_file', 'N/A')}")

    # Show cache file contents - use new file name format
    expected_cache_file = "translation_cache/translations_DeeplTranslator.json"
    if os.path.exists(expected_cache_file):
        print(f"\nCache file contents ({expected_cache_file}):")
        with open(expected_cache_file, "r") as f:
            cache_data = json.load(f)
            print(f"Cache entries: {len(cache_data)}")
            for key, value in list(cache_data.items())[:3]:  # Show first 3 entries
                print(f"  {key[:20]}... -> {value[:50]}...")
    else:
        print(f"\nCache file not found: {expected_cache_file}")

    # Test different service names
    print("\nTesting different service names:")
    test_services = ["DeeplTranslator", "RivaTranslator", "LocalTranslator", "default"]
    for service_name in test_services:
        cache_instance = get_translation_cache(service_name)
        stats = cache_instance.get_cache_stats()
        print(f"  {service_name}: {stats.get('cache_file', 'N/A')}")

    # Cleanup
    if os.path.exists("test_data.txt"):
        os.remove("test_data.txt")
    if os.path.exists("translation_config.yaml"):
        os.remove("translation_config.yaml")


class TestTranslationCache:
    """Test cases for TranslationCache class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "test_cache")

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_translation_cache_initialization(self):
        """Test TranslationCache initialization with different service names."""
        # Test with default service name
        cache1 = TranslationCache(cache_dir=self.cache_dir, service_name="default")
        assert cache1.cache_file == Path(self.cache_dir) / "translations_default.json"

        # Test with custom service name
        cache2 = TranslationCache(
            cache_dir=self.cache_dir, service_name="DeeplTranslator"
        )
        assert (
            cache2.cache_file
            == Path(self.cache_dir) / "translations_DeeplTranslator.json"
        )

        # Test with service name containing special characters
        cache3 = TranslationCache(
            cache_dir=self.cache_dir, service_name="remote/DeeplTranslator"
        )
        assert (
            cache3.cache_file
            == Path(self.cache_dir) / "translations_remote_DeeplTranslator.json"
        )

    def test_cache_operations(self):
        """Test basic cache operations (get, set)."""
        cache = TranslationCache(cache_dir=self.cache_dir, service_name="test_service")

        # Test setting and getting cache entries
        text = "Hello, world!"
        target_lang = "ja"
        translated_text = "こんにちは、世界！"

        # Initially, cache should be empty
        assert cache.get(text, target_lang) is None

        # Set cache entry
        cache.set(text, target_lang, translated_text)

        # Get cache entry
        result = cache.get(text, target_lang)
        assert result["translation"] == translated_text

    def test_cache_persistence(self):
        """Test that cache persists between instances."""
        service_name = "persistence_test"
        text = "Test message"
        target_lang = "fr"
        translated_text = "Message de test"

        # Create first cache instance and set entry
        cache1 = TranslationCache(cache_dir=self.cache_dir, service_name=service_name)
        cache1.set(text, target_lang, translated_text)

        # Create second cache instance and check if entry exists
        cache2 = TranslationCache(cache_dir=self.cache_dir, service_name=service_name)
        result = cache2.get(text, target_lang)
        assert result["translation"] == translated_text

    def test_cache_stats(self):
        """Test cache statistics functionality."""
        cache = TranslationCache(cache_dir=self.cache_dir, service_name="stats_test")

        # Add some entries
        cache.set("text1", "ja", "translation1")
        cache.set("text2", "ja", "translation2")

        stats = cache.get_cache_stats()

        assert "total_entries" in stats
        assert "cache_size_bytes" in stats
        assert "cache_size_mb" in stats
        assert "cache_file" in stats
        assert stats["total_entries"] == 2
        assert stats["cache_file"] == str(cache.cache_file)

    def test_get_translation_cache_function(self):
        """Test get_translation_cache function with different service names."""
        # Test with different service names
        service_names = [
            "DeeplTranslator",
            "RivaTranslator",
            "LocalTranslator",
            "default",
        ]
        cache_instances = {}

        for service_name in service_names:
            cache = get_translation_cache(service_name)
            cache_instances[service_name] = cache

            # Verify cache file name
            expected_file = f"translations_{service_name}.json"
            assert cache.cache_file.name == expected_file

        # Verify that different service names create different cache instances
        assert (
            cache_instances["DeeplTranslator"] is not cache_instances["RivaTranslator"]
        )
        assert (
            cache_instances["RivaTranslator"] is not cache_instances["LocalTranslator"]
        )

    def test_cache_key_generation(self):
        """Test cache key generation."""
        cache = TranslationCache(cache_dir=self.cache_dir, service_name="key_test")

        # Test cache key generation
        text = "Hello, world!"
        target_lang = "ja"
        actual_key = cache._get_cache_key(text, target_lang)
        assert isinstance(actual_key, str)


if __name__ == "__main__":
    test_translation_cache()

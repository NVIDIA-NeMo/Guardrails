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

import asyncio
from typing import Any

import pytest

from nemoguardrails.llm.cache.interface import CacheInterface


class MinimalCache(CacheInterface):
    def __init__(self, maxsize: int = 10):
        self._cache = {}
        self._maxsize = maxsize

    def get(self, key: Any, default: Any = None) -> Any:
        return self._cache.get(key, default)

    def put(self, key: Any, value: Any) -> None:
        self._cache[key] = value

    def size(self) -> int:
        return len(self._cache)

    def is_empty(self) -> bool:
        return len(self._cache) == 0

    def clear(self) -> None:
        self._cache.clear()

    @property
    def maxsize(self) -> int:
        return self._maxsize


@pytest.fixture
def cache():
    return MinimalCache()


def test_contains(cache):
    cache.put("key1", "value1")
    assert cache.contains("key1")


def test_get_stats(cache):
    stats = cache.get_stats()
    assert isinstance(stats, dict)
    assert "message" in stats


def test_reset_stats(cache):
    cache.reset_stats()


def test_log_stats_now(cache):
    cache.log_stats_now()


def test_supports_stats_logging(cache):
    assert cache.supports_stats_logging() is False


@pytest.mark.asyncio
async def test_get_or_compute_cache_hit(cache):
    cache.put("key1", "cached_value")

    async def compute_fn():
        return "computed_value"

    result = await cache.get_or_compute("key1", compute_fn)
    assert result == "cached_value"


@pytest.mark.asyncio
async def test_get_or_compute_cache_miss(cache):
    async def compute_fn():
        return "computed_value"

    result = await cache.get_or_compute("key1", compute_fn)
    assert result == "computed_value"
    assert cache.get("key1") == "computed_value"


@pytest.mark.asyncio
async def test_get_or_compute_exception(cache):
    async def failing_compute():
        raise ValueError("Computation failed")

    result = await cache.get_or_compute("key1", failing_compute, default="fallback")
    assert result == "fallback"

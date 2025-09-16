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

import logging
from typing import Optional

from nemoguardrails.cache.interface import CacheInterface
from nemoguardrails.cache.lfu import LFUCache
from nemoguardrails.rails.llm.config import ModelCacheConfig

log = logging.getLogger(__name__)


class ContentSafetyManager:
    """Manages content safety functionality for a specific model."""

    def __init__(
        self, model_name: str, cache_config: Optional[ModelCacheConfig] = None
    ):
        self.model_name = model_name
        self.cache_config = cache_config
        self._cache: Optional[CacheInterface] = None
        self._initialize_cache()

    def _initialize_cache(self):
        """Initialize cache based on configuration."""
        if not self.cache_config or not self.cache_config.enabled:
            log.debug(
                f"Content safety caching is disabled for model '{self.model_name}'"
            )
            return

        # Create cache based on store type
        if self.cache_config.store == "memory":
            # Determine persistence settings
            persistence_path = None
            persistence_interval = None

            if (
                self.cache_config.persistence.enabled
                and self.cache_config.persistence.interval is not None
            ):
                persistence_interval = self.cache_config.persistence.interval

                if self.cache_config.persistence.path:
                    # Use configured path, replacing {model_name} if present
                    persistence_path = self.cache_config.persistence.path.replace(
                        "{model_name}", self.model_name
                    )
                else:
                    # Default path if persistence is enabled but no path specified
                    persistence_path = f"cache_{self.model_name}.json"

            # Determine stats logging settings
            stats_logging_interval = None
            if (
                self.cache_config.stats.enabled
                and self.cache_config.stats.log_interval is not None
            ):
                stats_logging_interval = self.cache_config.stats.log_interval

            self._cache = LFUCache(
                capacity=self.cache_config.capacity_per_model,
                track_stats=self.cache_config.stats.enabled,
                persistence_interval=persistence_interval,
                persistence_path=persistence_path,
                stats_logging_interval=stats_logging_interval,
            )

            log.info(
                f"Created cache for model '{self.model_name}' with capacity {self.cache_config.capacity_per_model}"
            )
        # elif self.cache_config.store == "filesystem":
        #     self._cache = FilesystemCache(...)
        # elif self.cache_config.store == "redis":
        #     self._cache = RedisCache(...)

    def get_cache(self) -> Optional[CacheInterface]:
        """Get the cache for this model."""
        return self._cache

    def persist_cache(self):
        """Force immediate persistence of cache if it supports it."""
        if self._cache and self._cache.supports_persistence():
            self._cache.persist_now()
            log.info(f"Persisted cache for model: {self.model_name}")

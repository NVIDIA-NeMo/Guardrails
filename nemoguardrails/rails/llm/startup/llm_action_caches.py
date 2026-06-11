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

"""LLM action model caches."""

import logging
from typing import Dict

from nemoguardrails.llm.cache import CacheInterface, LFUCache
from nemoguardrails.rails.llm.config import Model, RailsConfig
from nemoguardrails.rails.llm.types import LLMActionCacheSurface

log = logging.getLogger(__name__)

__all__ = [
    "build_llm_action_cache",
    "build_llm_action_caches",
    "initialize_llm_action_caches",
]


def build_llm_action_cache(model: Model) -> LFUCache:
    """Build the cache configured for one LLM action model."""
    if model.cache is None:
        raise ValueError(f"Missing cache configuration for model '{model.type}'.")

    if model.cache.maxsize <= 0:
        raise ValueError(
            f"Invalid cache maxsize for model '{model.type}': {model.cache.maxsize}. "
            "Capacity must be greater than 0. Skipping cache creation."
        )

    stats_logging_interval = None
    if model.cache.stats.enabled and model.cache.stats.log_interval is not None:
        stats_logging_interval = model.cache.stats.log_interval

    cache = LFUCache(
        maxsize=model.cache.maxsize,
        track_stats=model.cache.stats.enabled,
        stats_logging_interval=stats_logging_interval,
    )

    log.info(f"Created cache for model '{model.type}' with maxsize {model.cache.maxsize}")

    return cache


def build_llm_action_caches(config: RailsConfig) -> Dict[str, CacheInterface]:
    """Build caches for configured action models."""
    model_caches: Dict[str, CacheInterface] = dict()
    for model in config.models:
        if model.type in ["main", "embeddings"]:
            continue

        if model.cache and model.cache.enabled:
            cache = build_llm_action_cache(model)
            model_caches[model.type] = cache

            log.info(
                f"Initialized model '{model.type}' with cache %s",
                "enabled" if cache else "disabled",
            )

    return model_caches


def initialize_llm_action_caches(rails: LLMActionCacheSurface) -> None:
    """Initialize configured action model caches for an LLMRails instance."""
    model_caches = build_llm_action_caches(rails.config)
    if model_caches:
        rails.runtime.register_action_param("model_caches", model_caches)

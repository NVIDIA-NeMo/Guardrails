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

import pytest

from nemoguardrails.rails.llm.config import (
    CacheStatsConfig,
    Model,
    ModelCacheConfig,
    RailsConfig,
)
from nemoguardrails.rails.llm.startup.llm_action_caches import (
    build_llm_action_cache,
    build_llm_action_caches,
)


def test_build_llm_action_caches_skips_main_embeddings_and_disabled_caches():
    config = RailsConfig(
        models=[
            Model(
                type="main",
                engine="fake",
                model="fake",
                cache=ModelCacheConfig(enabled=True),
            ),
            Model(
                type="embeddings",
                engine="fake",
                model="fake",
                cache=ModelCacheConfig(enabled=True),
            ),
            Model(
                type="content_safety",
                engine="fake",
                model="fake",
                cache=ModelCacheConfig(enabled=False),
            ),
        ]
    )

    assert build_llm_action_caches(config) == {}


def test_build_llm_action_caches_creates_enabled_action_model_caches():
    config = RailsConfig(
        models=[
            Model(
                type="content_safety",
                engine="fake",
                model="fake",
                cache=ModelCacheConfig(enabled=True, maxsize=1000),
            ),
            Model(
                type="jailbreak_detection",
                engine="fake",
                model="fake",
                cache=ModelCacheConfig(enabled=True, maxsize=2000),
            ),
        ]
    )

    model_caches = build_llm_action_caches(config)

    assert set(model_caches.keys()) == {"content_safety", "jailbreak_detection"}
    assert model_caches["content_safety"].maxsize == 1000
    assert model_caches["jailbreak_detection"].maxsize == 2000


def test_build_llm_action_cache_preserves_stats_config():
    model = Model(
        type="content_safety",
        engine="fake",
        model="fake",
        cache=ModelCacheConfig(
            enabled=True,
            maxsize=5000,
            stats=CacheStatsConfig(enabled=True, log_interval=60.0),
        ),
    )

    cache = build_llm_action_cache(model)

    assert cache.maxsize == 5000
    assert cache.track_stats is True
    assert cache.stats_logging_interval == 60.0
    assert cache.supports_stats_logging() is True


def test_build_llm_action_cache_rejects_invalid_maxsize():
    model = Model(
        type="content_safety",
        engine="fake",
        model="fake",
        cache=ModelCacheConfig(enabled=True, maxsize=0),
    )

    with pytest.raises(ValueError, match="Invalid cache maxsize"):
        build_llm_action_cache(model)

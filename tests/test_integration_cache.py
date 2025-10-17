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

from unittest.mock import patch

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.config import CacheStatsConfig, Model, ModelCacheConfig
from tests.utils import FakeLLM


@pytest.mark.asyncio
@patch("nemoguardrails.rails.llm.llmrails.init_llm_model")
async def test_end_to_end_cache_integration_with_content_safety(mock_init_llm_model):
    mock_llm = FakeLLM(responses=["express greeting"])
    mock_init_llm_model.return_value = mock_llm

    config = RailsConfig(
        models=[
            Model(
                type="main",
                engine="fake",
                model="fake",
            ),
            Model(
                type="content_safety",
                engine="fake",
                model="fake-content-safety",
                cache=ModelCacheConfig(
                    enabled=True,
                    maxsize=100,
                    stats=CacheStatsConfig(enabled=True),
                ),
            ),
        ]
    )

    llm = FakeLLM(responses=["express greeting"])

    rails = LLMRails(config=config, llm=llm, verbose=False)

    model_caches = rails.runtime.registered_action_params.get("model_caches", {})
    assert "content_safety" in model_caches
    cache = model_caches["content_safety"]
    assert cache is not None

    assert cache.size() == 0

    messages = [{"role": "user", "content": "Hello!"}]
    await rails.generate_async(messages=messages)

    if cache.size() > 0:
        initial_stats = cache.get_stats()
        await rails.generate_async(messages=messages)
        second_stats = cache.get_stats()
        assert second_stats["hits"] >= initial_stats["hits"]


@pytest.mark.asyncio
@patch("nemoguardrails.rails.llm.llmrails.init_llm_model")
async def test_cache_isolation_between_models(mock_init_llm_model):
    mock_llm = FakeLLM(responses=["safe"])
    mock_init_llm_model.return_value = mock_llm

    config = RailsConfig(
        models=[
            Model(
                type="main",
                engine="fake",
                model="fake",
            ),
            Model(
                type="content_safety",
                engine="fake",
                model="fake-content-safety",
                cache=ModelCacheConfig(enabled=True, maxsize=50),
            ),
            Model(
                type="jailbreak_detection",
                engine="fake",
                model="fake-jailbreak",
                cache=ModelCacheConfig(enabled=True, maxsize=100),
            ),
        ]
    )

    llm = FakeLLM(responses=["safe"])

    rails = LLMRails(config=config, llm=llm, verbose=False)

    model_caches = rails.runtime.registered_action_params.get("model_caches", {})
    assert "content_safety" in model_caches
    assert "jailbreak_detection" in model_caches

    content_safety_cache = model_caches["content_safety"]
    jailbreak_cache = model_caches["jailbreak_detection"]

    assert content_safety_cache is not jailbreak_cache
    assert content_safety_cache.maxsize == 50
    assert jailbreak_cache.maxsize == 100

    content_safety_cache.put("key1", "value1")
    assert content_safety_cache.get("key1") == "value1"
    assert jailbreak_cache.get("key1") is None


@pytest.mark.asyncio
@patch("nemoguardrails.rails.llm.llmrails.init_llm_model")
async def test_cache_disabled_for_main_model_in_integration(mock_init_llm_model):
    mock_llm = FakeLLM(responses=["safe"])
    mock_init_llm_model.return_value = mock_llm

    config = RailsConfig(
        models=[
            Model(
                type="main",
                engine="fake",
                model="fake",
                cache=ModelCacheConfig(enabled=True, maxsize=100),
            ),
            Model(
                type="content_safety",
                engine="fake",
                model="fake-content-safety",
                cache=ModelCacheConfig(enabled=True, maxsize=100),
            ),
        ]
    )

    llm = FakeLLM(responses=["safe"])

    rails = LLMRails(config=config, llm=llm, verbose=False)

    model_caches = rails.runtime.registered_action_params.get("model_caches", {})
    assert "main" not in model_caches or model_caches["main"] is None
    assert "content_safety" in model_caches
    assert model_caches["content_safety"] is not None

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

from unittest.mock import MagicMock

import pytest

from nemoguardrails.context import llm_call_info_var, llm_stats_var
from nemoguardrails.library.content_safety.actions import content_safety_check_input
from nemoguardrails.llm.cache.lfu import LFUCache
from nemoguardrails.llm.cache.utils import create_normalized_cache_key
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.logging.stats import LLMStats
from tests.utils import FakeLLM


@pytest.fixture
def mock_task_manager():
    tm = MagicMock()
    tm.render_task_prompt.return_value = "test prompt"
    tm.get_stop_tokens.return_value = []
    tm.get_max_tokens.return_value = 3
    tm.parse_task_output.return_value = [True, "policy1"]
    return tm


@pytest.fixture
def fake_llm_with_stats():
    llm = FakeLLM(responses=["safe"])
    return {"test_model": llm}


@pytest.mark.asyncio
async def test_content_safety_cache_stores_result_and_stats(
    fake_llm_with_stats, mock_task_manager
):
    cache = LFUCache(maxsize=10)
    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    result = await content_safety_check_input(
        llms=fake_llm_with_stats,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context={"user_message": "test input"},
        model_caches={"test_model": cache},
    )

    assert result["allowed"] is True
    assert result["policy_violations"] == ["policy1"]
    assert cache.size() == 1

    llm_call_info = llm_call_info_var.get()

    cache_key = create_normalized_cache_key("test prompt")
    cached_entry = cache.get(cache_key)
    assert cached_entry is not None
    assert "result" in cached_entry
    assert "llm_stats" in cached_entry

    if llm_call_info and (
        llm_call_info.total_tokens
        or llm_call_info.prompt_tokens
        or llm_call_info.completion_tokens
    ):
        assert cached_entry["llm_stats"] is not None
    else:
        assert cached_entry["llm_stats"] is None or all(
            v == 0 for v in cached_entry["llm_stats"].values()
        )


@pytest.mark.asyncio
async def test_content_safety_cache_retrieves_result_and_restores_stats(
    fake_llm_with_stats, mock_task_manager
):
    cache = LFUCache(maxsize=10)

    cache_entry = {
        "result": {"allowed": True, "policy_violations": ["policy1"]},
        "llm_stats": {
            "total_tokens": 100,
            "prompt_tokens": 80,
            "completion_tokens": 20,
        },
    }
    cache_key = create_normalized_cache_key("test prompt")
    cache.put(cache_key, cache_entry)

    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    result = await content_safety_check_input(
        llms=fake_llm_with_stats,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context={"user_message": "test input"},
        model_caches={"test_model": cache},
    )

    llm_call_info = llm_call_info_var.get()

    assert result == cache_entry["result"]
    assert llm_stats.get_stat("total_calls") == 1
    assert llm_stats.get_stat("total_tokens") == 100
    assert llm_stats.get_stat("total_prompt_tokens") == 80
    assert llm_stats.get_stat("total_completion_tokens") == 20

    assert llm_call_info.from_cache is True
    assert llm_call_info.total_tokens == 100
    assert llm_call_info.prompt_tokens == 80
    assert llm_call_info.completion_tokens == 20
    assert llm_call_info.duration is not None


@pytest.mark.asyncio
async def test_content_safety_cache_duration_reflects_cache_read_time(
    fake_llm_with_stats, mock_task_manager
):
    cache = LFUCache(maxsize=10)

    cache_entry = {
        "result": {"allowed": True, "policy_violations": []},
        "llm_stats": {
            "total_tokens": 50,
            "prompt_tokens": 40,
            "completion_tokens": 10,
        },
    }
    cache_key = create_normalized_cache_key("test prompt")
    cache.put(cache_key, cache_entry)

    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    await content_safety_check_input(
        llms=fake_llm_with_stats,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context={"user_message": "test input"},
        model_caches={"test_model": cache},
    )

    llm_call_info = llm_call_info_var.get()
    cache_duration = llm_call_info.duration

    assert cache_duration is not None
    assert cache_duration < 0.1
    assert llm_call_info.from_cache is True


@pytest.mark.asyncio
async def test_content_safety_without_cache_does_not_store(
    fake_llm_with_stats, mock_task_manager
):
    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    llm_call_info = LLMCallInfo(task="content_safety_check_input $model=test_model")
    llm_call_info_var.set(llm_call_info)

    result = await content_safety_check_input(
        llms=fake_llm_with_stats,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context={"user_message": "test input"},
    )

    assert result["allowed"] is True
    assert llm_call_info.from_cache is False


@pytest.mark.asyncio
async def test_content_safety_cache_handles_missing_stats_gracefully(
    fake_llm_with_stats, mock_task_manager
):
    cache = LFUCache(maxsize=10)

    cache_entry = {
        "result": {"allowed": True, "policy_violations": []},
        "llm_stats": None,
    }
    cache_key = create_normalized_cache_key("test_key")
    cache.put(cache_key, cache_entry)

    mock_task_manager.render_task_prompt.return_value = "test_key"

    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    llm_call_info = LLMCallInfo(task="content_safety_check_input $model=test_model")
    llm_call_info_var.set(llm_call_info)

    result = await content_safety_check_input(
        llms=fake_llm_with_stats,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context={"user_message": "test input"},
        model_caches={"test_model": cache},
    )

    assert result["allowed"] is True
    assert llm_stats.get_stat("total_calls") == 0

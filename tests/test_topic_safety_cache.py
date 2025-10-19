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
from nemoguardrails.library.topic_safety.actions import topic_safety_check_input
from nemoguardrails.llm.cache.lfu import LFUCache
from nemoguardrails.llm.cache.utils import create_normalized_cache_key
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.logging.stats import LLMStats
from tests.utils import FakeLLM


@pytest.fixture
def mock_task_manager():
    tm = MagicMock()
    tm.render_task_prompt.return_value = "Is this on topic?"
    tm.get_stop_tokens.return_value = []
    tm.get_max_tokens.return_value = 10
    return tm


@pytest.fixture
def fake_llm_topic():
    llm = FakeLLM(responses=["on-topic"])
    return {"test_model": llm}


@pytest.mark.asyncio
async def test_topic_safety_cache_stores_result(fake_llm_topic, mock_task_manager):
    cache = LFUCache(maxsize=10)

    result = await topic_safety_check_input(
        llms=fake_llm_topic,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context={"user_message": "What is AI?"},
        events=[],
        model_caches={"test_model": cache},
    )

    assert result["on_topic"] is True
    assert cache.size() == 1


@pytest.mark.asyncio
async def test_topic_safety_cache_hit(fake_llm_topic, mock_task_manager):
    cache = LFUCache(maxsize=10)

    system_prompt_with_restriction = (
        "Is this on topic?\n\n"
        'If any of the above conditions are violated, please respond with "off-topic". '
        'Otherwise, respond with "on-topic". '
        'You must respond with "on-topic" or "off-topic".'
    )

    messages = [
        {"type": "system", "content": system_prompt_with_restriction},
        {"type": "user", "content": "What is AI?"},
    ]
    cache_entry = {
        "result": {"on_topic": False},
        "llm_stats": {
            "total_tokens": 50,
            "prompt_tokens": 40,
            "completion_tokens": 10,
        },
        "llm_metadata": None,
    }
    cache_key = create_normalized_cache_key(messages)
    cache.put(cache_key, cache_entry)

    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    llm_call_info = LLMCallInfo(task="topic_safety_check_input $model=test_model")
    llm_call_info_var.set(llm_call_info)

    result = await topic_safety_check_input(
        llms=fake_llm_topic,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context={"user_message": "What is AI?"},
        events=[],
        model_caches={"test_model": cache},
    )

    assert result["on_topic"] is False
    assert llm_stats.get_stat("total_calls") == 1
    assert llm_stats.get_stat("total_tokens") == 50

    llm_call_info = llm_call_info_var.get()
    assert llm_call_info.from_cache is True


@pytest.mark.asyncio
async def test_topic_safety_cache_miss(fake_llm_topic, mock_task_manager):
    cache = LFUCache(maxsize=10)

    different_messages = [
        {"type": "system", "content": "Different prompt"},
        {"type": "user", "content": "Different question"},
    ]
    cache_entry = {
        "result": {"on_topic": True},
        "llm_stats": {
            "total_tokens": 30,
            "prompt_tokens": 25,
            "completion_tokens": 5,
        },
        "llm_metadata": None,
    }
    cache_key = create_normalized_cache_key(different_messages)
    cache.put(cache_key, cache_entry)

    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    llm_call_info = LLMCallInfo(task="topic_safety_check_input $model=test_model")
    llm_call_info_var.set(llm_call_info)

    result = await topic_safety_check_input(
        llms=fake_llm_topic,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context={"user_message": "What is machine learning?"},
        events=[],
        model_caches={"test_model": cache},
    )

    assert result["on_topic"] is True
    assert cache.size() == 2

    llm_call_info = llm_call_info_var.get()
    assert llm_call_info.from_cache is False


@pytest.mark.asyncio
async def test_topic_safety_without_cache(fake_llm_topic, mock_task_manager):
    result = await topic_safety_check_input(
        llms=fake_llm_topic,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context={"user_message": "What is AI?"},
        events=[],
    )

    assert result["on_topic"] is True

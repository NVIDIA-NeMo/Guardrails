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


import pytest

from nemoguardrails.context import llm_call_info_var, llm_stats_var
from nemoguardrails.llm.cache.lfu import LFUCache
from nemoguardrails.llm.cache.utils import (
    CacheEntry,
    LLMMetadataDict,
    LLMStatsDict,
    create_normalized_cache_key,
    extract_llm_metadata_for_cache,
    extract_llm_stats_for_cache,
    get_from_cache_and_restore_stats,
    restore_llm_metadata_from_cache,
    restore_llm_stats_from_cache,
)
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.logging.processing_log import processing_log_var
from nemoguardrails.logging.stats import LLMStats


class TestCacheUtils:
    @pytest.fixture(autouse=True)
    def isolated_llm_call_info_var(self):
        llm_call_info_var.set(None)
        yield
        llm_call_info_var.set(None)

    def test_create_normalized_cache_key_returns_sha256_hash(self):
        key = create_normalized_cache_key("Hello world")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    @pytest.mark.parametrize(
        "prompt",
        [
            "Hello world",
            "",
            "   Hello world   ",
            "Hello      world      test",
            "Hello\t\n\r world",
            "Hello    \n\t  world",
        ],
    )
    def test_create_normalized_cache_key_with_whitespace_normalization(self, prompt):
        key = create_normalized_cache_key(prompt, normalize_whitespace=True)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    @pytest.mark.parametrize(
        "prompt",
        [
            "Hello world",
            "Hello    \n\t  world",
            "   spaces   ",
        ],
    )
    def test_create_normalized_cache_key_without_whitespace_normalization(self, prompt):
        key = create_normalized_cache_key(prompt, normalize_whitespace=False)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    @pytest.mark.parametrize(
        "prompt1,prompt2",
        [
            ("Hello   \n  world", "Hello     world"),
            ("test\t\nstring", "test  string"),
            ("   leading", "leading"),
        ],
    )
    def test_create_normalized_cache_key_consistent_for_same_input(self, prompt1, prompt2):
        key1 = create_normalized_cache_key(prompt1, normalize_whitespace=True)
        key2 = create_normalized_cache_key(prompt2, normalize_whitespace=True)
        assert key1 == key2

    @pytest.mark.parametrize(
        "prompt1,prompt2",
        [
            ("Hello world", "Hello world!"),
            ("test", "testing"),
            ("case", "Case"),
        ],
    )
    def test_create_normalized_cache_key_different_for_different_input(self, prompt1, prompt2):
        key1 = create_normalized_cache_key(prompt1)
        key2 = create_normalized_cache_key(prompt2)
        assert key1 != key2

    def test_create_normalized_cache_key_invalid_type_raises_error(self):
        with pytest.raises(TypeError, match="Invalid type for prompt: int"):
            create_normalized_cache_key(123)  # type: ignore

        with pytest.raises(TypeError, match="Invalid type for prompt: dict"):
            create_normalized_cache_key({"key": "value"})  # type: ignore

    def test_create_normalized_cache_key_list_of_dicts(self):
        messages = [
            {"type": "user", "content": "Hello"},
            {"type": "assistant", "content": "Hi there!"},
        ]
        key = create_normalized_cache_key(messages)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_create_normalized_cache_key_list_of_dicts_order_independent(self):
        messages1 = [
            {"content": "Hello", "role": "user"},
            {"content": "Hi there!", "role": "assistant"},
        ]
        messages2 = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        key1 = create_normalized_cache_key(messages1)
        key2 = create_normalized_cache_key(messages2)
        assert key1 == key2

    def test_create_normalized_cache_key_invalid_list_raises_error(self):
        with pytest.raises(
            TypeError,
            match="All elements in prompt list must be dictionaries",
        ):
            create_normalized_cache_key(["hello", "world"])  # type: ignore

        with pytest.raises(
            TypeError,
            match="All elements in prompt list must be dictionaries",
        ):
            create_normalized_cache_key([{"key": "value"}, "test"])  # type: ignore

        with pytest.raises(
            TypeError,
            match="All elements in prompt list must be dictionaries",
        ):
            create_normalized_cache_key([123, 456])  # type: ignore

    def test_extract_llm_stats_for_cache_with_llm_call_info(self):
        llm_call_info = LLMCallInfo(task="test_task", total_tokens=100, prompt_tokens=50, completion_tokens=50)
        llm_call_info_var.set(llm_call_info)

        stats = extract_llm_stats_for_cache()

        assert stats is not None
        assert stats["total_tokens"] == 100
        assert stats["prompt_tokens"] == 50
        assert stats["completion_tokens"] == 50

        llm_call_info_var.set(None)

    def test_extract_llm_stats_for_cache_without_llm_call_info(self):
        llm_call_info_var.set(None)

        stats = extract_llm_stats_for_cache()

        assert stats is None

    def test_extract_llm_stats_for_cache_with_none_values(self):
        llm_call_info = LLMCallInfo(
            task="test_task",
            total_tokens=None,
            prompt_tokens=None,
            completion_tokens=None,
        )
        llm_call_info_var.set(llm_call_info)

        stats = extract_llm_stats_for_cache()

        assert stats is not None
        assert stats["total_tokens"] == 0
        assert stats["prompt_tokens"] == 0
        assert stats["completion_tokens"] == 0

        llm_call_info_var.set(None)

    def test_restore_llm_stats_from_cache_creates_new_llm_stats(self):
        llm_stats_var.set(None)
        llm_call_info_var.set(None)

        cached_stats: LLMStatsDict = {
            "total_tokens": 100,
            "prompt_tokens": 50,
            "completion_tokens": 50,
        }

        restore_llm_stats_from_cache(cached_stats, cache_read_duration_s=0.01)

        llm_stats = llm_stats_var.get()
        assert llm_stats is not None
        assert llm_stats.get_stat("total_calls") == 1
        assert llm_stats.get_stat("total_time") == 0.01
        assert llm_stats.get_stat("total_tokens") == 100
        assert llm_stats.get_stat("total_prompt_tokens") == 50
        assert llm_stats.get_stat("total_completion_tokens") == 50

        llm_stats_var.set(None)

    def test_restore_llm_stats_from_cache_updates_existing_llm_stats(self):
        llm_stats = LLMStats()
        llm_stats.inc("total_calls", 5)
        llm_stats.inc("total_time", 1.0)
        llm_stats.inc("total_tokens", 200)
        llm_stats_var.set(llm_stats)

        cached_stats: LLMStatsDict = {
            "total_tokens": 100,
            "prompt_tokens": 50,
            "completion_tokens": 50,
        }

        restore_llm_stats_from_cache(cached_stats, cache_read_duration_s=0.5)

        llm_stats = llm_stats_var.get()
        assert llm_stats is not None
        assert llm_stats.get_stat("total_calls") == 6
        assert llm_stats.get_stat("total_time") == 1.5
        assert llm_stats.get_stat("total_tokens") == 300

        llm_stats_var.set(None)

    def test_restore_llm_stats_from_cache_updates_llm_call_info(self):
        llm_call_info = LLMCallInfo(task="test_task")
        llm_call_info_var.set(llm_call_info)
        llm_stats_var.set(None)

        cached_stats: LLMStatsDict = {
            "total_tokens": 100,
            "prompt_tokens": 50,
            "completion_tokens": 50,
        }

        restore_llm_stats_from_cache(cached_stats, cache_read_duration_s=0.02)

        updated_info = llm_call_info_var.get()
        assert updated_info is not None
        assert updated_info.duration == 0.02
        assert updated_info.total_tokens == 100
        assert updated_info.prompt_tokens == 50
        assert updated_info.completion_tokens == 50
        assert updated_info.from_cache is True
        assert updated_info.started_at is not None
        assert updated_info.finished_at is not None

        llm_call_info_var.set(None)
        llm_stats_var.set(None)

    def test_get_from_cache_and_restore_stats_cache_miss(self):
        cache = LFUCache(maxsize=10)
        llm_call_info_var.set(None)
        llm_stats_var.set(None)

        result = get_from_cache_and_restore_stats(cache, "nonexistent_key")

        assert result is None

        llm_call_info_var.set(None)
        llm_stats_var.set(None)

    def test_get_from_cache_and_restore_stats_cache_hit(self):
        cache = LFUCache(maxsize=10)
        cache_entry = {
            "result": {"allowed": True, "policy_violations": []},
            "llm_stats": {
                "total_tokens": 100,
                "prompt_tokens": 50,
                "completion_tokens": 50,
            },
            "llm_metadata": None,
        }
        cache.put("test_key", cache_entry)

        llm_call_info = LLMCallInfo(task="test_task")
        llm_call_info_var.set(llm_call_info)
        llm_stats_var.set(None)

        result = get_from_cache_and_restore_stats(cache, "test_key")

        assert result is not None
        assert result == {"allowed": True, "policy_violations": []}

        llm_stats = llm_stats_var.get()
        assert llm_stats is not None
        assert llm_stats.get_stat("total_calls") == 1
        assert llm_stats.get_stat("total_tokens") == 100

        updated_info = llm_call_info_var.get()
        assert updated_info is not None
        assert updated_info.from_cache is True

        llm_call_info_var.set(None)
        llm_stats_var.set(None)

    def test_get_from_cache_and_restore_stats_without_llm_stats(self):
        cache = LFUCache(maxsize=10)
        cache_entry = {
            "result": {"allowed": False, "policy_violations": ["policy1"]},
            "llm_stats": None,
            "llm_metadata": None,
        }
        cache.put("test_key", cache_entry)

        llm_call_info_var.set(None)
        llm_stats_var.set(None)

        result = get_from_cache_and_restore_stats(cache, "test_key")

        assert result is not None
        assert result == {"allowed": False, "policy_violations": ["policy1"]}

        llm_call_info_var.set(None)
        llm_stats_var.set(None)

    def test_get_from_cache_and_restore_stats_with_processing_log(self):
        cache = LFUCache(maxsize=10)
        cache_entry = {
            "result": {"allowed": True, "policy_violations": []},
            "llm_stats": {
                "total_tokens": 80,
                "prompt_tokens": 60,
                "completion_tokens": 20,
            },
            "llm_metadata": None,
        }
        cache.put("test_key", cache_entry)

        llm_call_info = LLMCallInfo(task="test_task")
        llm_call_info_var.set(llm_call_info)
        llm_stats_var.set(None)

        processing_log = []
        processing_log_var.set(processing_log)  # type: ignore

        result = get_from_cache_and_restore_stats(cache, "test_key")

        assert result is not None
        assert result == {"allowed": True, "policy_violations": []}

        retrieved_log = processing_log_var.get()
        assert retrieved_log is not None
        assert len(retrieved_log) == 1
        assert retrieved_log[0]["type"] == "llm_call_info"
        assert "timestamp" in retrieved_log[0]
        assert "data" in retrieved_log[0]
        assert retrieved_log[0]["data"] == llm_call_info

        llm_call_info_var.set(None)
        llm_stats_var.set(None)
        processing_log_var.set(None)

    def test_get_from_cache_and_restore_stats_without_processing_log(self):
        cache = LFUCache(maxsize=10)
        cache_entry = {
            "result": {"allowed": True, "policy_violations": []},
            "llm_stats": {
                "total_tokens": 50,
                "prompt_tokens": 30,
                "completion_tokens": 20,
            },
            "llm_metadata": None,
        }
        cache.put("test_key", cache_entry)

        llm_call_info = LLMCallInfo(task="test_task")
        llm_call_info_var.set(llm_call_info)
        llm_stats_var.set(None)
        processing_log_var.set(None)

        result = get_from_cache_and_restore_stats(cache, "test_key")

        assert result is not None
        assert result == {"allowed": True, "policy_violations": []}

        llm_call_info_var.set(None)
        llm_stats_var.set(None)

    def test_extract_llm_metadata_for_cache_with_model_info(self):
        llm_call_info = LLMCallInfo(task="test_task", llm_model_name="gpt-4", llm_provider_name="openai")
        llm_call_info_var.set(llm_call_info)

        metadata = extract_llm_metadata_for_cache()

        assert metadata is not None
        assert metadata["model_name"] == "gpt-4"
        assert metadata["provider_name"] == "openai"

        llm_call_info_var.set(None)

    def test_extract_llm_metadata_for_cache_without_llm_call_info(self):
        llm_call_info_var.set(None)

        metadata = extract_llm_metadata_for_cache()

        assert metadata is None

    def test_extract_llm_metadata_for_cache_with_defaults(self):
        llm_call_info = LLMCallInfo(task="test_task")
        llm_call_info_var.set(llm_call_info)

        metadata = extract_llm_metadata_for_cache()

        assert metadata is not None
        assert metadata["model_name"] == "unknown"
        assert metadata["provider_name"] == "unknown"

        llm_call_info_var.set(None)

    def test_restore_llm_metadata_from_cache(self):
        llm_call_info = LLMCallInfo(task="test_task")
        llm_call_info_var.set(llm_call_info)

        cached_metadata: LLMMetadataDict = {
            "model_name": "nvidia/llama-3.1-nemoguard-8b-content-safety",
            "provider_name": "nim",
        }

        restore_llm_metadata_from_cache(cached_metadata)

        updated_info = llm_call_info_var.get()
        assert updated_info is not None
        assert updated_info.llm_model_name == "nvidia/llama-3.1-nemoguard-8b-content-safety"
        assert updated_info.llm_provider_name == "nim"

        llm_call_info_var.set(None)

    def test_get_from_cache_and_restore_stats_with_metadata(self):
        cache = LFUCache(maxsize=10)
        cache_entry: CacheEntry = {
            "result": {"allowed": True, "policy_violations": []},
            "llm_stats": {
                "total_tokens": 100,
                "prompt_tokens": 50,
                "completion_tokens": 50,
            },
            "llm_metadata": {
                "model_name": "gpt-4-turbo",
                "provider_name": "openai",
            },
        }
        cache.put("test_key", cache_entry)

        llm_call_info = LLMCallInfo(task="test_task")
        llm_call_info_var.set(llm_call_info)
        llm_stats_var.set(None)

        result = get_from_cache_and_restore_stats(cache, "test_key")

        assert result is not None
        assert result == {"allowed": True, "policy_violations": []}

        updated_info = llm_call_info_var.get()
        assert updated_info is not None
        assert updated_info.from_cache is True
        assert updated_info.llm_model_name == "gpt-4-turbo"
        assert updated_info.llm_provider_name == "openai"

        llm_call_info_var.set(None)
        llm_stats_var.set(None)

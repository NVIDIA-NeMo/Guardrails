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

import hashlib
import json
import re
from time import time
from typing import TYPE_CHECKING, List, Optional, TypedDict, Union

from nemoguardrails.context import llm_call_info_var, llm_stats_var
from nemoguardrails.logging.processing_log import processing_log_var
from nemoguardrails.logging.stats import LLMStats

if TYPE_CHECKING:
    from nemoguardrails.llm.cache.interface import CacheInterface

PROMPT_PATTERN_WHITESPACES = re.compile(r"\s+")


class LLMStatsDict(TypedDict):
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int


class CacheEntry(TypedDict):
    result: dict
    llm_stats: Optional[LLMStatsDict]


def create_normalized_cache_key(
    prompt: Union[str, List[dict]], normalize_whitespace: bool = True
) -> str:
    """
    Create a normalized, hashed cache key from a prompt.

    This function generates a deterministic cache key by normalizing the prompt
    and applying SHA-256 hashing. The normalization ensures that semantically
    equivalent prompts produce the same cache key.

    Args:
        prompt: The prompt to be cached. Can be:
            - str: A single prompt string (for completion models)
            - List[dict]: A list of message dictionaries for chat models
              (e.g., [{"type": "user", "content": "Hello"}])
              Note: render_task_prompt() returns Union[str, List[dict]]
        normalize_whitespace: Whether to normalize whitespace characters.
            When True, collapses all whitespace sequences to single spaces and
            strips leading/trailing whitespace. Default: True

    Returns:
        A SHA-256 hex digest string (64 characters) suitable for use as a cache key

    Raises:
        TypeError: If prompt is not a str or List[dict]

    Examples:
        >>> create_normalized_cache_key("Hello world")
        '64ec88ca00b268e5ba1a35678a1b5316d212f4f366b2477232534a8aeca37f3c'

        >>> create_normalized_cache_key([{"type": "user", "content": "Hello"}])
        'b2f5c9d8e3a1f7b6c4d2e5f8a9c1d3e5f7b9a2c4d6e8f1a3b5c7d9e2f4a6b8'
    """
    if isinstance(prompt, str):
        prompt_str = prompt
    elif isinstance(prompt, list):
        if not all(isinstance(p, dict) for p in prompt):
            raise TypeError(
                f"All elements in prompt list must be dictionaries (messages). "
                f"Got types: {[type(p).__name__ for p in prompt]}"
            )
        prompt_str = json.dumps(prompt, sort_keys=True)
    else:
        raise TypeError(
            f"Invalid type for prompt: {type(prompt).__name__}. "
            f"Expected str or List[dict]."
        )

    if normalize_whitespace:
        prompt_str = PROMPT_PATTERN_WHITESPACES.sub(" ", prompt_str).strip()

    return hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()


def restore_llm_stats_from_cache(
    cached_stats: LLMStatsDict, cache_read_duration: float
) -> None:
    llm_stats = llm_stats_var.get()
    if llm_stats is None:
        llm_stats = LLMStats()
        llm_stats_var.set(llm_stats)

    llm_stats.inc("total_calls")
    llm_stats.inc("total_time", cache_read_duration)
    llm_stats.inc("total_tokens", cached_stats.get("total_tokens", 0))
    llm_stats.inc("total_prompt_tokens", cached_stats.get("prompt_tokens", 0))
    llm_stats.inc("total_completion_tokens", cached_stats.get("completion_tokens", 0))

    llm_call_info = llm_call_info_var.get()
    if llm_call_info:
        llm_call_info.duration = cache_read_duration
        llm_call_info.total_tokens = cached_stats.get("total_tokens", 0)
        llm_call_info.prompt_tokens = cached_stats.get("prompt_tokens", 0)
        llm_call_info.completion_tokens = cached_stats.get("completion_tokens", 0)
        llm_call_info.from_cache = True
        llm_call_info.started_at = time() - cache_read_duration
        llm_call_info.finished_at = time()


def extract_llm_stats_for_cache() -> Optional[LLMStatsDict]:
    llm_call_info = llm_call_info_var.get()
    if llm_call_info:
        return {
            "total_tokens": llm_call_info.total_tokens or 0,
            "prompt_tokens": llm_call_info.prompt_tokens or 0,
            "completion_tokens": llm_call_info.completion_tokens or 0,
        }
    return None


def get_from_cache_and_restore_stats(
    cache: "CacheInterface", cache_key: str
) -> Optional[dict]:
    cached_entry = cache.get(cache_key)
    if cached_entry is None:
        return None

    cache_read_start = time()
    final_result = cached_entry["result"]
    cached_stats = cached_entry.get("llm_stats")
    cache_read_duration = time() - cache_read_start

    if cached_stats:
        restore_llm_stats_from_cache(cached_stats, cache_read_duration)

    processing_log = processing_log_var.get()
    if processing_log is not None:
        llm_call_info = llm_call_info_var.get()
        if llm_call_info:
            processing_log.append(
                {
                    "type": "llm_call_info",
                    "timestamp": time(),
                    "data": llm_call_info,
                }
            )

    return final_result

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
from time import time
from typing import Dict, Optional

from langchain_core.language_models.llms import BaseLLM

from nemoguardrails.actions.actions import action
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.cache import CacheInterface
from nemoguardrails.cache.utils import create_normalized_cache_key
from nemoguardrails.context import llm_call_info_var, llm_stats_var
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.logging.processing_log import processing_log_var
from nemoguardrails.logging.stats import LLMStats

log = logging.getLogger(__name__)


def _restore_llm_stats_from_cache(
    cached_stats: dict, cache_read_duration: float
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


def _extract_llm_stats_for_cache() -> Optional[dict]:
    llm_call_info = llm_call_info_var.get()
    if llm_call_info:
        return {
            "total_tokens": llm_call_info.total_tokens or 0,
            "prompt_tokens": llm_call_info.prompt_tokens or 0,
            "completion_tokens": llm_call_info.completion_tokens or 0,
        }
    return None


@action()
async def content_safety_check_input(
    llms: Dict[str, BaseLLM],
    llm_task_manager: LLMTaskManager,
    model_name: Optional[str] = None,
    context: Optional[dict] = None,
    model_caches: Optional[Dict[str, CacheInterface]] = None,
    **kwargs,
) -> dict:
    _MAX_TOKENS = 3
    user_input: str = ""

    if context is not None:
        user_input = context.get("user_message", "")
        model_name = model_name or context.get("model", None)

    if model_name is None:
        error_msg = (
            "Model name is required for content safety check, "
            "please provide it as an argument in the config.yml. "
            "e.g. content safety check input $model=llama_guard"
        )
        raise ValueError(error_msg)

    llm = llms.get(model_name, None)

    if llm is None:
        error_msg = (
            f"Model {model_name} not found in the list of available models for content safety check. "
            "Please provide a valid model name."
        )
        raise ValueError(error_msg)

    task = f"content_safety_check_input $model={model_name}"

    check_input_prompt = llm_task_manager.render_task_prompt(
        task=task,
        context={
            "user_input": user_input,
        },
    )

    stop = llm_task_manager.get_stop_tokens(task=task)
    max_tokens = llm_task_manager.get_max_tokens(task=task)

    llm_call_info_var.set(LLMCallInfo(task=task))

    max_tokens = max_tokens or _MAX_TOKENS

    cache = model_caches.get(model_name) if model_caches else None

    if cache:
        cache_key = create_normalized_cache_key(check_input_prompt)
        cached_entry = cache.get(cache_key)
        if cached_entry is not None:
            log.debug(f"Content safety cache hit for model '{model_name}'")

            cache_read_start = time()
            final_result = cached_entry["result"]
            cached_stats = cached_entry.get("llm_stats")
            cache_read_duration = time() - cache_read_start

            if cached_stats:
                _restore_llm_stats_from_cache(cached_stats, cache_read_duration)

            processing_log = processing_log_var.get()
            if processing_log:
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

    result = await llm_call(
        llm,
        check_input_prompt,
        stop=stop,
        llm_params={"temperature": 1e-20, "max_tokens": max_tokens},
    )

    result = llm_task_manager.parse_task_output(task, output=result)

    is_safe, *violated_policies = result

    final_result = {"allowed": is_safe, "policy_violations": violated_policies}

    if cache:
        cache_entry = {
            "result": final_result,
            "llm_stats": _extract_llm_stats_for_cache(),
        }
        cache.put(cache_key, cache_entry)
        log.debug(f"Content safety result cached for model '{model_name}'")

    return final_result


def content_safety_check_output_mapping(result: dict) -> bool:
    """
    Mapping function for content_safety_check_output.

    Assumes result is a dictionary with:
      - "allowed": a boolean where True means the content is safe.
      - "policy_violations": a list of policies that were violated (optional in the mapping logic).

    Returns:
        True if the content should be blocked (i.e. allowed is False),
        False if the content is safe.
    """
    allowed = result.get("allowed", True)
    return not allowed


@action(output_mapping=content_safety_check_output_mapping)
async def content_safety_check_output(
    llms: Dict[str, BaseLLM],
    llm_task_manager: LLMTaskManager,
    model_name: Optional[str] = None,
    context: Optional[dict] = None,
    **kwargs,
) -> dict:
    _MAX_TOKENS = 3
    user_input: str = ""
    bot_response: str = ""

    if context is not None:
        user_input = context.get("user_message", "")
        bot_response = context.get("bot_message", "")
        model_name = model_name or context.get("model", None)

    if model_name is None:
        error_msg = (
            "Model name is required for content safety check, "
            "please provide it as an argument in the config.yml. "
            "e.g. flow content safety (model_name='llama_guard')"
        )
        raise ValueError(error_msg)

    llm = llms.get(model_name, None)

    if llm is None:
        error_msg = (
            f"Model {model_name} not found in the list of available models for content safety check. "
            "Please provide a valid model name."
        )
        raise ValueError(error_msg)

    task = f"content_safety_check_output $model={model_name}"

    check_output_prompt = llm_task_manager.render_task_prompt(
        task=task,
        context={
            "user_input": user_input,
            "bot_response": bot_response,
        },
    )
    stop = llm_task_manager.get_stop_tokens(task=task)
    max_tokens = llm_task_manager.get_max_tokens(task=task)

    max_tokens = max_tokens or _MAX_TOKENS

    llm_call_info_var.set(LLMCallInfo(task=task))

    result = await llm_call(
        llm,
        check_output_prompt,
        stop=stop,
        llm_params={"temperature": 1e-20, "max_tokens": max_tokens},
    )

    result = llm_task_manager.parse_task_output(task, output=result)

    is_safe, *violated_policies = result

    return {"allowed": is_safe, "policy_violations": violated_policies}

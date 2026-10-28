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

import logging
from typing import Any, Dict, Optional, cast

from nemoguardrails.actions.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.library.utils.language_detection import (
    DEFAULT_REFUSAL_MESSAGES,  # noqa: F401  (re-exported for backward compatibility)
    SUPPORTED_LANGUAGES,
)
from nemoguardrails.library.utils.language_detection import (
    detect_language as _detect_language,
)
from nemoguardrails.library.utils.language_detection import (
    get_refusal_message as _get_refusal_message,
)
from nemoguardrails.llm.cache import CacheInterface
from nemoguardrails.llm.cache.utils import (
    CacheEntry,
    create_normalized_cache_key,
    extract_llm_metadata_for_cache,
    extract_llm_stats_for_cache,
    get_from_cache_and_restore_stats,
)
from nemoguardrails.llm.call import llm_call, warn_if_truncated
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.types import LLMModel

log = logging.getLogger(__name__)


def _get_reasoning_enabled(llm_task_manager: LLMTaskManager) -> bool:
    return cast(Any, llm_task_manager.config).rails.config.content_safety.reasoning.enabled


@action()
async def content_safety_check_input(
    llms: Dict[str, LLMModel],
    llm_task_manager: LLMTaskManager,
    model_name: Optional[str] = None,
    context: Optional[dict] = None,
    user_message: Optional[str] = None,
    model_caches: Optional[Dict[str, CacheInterface]] = None,
    **kwargs,
) -> RailOutcome:
    _MAX_TOKENS = 1024
    user_input = user_message or ""

    if context is not None:
        if user_message is None:
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
            "reasoning_enabled": _get_reasoning_enabled(llm_task_manager),
        },
    )

    stop = llm_task_manager.get_stop_tokens(task=task)
    max_tokens = llm_task_manager.get_max_tokens(task=task)

    llm_call_info_var.set(LLMCallInfo(task=task))

    max_tokens = max_tokens or _MAX_TOKENS

    cache = model_caches.get(model_name) if model_caches else None

    if cache:
        cache_key = create_normalized_cache_key(check_input_prompt)
        cached_result = get_from_cache_and_restore_stats(cache, cache_key)
        if cached_result is not None:
            log.debug(f"Content safety cache hit for model '{model_name}'")
            return cached_result

    llm_response = await llm_call(
        llm,
        check_input_prompt,
        stop=stop,
        llm_params={"temperature": 1e-20, "max_tokens": max_tokens},
    )
    warn_if_truncated(llm_response, task)
    result = llm_task_manager.parse_task_output(task, output=llm_response.content)  # type: ignore[reportArgumentType]

    is_safe, *violated_policies = result

    final_result = (
        RailOutcome.allow(metadata={"policy_violations": violated_policies})
        if is_safe
        else RailOutcome.block(metadata={"policy_violations": violated_policies})
    )

    if cache:
        cache_key = create_normalized_cache_key(check_input_prompt)
        cache_entry: CacheEntry = {
            "result": final_result,
            "llm_stats": extract_llm_stats_for_cache(),
            "llm_metadata": extract_llm_metadata_for_cache(),
        }
        cache.put(cache_key, cache_entry)
        log.debug(f"Content safety result cached for model '{model_name}'")

    return final_result


@action()
async def content_safety_check_output(
    llms: Dict[str, LLMModel],
    llm_task_manager: LLMTaskManager,
    model_name: Optional[str] = None,
    context: Optional[dict] = None,
    user_message: Optional[str] = None,
    bot_message: Optional[str] = None,
    model_caches: Optional[Dict[str, CacheInterface]] = None,
    **kwargs,
) -> RailOutcome:
    _MAX_TOKENS = 1024
    user_input = user_message or ""
    bot_response = bot_message or ""

    if context is not None:
        if user_message is None:
            user_input = context.get("user_message", "")
        if bot_message is None:
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
            "reasoning_enabled": _get_reasoning_enabled(llm_task_manager),
        },
    )

    stop = llm_task_manager.get_stop_tokens(task=task)
    max_tokens = llm_task_manager.get_max_tokens(task=task)

    llm_call_info_var.set(LLMCallInfo(task=task))

    max_tokens = max_tokens or _MAX_TOKENS

    cache = model_caches.get(model_name) if model_caches else None

    if cache:
        cache_key = create_normalized_cache_key(check_output_prompt)
        cached_result = get_from_cache_and_restore_stats(cache, cache_key)
        if cached_result is not None:
            log.debug(f"Content safety output cache hit for model '{model_name}'")
            return cached_result

    llm_response = await llm_call(
        llm,
        check_output_prompt,
        stop=stop,
        llm_params={"temperature": 1e-20, "max_tokens": max_tokens},
    )
    warn_if_truncated(llm_response, task)
    result = llm_task_manager.parse_task_output(task, output=llm_response.content)  # type: ignore[reportArgumentType]

    is_safe, *violated_policies = result

    final_result = (
        RailOutcome.allow(metadata={"policy_violations": violated_policies})
        if is_safe
        else RailOutcome.block(metadata={"policy_violations": violated_policies})
    )

    if cache:
        cache_key = create_normalized_cache_key(check_output_prompt)
        cache_entry: CacheEntry = {
            "result": final_result,
            "llm_stats": extract_llm_stats_for_cache(),
            "llm_metadata": extract_llm_metadata_for_cache(),
        }
        cache.put(cache_key, cache_entry)
        log.debug(f"Content safety output result cached for model '{model_name}'")

    return final_result


@action()
async def detect_language(
    context: Optional[dict] = None,
    config: Optional[Any] = None,
) -> dict:
    user_message = ""
    if context is not None:
        user_message = context.get("user_message", "")

    custom_messages = None
    if config is not None:
        multilingual_config = (
            config.rails.config.content_safety.multilingual
            if hasattr(config, "rails")
            and hasattr(config.rails, "config")
            and hasattr(config.rails.config, "content_safety")
            and hasattr(config.rails.config.content_safety, "multilingual")
            else None
        )
        if multilingual_config:
            custom_messages = multilingual_config.refusal_messages

    lang = _detect_language(user_message) or "en"

    if lang not in SUPPORTED_LANGUAGES:
        lang = "en"

    refusal_message = _get_refusal_message(lang, custom_messages)

    return {
        "language": lang,
        "refusal_message": refusal_message,
    }

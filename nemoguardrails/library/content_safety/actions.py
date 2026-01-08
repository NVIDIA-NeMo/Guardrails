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
from typing import Dict, FrozenSet, Optional

from langchain_core.language_models import BaseLLM

from nemoguardrails.actions.actions import action
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.llm.cache import CacheInterface
from nemoguardrails.llm.cache.utils import (
    CacheEntry,
    create_normalized_cache_key,
    extract_llm_metadata_for_cache,
    extract_llm_stats_for_cache,
    get_from_cache_and_restore_stats,
)
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.logging.explain import LLMCallInfo

log = logging.getLogger(__name__)


def _get_reasoning_enabled(llm_task_manager: LLMTaskManager) -> bool:
    return llm_task_manager.config.rails.config.content_safety.reasoning.enabled


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
        cache_key = create_normalized_cache_key(check_input_prompt)
        cache_entry: CacheEntry = {
            "result": final_result,
            "llm_stats": extract_llm_stats_for_cache(),
            "llm_metadata": extract_llm_metadata_for_cache(),
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
    model_caches: Optional[Dict[str, CacheInterface]] = None,
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

    result = await llm_call(
        llm,
        check_output_prompt,
        stop=stop,
        llm_params={"temperature": 1e-20, "max_tokens": max_tokens},
    )

    result = llm_task_manager.parse_task_output(task, output=result)

    is_safe, *violated_policies = result

    final_result = {"allowed": is_safe, "policy_violations": violated_policies}

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


SUPPORTED_LANGUAGES: FrozenSet[str] = frozenset({"en", "es", "zh", "de", "fr", "hi", "ja", "ar", "th"})

DEFAULT_REFUSAL_MESSAGES: Dict[str, str] = {
    "en": "I'm sorry, I can't respond to that.",
    "es": "Lo siento, no puedo responder a eso.",
    "zh": "抱歉，我无法回应。",
    "de": "Es tut mir leid, darauf kann ich nicht antworten.",
    "fr": "Je suis désolé, je ne peux pas répondre à cela.",
    "hi": "मुझे खेद है, मैं इसका जवाब नहीं दे सकता।",
    "ja": "申し訳ありませんが、それには回答できません。",
    "ar": "عذراً، لا أستطيع الرد على ذلك.",
    "th": "ขออภัย ฉันไม่สามารถตอบได้",
}


def _detect_language(text: str) -> Optional[str]:
    try:
        from fast_langdetect import detect

        result = detect(text, k=1)
        if result and len(result) > 0:
            return result[0].get("lang")
        return None
    except ImportError:
        log.warning("fast-langdetect not installed, skipping")
        return None
    except Exception as e:
        log.warning(f"fast-langdetect detection failed: {e}")
        return None


def _get_refusal_message(lang: str, custom_messages: Optional[Dict[str, str]]) -> str:
    if custom_messages and lang in custom_messages:
        return custom_messages[lang]
    if lang in DEFAULT_REFUSAL_MESSAGES:
        return DEFAULT_REFUSAL_MESSAGES[lang]
    if custom_messages and "en" in custom_messages:
        return custom_messages["en"]
    return DEFAULT_REFUSAL_MESSAGES["en"]


@action()
async def detect_language(
    context: Optional[dict] = None,
    config: Optional[dict] = None,
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

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

import asyncio
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

# get_cpu_executor provides a shared thread-pool used to offload CPU-bound
# work (e.g. language detection) so the async event loop is never blocked.
from nemoguardrails.rails.llm.dag_scheduler import get_cpu_executor

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
    # Safety models typically return a single token ("safe"/"unsafe"); cap at 3
    # to avoid runaway generation whilst still allowing short category labels.
    _MAX_TOKENS = 3
    user_input: str = ""

    if context is not None:
        user_input = context.get("user_message", "")
        # Allow the model name to be overridden via the runtime context.
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

    # Build a task identifier that includes the model so the task manager
    # can resolve model-specific prompt templates and stop tokens.
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

    # Attach tracing metadata so downstream logging can attribute this LLM call.
    llm_call_info_var.set(LLMCallInfo(task=task))

    # Fall back to the local cap when the task definition has no max_tokens.
    max_tokens = max_tokens or _MAX_TOKENS

    cache = model_caches.get(model_name) if model_caches else None

    # Attempt a cache lookup before making the (potentially expensive) LLM call.
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
        # Near-zero temperature for deterministic safety judgements.
        llm_params={"temperature": 1e-20, "max_tokens": max_tokens},
    )

    # The task manager's output parser converts raw text into a structured
    # tuple: (is_safe: bool, *violated_policy_names).
    result = llm_task_manager.parse_task_output(task, output=result)

    # Unpack: first element is the boolean verdict, remainder is zero or
    # more policy identifiers that were violated.
    is_safe, *violated_policies = result

    final_result = {"allowed": is_safe, "policy_violations": violated_policies}

    # Persist the result so identical prompts are served from cache next time.
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
    # Default to True (safe) when the key is absent so we fail open rather
    # than silently blocking legitimate content.
    allowed = result.get("allowed", True)
    # Invert the semantics: the DAG scheduler treats True as "condition met",
    # meaning the content should be *blocked*.  So we return True when
    # allowed is False and vice versa.
    return not allowed


# Register the action with an output_mapping so the DAG scheduler can
# automatically convert the raw dict result into a boolean "should block"
# signal without requiring an extra Colang step.
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
        # Output checking requires both the user's prompt and the bot's reply
        # so the safety model can evaluate the full conversational context.
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


# frozenset is used deliberately: the set of supported language codes is a
# compile-time constant that must never be mutated at runtime.  Using a
# frozenset rather than a plain set makes that invariant explicit and also
# allows the object to be safely shared across threads and used as a dict key
# or set member if needed.
SUPPORTED_LANGUAGES: FrozenSet[str] = frozenset({"en", "es", "zh", "de", "fr", "hi", "ja", "ar", "th"})

# Fallback refusal strings keyed by ISO 639-1 language code.  These are
# returned when the deployer has not provided custom refusal messages in
# config.yml for the detected language.
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
    """Synchronous helper — meant to be called inside a thread-pool executor."""
    try:
        # Lazy import: fast_langdetect is an optional dependency.  Importing
        # at call-time avoids a hard start-up failure when the package is
        # absent, and keeps the module importable for unit tests that don't
        # exercise multilingual behaviour.
        from fast_langdetect import detect

        # k=1 requests only the single most probable language, which is
        # sufficient for choosing a refusal message.
        result = detect(text, k=1)
        if result and len(result) > 0:
            return result[0].get("lang")
        return None
    except ImportError:
        # Gracefully degrade: log a warning and let the caller fall back to
        # English rather than raising.
        log.warning("fast-langdetect not installed, skipping")
        return None
    except Exception as e:
        log.warning(f"fast-langdetect detection failed: {e}")
        return None


def _get_refusal_message(lang: str, custom_messages: Optional[Dict[str, str]]) -> str:
    # Three-tier lookup chain with an ultimate English fallback:
    #   1. Custom message for the detected language (deployer override).
    #   2. Built-in default message for the detected language.
    #   3. Custom English message (deployer override for "en").
    #   4. Built-in English default — guaranteed to exist, so the function
    #      never returns None.
    if custom_messages and lang in custom_messages:
        return custom_messages[lang]
    if lang in DEFAULT_REFUSAL_MESSAGES:
        return DEFAULT_REFUSAL_MESSAGES[lang]
    # If neither custom nor default messages exist for the detected language,
    # fall back to English.  Check the deployer's custom English first.
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

    # Attempt to load deployer-provided per-language refusal messages from
    # the nested config object.  The defensive hasattr chain guards against
    # partially initialised configuration (e.g. in unit tests).
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

    # Language detection via fast_langdetect is CPU-bound (it runs a small
    # classifier model under the hood).  Offloading to run_in_executor
    # prevents it from blocking the async event loop, which would stall all
    # concurrent guardrail evaluations.  The shared CPU executor is obtained
    # via get_cpu_executor() so the thread-pool size is centrally managed.
    loop = asyncio.get_running_loop()
    lang = await loop.run_in_executor(get_cpu_executor(), _detect_language, user_message) or "en"

    # Clamp to the supported set; unsupported codes fall back to English so
    # the user always receives a valid refusal message.
    if lang not in SUPPORTED_LANGUAGES:
        lang = "en"

    refusal_message = _get_refusal_message(lang, custom_messages)

    return {
        "language": lang,
        "refusal_message": refusal_message,
    }

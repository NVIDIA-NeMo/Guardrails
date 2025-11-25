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

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemoguardrails.actions import action
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.library.jailbreak_detection.request import (
    jailbreak_detection_heuristics_request,
    jailbreak_detection_model_request,
    jailbreak_nim_request,
)
from nemoguardrails.llm.cache import CacheInterface
from nemoguardrails.llm.cache.utils import (
    CacheEntry,
    create_normalized_cache_key,
    get_from_cache_and_restore_stats,
)
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.logging.processing_log import processing_log_var

log = logging.getLogger(__name__)


@action()
async def jailbreak_detection_heuristics(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    **kwargs,
) -> bool:
    """Checks the user's prompt to determine if it is attempt to jailbreak the model."""
    jailbreak_config = llm_task_manager.config.rails.config.jailbreak_detection

    jailbreak_api_url = jailbreak_config.server_endpoint
    lp_threshold = jailbreak_config.length_per_perplexity_threshold
    ps_ppl_threshold = jailbreak_config.prefix_suffix_perplexity_threshold

    prompt = context.get("user_message")

    if not jailbreak_api_url:
        from nemoguardrails.library.jailbreak_detection.heuristics.checks import (
            check_jailbreak_length_per_perplexity,
            check_jailbreak_prefix_suffix_perplexity,
        )

        log.warning("No jailbreak detection endpoint set. Running in-process, NOT RECOMMENDED FOR PRODUCTION.")
        lp_check = check_jailbreak_length_per_perplexity(prompt, lp_threshold)
        ps_ppl_check = check_jailbreak_prefix_suffix_perplexity(prompt, ps_ppl_threshold)
        jailbreak = any([lp_check["jailbreak"], ps_ppl_check["jailbreak"]])
        return jailbreak

    jailbreak = await jailbreak_detection_heuristics_request(prompt, jailbreak_api_url, lp_threshold, ps_ppl_threshold)
    if jailbreak is None:
        log.warning("Jailbreak endpoint not set up properly.")
        # If no result, assume not a jailbreak
        return False
    else:
        return jailbreak


@action()
async def jailbreak_detection_model(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    model_caches: Optional[Dict[str, CacheInterface]] = None,
) -> bool:
    """Uses a trained classifier to determine if a user input is a jailbreak attempt"""
    prompt: str = ""
    jailbreak_config = llm_task_manager.config.rails.config.jailbreak_detection

    jailbreak_api_url = jailbreak_config.server_endpoint
    nim_base_url = jailbreak_config.nim_base_url
    nim_classification_path = jailbreak_config.nim_server_endpoint
    nim_auth_token = jailbreak_config.get_api_key()

    if context is not None:
        prompt = context.get("user_message", "")

    # we do this as a hack to treat this action as an LLM call for tracing
    llm_call_info_var.set(LLMCallInfo(task="jailbreak_detection_model"))

    cache = model_caches.get("jailbreak_detection") if model_caches else None

    if cache:
        cache_key = create_normalized_cache_key(prompt)
        cache_read_start = time()
        cached_result = get_from_cache_and_restore_stats(cache, cache_key)
        if cached_result is not None:
            cache_read_duration = time() - cache_read_start
            llm_call_info = llm_call_info_var.get()
            if llm_call_info:
                llm_call_info.from_cache = True
                llm_call_info.duration = cache_read_duration
                llm_call_info.started_at = time() - cache_read_duration
                llm_call_info.finished_at = time()

            log.debug("Jailbreak detection cache hit")
            return cached_result["jailbreak"]

    jailbreak_result = None
    api_start_time = time()

    if not jailbreak_api_url and not nim_base_url:
        from nemoguardrails.library.jailbreak_detection.model_based.checks import (
            check_jailbreak,
        )

        log.warning("No jailbreak detection endpoint set. Running in-process, NOT RECOMMENDED FOR PRODUCTION.")
        try:
            jailbreak = check_jailbreak(prompt=prompt)
            log.info(f"Local model jailbreak detection result: {jailbreak}")
            jailbreak_result = jailbreak["jailbreak"]
        except RuntimeError as e:
            log.error(f"Jailbreak detection model not available: {e}")
            jailbreak_result = False
        except ImportError as e:
            log.error(
                "Failed to import required dependencies for local model. Install scikit-learn and torch, or use NIM-based approach",
                exc_info=e,
            )
            jailbreak_result = False
    else:
        if nim_base_url:
            jailbreak = await jailbreak_nim_request(
                prompt=prompt,
                nim_url=nim_base_url,
                nim_auth_token=nim_auth_token,
                nim_classification_path=nim_classification_path,
            )
        elif jailbreak_api_url:
            jailbreak = await jailbreak_detection_model_request(prompt=prompt, api_url=jailbreak_api_url)

        if jailbreak is None:
            log.warning("Jailbreak endpoint not set up properly.")
            jailbreak_result = False
        else:
            jailbreak_result = jailbreak

    api_duration = time() - api_start_time

    llm_call_info = llm_call_info_var.get()
    if llm_call_info:
        llm_call_info.from_cache = False
        llm_call_info.duration = api_duration
        llm_call_info.started_at = api_start_time
        llm_call_info.finished_at = time()

        processing_log = processing_log_var.get()
        if processing_log is not None:
            processing_log.append(
                {
                    "type": "llm_call_info",
                    "timestamp": time(),
                    "data": llm_call_info,
                }
            )

    if cache:
        from nemoguardrails.llm.cache.utils import extract_llm_metadata_for_cache

        cache_key = create_normalized_cache_key(prompt)
        cache_entry: CacheEntry = {
            "result": {"jailbreak": jailbreak_result},
            "llm_stats": None,
            "llm_metadata": extract_llm_metadata_for_cache(),
        }
        cache.put(cache_key, cache_entry)
        log.debug("Jailbreak detection result cached")

    return jailbreak_result

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
from typing import Optional

from nemoguardrails.actions import action
from nemoguardrails.library.jailbreak_detection.request import (
    jailbreak_detection_heuristics_request,
    jailbreak_detection_model_request,
    jailbreak_nim_request,
)
from nemoguardrails.llm.taskmanager import LLMTaskManager

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

        log.warning(
            "No jailbreak detection endpoint set. Running in-process, NOT RECOMMENDED FOR PRODUCTION."
        )
        lp_check = check_jailbreak_length_per_perplexity(prompt, lp_threshold)
        ps_ppl_check = check_jailbreak_prefix_suffix_perplexity(
            prompt, ps_ppl_threshold
        )
        jailbreak = any([lp_check["jailbreak"], ps_ppl_check["jailbreak"]])
        return jailbreak

    jailbreak = await jailbreak_detection_heuristics_request(
        prompt, jailbreak_api_url, lp_threshold, ps_ppl_threshold
    )
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
) -> bool:
    """Uses a trained classifier to determine if a user input is a jailbreak attempt"""
    prompt: str = ""
    jailbreak_config = llm_task_manager.config.rails.config.jailbreak_detection

    jailbreak_api_url = jailbreak_config.server_endpoint
    nim_url = jailbreak_config.nim_url
    nim_port = jailbreak_config.nim_port
    nim_full_url = getattr(jailbreak_config, "nim_full_url", None)
    nim_auth_token = getattr(jailbreak_config, "nim_auth_token", None)

    log.info(f"NIM configuration: url={nim_url}, port={nim_port}, full_url={nim_full_url}")
    log.info(f"API URL: {jailbreak_api_url}")

    if context is not None:
        prompt = context.get("user_message", "")
        log.info(f"Checking jailbreak for message: {prompt}")

    # First check if we should use NIM
    if nim_url or nim_full_url:
        log.info("Using NIM-based approach")
        jailbreak = await jailbreak_nim_request(
            prompt=prompt,
            nim_url=nim_url,
            nim_port=nim_port,
            nim_full_url=nim_full_url,
            nim_auth_token=nim_auth_token,
        )
        log.info(f"NIM jailbreak detection result: {jailbreak}")
    # Then check if we should use the API endpoint
    elif jailbreak_api_url:
        log.info("Using API endpoint approach")
        jailbreak = await jailbreak_detection_model_request(
            prompt=prompt, api_url=jailbreak_api_url
        )
        log.info(f"API jailbreak detection result: {jailbreak}")
    # Only if neither NIM nor API endpoint is configured, fall back to local model
    else:
        log.info("Falling back to local model approach")
        # Import these only if we need to use the local model
        from nemoguardrails.library.jailbreak_detection.model_based.checks import (
            check_jailbreak,
            initialize_model,
        )
        log.warning(
            "No jailbreak detection endpoint set. Running in-process, NOT RECOMMENDED FOR PRODUCTION."
        )
        try:
            classifier = initialize_model()
            jailbreak = check_jailbreak(prompt=prompt, classifier=classifier)
            log.info(f"Local model jailbreak detection result: {jailbreak}")
            return jailbreak["jailbreak"]
        except ImportError as e:
            log.error(f"Failed to import required dependencies for local model: {e}")
            log.error("Please install scikit-learn and torch, or use NIM-based approach")
            return False

    if jailbreak is None:
        log.warning("Jailbreak endpoint not set up properly.")
        # If no result, assume not a jailbreak
        return False
    else:
        return jailbreak

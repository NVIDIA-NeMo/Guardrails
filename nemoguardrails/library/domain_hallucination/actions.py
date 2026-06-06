# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Guardrails actions for domain hallucination detection (Layer 1 priority)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:
    from nemoguardrails.actions import action
except Exception:

    def action(*_args: Any, **_kwargs: Any):
        def decorator(fn):
            return fn

        return decorator


from nemoguardrails import RailsConfig
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.types import LLMModel

from . import layer1_check, layer2_advanced

logger = logging.getLogger(__name__)


@action(output_mapping=lambda value: not value)
async def self_check_domain_hallucination(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    llm: Optional[LLMModel] = None,
    config: Optional[RailsConfig] = None,
    enable_layer1: bool = True,
    enable_layer2: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Domain hallucination detection using configurable Layer 1 and/or Layer 2.

    Supports two-layer architecture:
    - Layer 1: Fast LLM-based judgment (~100-200ms, zero network)
    - Layer 2: Advanced network verification (optional, DNS/HTTP/TLS)

    Args:
        llm_task_manager: Task manager for rendering prompts
        context: Context dict with bot_message and user_message
        llm: LLM model for inference
        config: Rails configuration
        enable_layer1: If True (default), use Layer 1 LLM judgment
        enable_layer2: If True, use Layer 2 network verification (only if Layer 1 is clean)

    Returns:
        True if output is safe (no hallucinated domains), False if hallucinated.
    """
    context = context or {}
    bot_response = (
        context.get("bot_message") or context.get("assistant_output") or context.get("last_bot_message") or ""
    )
    user_message = context.get("user_message") or context.get("last_user_message") or context.get("user_input") or ""

    if not bot_response:
        return True  # Nothing to check

    # If both layers disabled, fast pass
    if not enable_layer1 and not enable_layer2:
        logger.info("Both Layer 1 and Layer 2 disabled, passing")
        return True

    result = None

    # Layer 1: Fast LLM judgment (if enabled)
    if enable_layer1:
        result = await layer1_check.layer1_check_domain_hallucination(
            bot_response=bot_response,
            user_message=user_message,
            llm_call_func=llm_call,
            llm_task_manager=llm_task_manager,
            config=config,
            llm=llm,
        )
        logger.info(f"Layer 1 result: {result['status']}, is_hallucinated={result['is_hallucinated']}")

        # If Layer 1 found hallucination, return immediately
        if result["is_hallucinated"]:
            return False

    # Layer 2: Network verification (if enabled and Layer 1 passed or was skipped)
    if enable_layer2:
        # Extract entities for Layer 2 if Layer 1 was skipped
        if result is None:
            entities = layer1_check.extract_entities(bot_response)
        else:
            entities = result.get("entities", {})

        result = await layer2_advanced.layer2_check_with_verification(
            bot_response=bot_response,
            user_message=user_message,
            urls=entities.get("urls", []),
            domains=entities.get("domains", []),
            github_repos=entities.get("github_repos", []),
            llm_call_func=llm_call,
            llm_task_manager=llm_task_manager,
            config=config,
        )
        logger.info(f"Layer 2 result: {result['status']}, is_hallucinated={result['is_hallucinated']}")

        return not result["is_hallucinated"]

    # Return Layer 1 result if Layer 2 was not enabled
    return not result["is_hallucinated"]

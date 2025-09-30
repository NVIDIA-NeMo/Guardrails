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
from typing import Optional

from langchain_core.language_models.llms import BaseLLM

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.llm.types import Task
from nemoguardrails.logging.explain import LLMCallInfo

log = logging.getLogger(__name__)


def mapping_self_check_facts(result: float) -> bool:
    """
    Mapping for self_check_facts.

    Expects result to be a numeric score (float) representing the factual accuracy.
    Returns True (i.e. block the output) if the score is below 0.5.
    """
    THRESHOLD = 0.5
    return result < THRESHOLD


@action(output_mapping=mapping_self_check_facts)
async def self_check_facts(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    llm: Optional[BaseLLM] = None,
    config: Optional[RailsConfig] = None,
    **kwargs,
):
    """Checks the facts for the bot response by appropriately prompting the base llm."""
    _MAX_TOKENS = 3
    evidence = context.get("relevant_chunks", [])
    response = context.get("bot_message")

    if not evidence:
        # If there is no evidence, we always return true
        return True
    task = Task.SELF_CHECK_FACTS
    prompt = llm_task_manager.render_task_prompt(
        task=task,
        context={
            "evidence": evidence,
            "response": response,
        },
    )
    stop = llm_task_manager.get_stop_tokens(task=task)
    max_tokens = llm_task_manager.get_max_tokens(task=task)
    max_tokens = max_tokens or _MAX_TOKENS

    # Initialize the LLMCallInfo object
    llm_call_info_var.set(LLMCallInfo(task=task.value))

    response = await llm_call(
        llm,
        prompt,
        stop=stop,
        llm_params={"temperature": config.lowest_temperature, "max_tokens": max_tokens},
    )

    if llm_task_manager.has_output_parser(task):
        result = llm_task_manager.parse_task_output(task, output=response)
    else:
        result = llm_task_manager.parse_task_output(
            task, output=response, forced_output_parser="is_content_safe"
        )

    is_not_safe = result[0]

    result = float(not is_not_safe)
    return result

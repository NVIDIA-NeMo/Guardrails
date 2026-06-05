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
from typing import Optional

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.llm.utils import llm_call, warn_if_truncated
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.llm.types import Task
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.types import LLMModel

log = logging.getLogger(__name__)

FACT_CHECK_THRESHOLD = 0.5


def _fact_check_outcome(accuracy: float) -> RailOutcome:
    if accuracy < FACT_CHECK_THRESHOLD:
        return RailOutcome.block(accuracy=accuracy)
    return RailOutcome.allow(accuracy=accuracy)


@action()
async def self_check_facts(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    llm: Optional[LLMModel] = None,
    config: Optional[RailsConfig] = None,
    **kwargs,
) -> RailOutcome:
    """Checks the facts for the bot response by appropriately prompting the base llm."""
    _MAX_TOKENS = 1024
    context = context or {}
    evidence = context.get("relevant_chunks", [])
    response = context.get("bot_message")

    if not evidence:
        return _fact_check_outcome(1.0)
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
    temperature = config.lowest_temperature if config is not None else 0.0

    llm_call_info_var.set(LLMCallInfo(task=task.value))

    llm_response = await llm_call(
        llm,
        prompt,
        stop=stop,
        llm_params={"temperature": temperature, "max_tokens": max_tokens},
    )
    if warn_if_truncated(llm_response, task.value):
        return _fact_check_outcome(0.0)
    response = llm_response.content

    if llm_task_manager.has_output_parser(task):
        result = llm_task_manager.parse_task_output(task, output=response)
    else:
        result = llm_task_manager.parse_task_output(task, output=response, forced_output_parser="is_content_safe")

    is_not_safe = bool(result[0])

    result = float(not is_not_safe)
    return _fact_check_outcome(result)

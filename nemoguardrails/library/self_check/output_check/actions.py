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
from typing import Dict, List, Optional

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.library.self_check.utils import (
    SELF_CHECK_OUTPUT_DEFAULT_TASK,
    SELF_CHECK_OUTPUT_FLOW,
    SELF_CHECK_OUTPUT_VARIANT_PARAM,
    resolve_self_check_task,
    run_self_check_task,
)
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.types import LLMModel

log = logging.getLogger(__name__)

DEFAULT_TASK = SELF_CHECK_OUTPUT_DEFAULT_TASK


def _self_check_outcome(allowed: bool) -> RailOutcome:
    return RailOutcome.allow() if allowed else RailOutcome.block()


@action(is_system_action=True)
async def self_check_output(
    llms: Dict[str, LLMModel],
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    # LLMRails injects Colang events only to recover variants for legacy 1.0
    # flows, including parallel rails. Manifests bind `variant` instead so
    # engine-neutral callers do not provide or synthesize events.
    events: Optional[List[dict]] = None,
    llm: Optional[LLMModel] = None,
    config: Optional[RailsConfig] = None,
    variant: Optional[str] = None,
    **kwargs,
) -> RailOutcome:
    """Checks if the output from the bot.

    Prompt the LLM, using the `self_check_output` task prompt, to determine if the output
    from the bot should be allowed or not.

    The LLM call should return "yes" if the output is bad and should be blocked
    (this is consistent with self_check_input_prompt).

    Returns:
        RailOutcome.allow() if the output should be allowed, RailOutcome.block() otherwise.
    """

    context = context or {}
    bot_response = context.get("bot_message")
    user_input = context.get("user_message")
    bot_thinking = context.get("bot_thinking")

    task = resolve_self_check_task(
        variant,
        context,
        events,
        triggered_rail_key="triggered_output_rail",
        start_rail_event_type="StartOutputRail",
        flow_id=SELF_CHECK_OUTPUT_FLOW,
        variant_param=SELF_CHECK_OUTPUT_VARIANT_PARAM,
        default_task=DEFAULT_TASK,
    )

    if not bot_response:
        return _self_check_outcome(False)

    is_safe, response = await run_self_check_task(
        task=task,
        prompt_context={
            "user_input": user_input,
            "bot_response": bot_response,
            "bot_thinking": bot_thinking,
        },
        llms=llms,
        default_task=DEFAULT_TASK,
        main_llm=llm,
        llm_task_manager=llm_task_manager,
        lowest_temperature=config.lowest_temperature if config is not None else 0.0,
    )

    if task == DEFAULT_TASK:
        log.info(f"Output self-checking result is: `{response}`.")
    else:
        log.info(f"Output self-checking result for task={task} is: `{response}`.")

    return _self_check_outcome(is_safe)

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
from typing import Optional

from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.llm.utils import (
    get_multiline_response,
    llm_call,
    strip_quotes,
)
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.llm.types import Task
from nemoguardrails.logging.explain import LLMCallInfo

log = logging.getLogger(__name__)

HALLUCINATION_NUM_EXTRA_RESPONSES = 2


@action(output_mapping=lambda value: value)
async def self_check_hallucination(
    llm: BaseLLM,
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    use_llm_checking: bool = True,
    config: Optional[RailsConfig] = None,
    **kwargs,
):
    """Checks if the last bot response is a hallucination by checking multiple completions for self-consistency.

    :return: True if hallucination is detected, False otherwise.
    """
    bot_response = context.get("bot_message")
    last_bot_prompt_string = context.get("_last_bot_prompt")

    if bot_response and last_bot_prompt_string:
        num_responses = HALLUCINATION_NUM_EXTRA_RESPONSES

        last_bot_prompt = PromptTemplate(template="{text}", input_variables=["text"])
        formatted_prompt = last_bot_prompt.format(text=last_bot_prompt_string)

        async def _generate_extra_response(index: int) -> Optional[str]:
            llm_call_info_var.set(LLMCallInfo(task=Task.SELF_CHECK_HALLUCINATION.value))
            try:
                result = await llm_call(
                    llm,
                    formatted_prompt,
                    llm_params={"temperature": 1.0},
                )
                result = get_multiline_response(result)
                result = strip_quotes(result)
                return result
            except Exception as e:
                log.warning(f"Extra LLM response {index + 1}/{num_responses} failed: {e}")
                return None

        results = await asyncio.gather(*[_generate_extra_response(i) for i in range(num_responses)])
        extra_responses = [r for r in results if r is not None]

        if len(extra_responses) == 0:
            # Log message and return that no hallucination was found
            log.warning(f"No extra LLM responses were generated for '{bot_response}' hallucination check.")
            return False
        elif len(extra_responses) < num_responses:
            log.warning(
                f"Requested {num_responses} extra LLM responses for hallucination check, "
                f"received {len(extra_responses)}."
            )

        if use_llm_checking:
            # Only support LLM-based agreement check in current version
            prompt = llm_task_manager.render_task_prompt(
                task=Task.SELF_CHECK_HALLUCINATION,
                context={
                    "statement": bot_response,
                    "paragraph": ". ".join(extra_responses),
                },
            )

            # Initialize the LLMCallInfo object
            llm_call_info_var.set(LLMCallInfo(task=Task.SELF_CHECK_HALLUCINATION.value))
            stop = llm_task_manager.get_stop_tokens(task=Task.SELF_CHECK_HALLUCINATION)

            agreement = await llm_call(
                llm,
                prompt,
                stop=stop,
                llm_params={"temperature": config.lowest_temperature},
            )

            agreement = agreement.lower().strip()
            log.info(f"Agreement result for looking for hallucination is {agreement}.")

            # Return True if the hallucination check fails
            return "no" in agreement
        else:
            # TODO Implement BERT-Score based consistency method proposed by SelfCheckGPT paper
            # See details: https://arxiv.org/abs/2303.08896
            return False

    return False

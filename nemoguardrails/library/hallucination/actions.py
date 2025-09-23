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

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import BaseLLM

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
from nemoguardrails.logging.callbacks import logging_callback_manager_for_chain
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
    try:
        from langchain_openai import OpenAI
    except ImportError:
        log.warning(
            "The langchain_openai module is not installed. Please install it using pip: pip install langchain_openai"
        )

    bot_response = context.get("bot_message")
    last_bot_prompt_string = context.get("_last_bot_prompt")

    if bot_response and last_bot_prompt_string:
        num_responses = HALLUCINATION_NUM_EXTRA_RESPONSES
        # Use beam search for the LLM call, to get several completions with only one call.
        # At the current moment, only OpenAI LLM engines are supported for computing the additional completions.

        if "openai" not in str(type(llm)).lower():
            log.warning(
                f"Hallucination rail is optimized for OpenAI LLM engines. "
                f"Current LLM engine is {type(llm).__name__}, which may not support all features."
            )

            if "n" not in llm.__fields__:
                log.warning(
                    f"LLM engine {type(llm).__name__} does not support the 'n' parameter for generating multiple completion choices. "
                    f"Please use an OpenAI LLM engine or a model that supports the 'n' parameter for optimal performance."
                )
                return False

        # Use the "generate" call from langchain to get all completions in the same response.
        last_bot_prompt = PromptTemplate(template="{text}", input_variables=["text"])
        chain = LLMChain(prompt=last_bot_prompt, llm=llm)

        # Generate multiple responses with temperature 1.
        # Use chain.with_config for runtime parameters
        configured_chain = chain.with_config(
            configurable={"temperature": 1.0, "n": num_responses}
        )
        extra_llm_response = await configured_chain.agenerate(
            [{"text": last_bot_prompt_string}],
            run_manager=logging_callback_manager_for_chain,
        )

        extra_llm_completions = []
        if len(extra_llm_response.generations) > 0:
            extra_llm_completions = extra_llm_response.generations[0]

        extra_responses = []
        i = 0
        while i < num_responses and i < len(extra_llm_completions):
            result = extra_llm_completions[i].text
            # We need the same post-processing of responses as in "generate_bot_message"
            result = get_multiline_response(result)
            result = strip_quotes(result)
            extra_responses.append(result)
            i += 1

        if len(extra_responses) == 0:
            # Log message and return that no hallucination was found
            log.warning(
                f"No extra LLM responses were generated for '{bot_response}' hallucination check."
            )
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

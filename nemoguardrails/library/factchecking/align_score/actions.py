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
from nemoguardrails.library.factchecking.align_score.request import alignscore_request
from nemoguardrails.library.self_check.facts.actions import self_check_facts
from nemoguardrails.llm.taskmanager import LLMTaskManager

log = logging.getLogger(__name__)


@action()
async def alignscore_check_facts(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    llm: Optional[BaseLLM] = None,
    config: Optional[RailsConfig] = None,
):
    """Checks the facts for the bot response using an information alignment score."""
    fact_checking_config = llm_task_manager.config.rails.config.fact_checking
    fallback_to_self_check = fact_checking_config.fallback_to_self_check

    alignscore_api_url = fact_checking_config.parameters.get("endpoint")

    evidence = context.get("relevant_chunks", [])
    response = context.get("bot_message")

    alignscore = await alignscore_request(alignscore_api_url, evidence, response)
    if alignscore is None:
        log.warning(
            "AlignScore endpoint not set up properly. Falling back to the ask_llm approach for fact-checking."
        )
        # If fallback is enabled, we use AskLLM
        if fallback_to_self_check:
            return await self_check_facts(llm_task_manager, context, llm, config)
        else:
            # If we can't verify the facts, we assume it's ok
            # TODO: should this default be configurable?
            return 1.0
    else:
        return alignscore

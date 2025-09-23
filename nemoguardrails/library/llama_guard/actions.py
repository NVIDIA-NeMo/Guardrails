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
from typing import List, Optional, Tuple

from langchain_core.language_models.llms import BaseLLM

from nemoguardrails.actions import action
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.llm.types import Task
from nemoguardrails.logging.explain import LLMCallInfo

log = logging.getLogger(__name__)


def parse_llama_guard_response(response: str) -> Tuple[bool, Optional[List[str]]]:
    """
    Parses the response from the Llama Guard LLM and returns a tuple of:
    - Whether the response is safe or not.
    - If not safe, a list of the violated policies.
    """
    response = response.lower().strip()
    log.info(f"Llama Guard response: {response}.")
    if response.startswith("safe"):
        return True, None

    # If unsafe, extract the violated policy numbers and return it as an array.
    elif response.startswith("unsafe"):
        policy_violations = response.split("unsafe")[1].strip().split(" ")
        log.info(f"Violated policies: {policy_violations}")
        return False, policy_violations

    log.warning(
        f"""Unexpected Llama Guard response: {response}\n
                If prompted correctly, it should always start with 'safe' or 'unsafe'"""
    )
    return False, []


@action()
async def llama_guard_check_input(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    llama_guard_llm: Optional[BaseLLM] = None,
    **kwargs,
) -> dict:
    """
    Checks user messages using the configured Llama Guard model
    and the configured prompt containing the safety guidelines.
    """
    user_input = context.get("user_message")
    check_input_prompt = llm_task_manager.render_task_prompt(
        task=Task.LLAMA_GUARD_CHECK_INPUT,
        context={
            "user_input": user_input,
        },
    )
    stop = llm_task_manager.get_stop_tokens(task=Task.LLAMA_GUARD_CHECK_INPUT)

    # Initialize the LLMCallInfo object
    llm_call_info_var.set(LLMCallInfo(task=Task.SELF_CHECK_INPUT.value))

    result = await llm_call(
        llama_guard_llm, check_input_prompt, stop=stop, llm_params={"temperature": 0.0}
    )

    allowed, policy_violations = parse_llama_guard_response(result)
    return {"allowed": allowed, "policy_violations": policy_violations}


def llama_guard_check_output_mapping(result: dict) -> bool:
    """
    Mapping for llama_guard_check_output.

    Expects result to be a dict with:
      - "allowed": a boolean indicating if the response passed the safety check.
      - "policy_violations": additional details (not used in the mapping logic).

    Returns:
        True if the response should be blocked (i.e. if "allowed" is False),
        False otherwise.
    """
    allowed = result.get("allowed", True)
    return not allowed


@action(output_mapping=llama_guard_check_output_mapping)
async def llama_guard_check_output(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    llama_guard_llm: Optional[BaseLLM] = None,
) -> dict:
    """
    Check the bot response using the configured Llama Guard model
    and the configured prompt containing the safety guidelines.
    """
    user_input = context.get("user_message")
    bot_response = context.get("bot_message")

    check_output_prompt = llm_task_manager.render_task_prompt(
        task=Task.LLAMA_GUARD_CHECK_OUTPUT,
        context={
            "user_input": user_input,
            "bot_response": bot_response,
        },
    )
    stop = llm_task_manager.get_stop_tokens(task=Task.LLAMA_GUARD_CHECK_OUTPUT)

    # Initialize the LLMCallInfo object
    llm_call_info_var.set(LLMCallInfo(task=Task.SELF_CHECK_OUTPUT.value))

    result = await llm_call(
        llama_guard_llm, check_output_prompt, stop=stop, llm_params={"temperature": 0.0}
    )

    allowed, policy_violations = parse_llama_guard_response(result)
    return {"allowed": allowed, "policy_violations": policy_violations}

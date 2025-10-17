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

import logging
import os
import re
from typing import List, Literal, Optional, Tuple, Union

import aiohttp
from langchain_core.language_models.llms import BaseLLM

from nemoguardrails.actions import action
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.llm.types import Task
from nemoguardrails.logging.explain import LLMCallInfo

log = logging.getLogger(__name__)


def parse_patronus_lynx_response(
    response: str,
) -> Tuple[bool, Union[List[str], None]]:
    """
    Parses the response from the Patronus Lynx LLM and returns a tuple of:
    - Whether the response is hallucinated or not.
    - A reasoning trace explaining the decision.
    """
    log.info(f"Patronus Lynx response: {response}.")
    # Default to hallucinated
    hallucination, reasoning = True, None
    reasoning_pattern = r'"REASONING":\s*\[(.*?)\]'
    score_pattern = r'"SCORE":\s*"?\b(PASS|FAIL)\b"?'

    reasoning_match = re.search(reasoning_pattern, response, re.DOTALL)
    score_match = re.search(score_pattern, response)

    if score_match:
        score = score_match.group(1)
        if score == "PASS":
            hallucination = False
    if reasoning_match:
        reasoning_content = reasoning_match.group(1)
        reasoning = re.split(r"['\"],\s*['\"]", reasoning_content)

    return hallucination, reasoning


def patronus_lynx_check_output_hallucination_mapping(result: dict) -> bool:
    """
    Mapping for patronus_lynx_check_output_hallucination.

    Expects result to be a dict with:
        "hallucination": a boolean where True indicates a hallucination was detected.

    Block (return True) if "hallucination" is True.
    """
    return result.get("hallucination", False)


@action(output_mapping=patronus_lynx_check_output_hallucination_mapping)
async def patronus_lynx_check_output_hallucination(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    patronus_lynx_llm: Optional[BaseLLM] = None,
    **kwargs,
) -> dict:
    """
    Check the bot response for hallucinations based on the given chunks
    using the configured Patronus Lynx model.
    """
    user_input = context.get("user_message")
    bot_response = context.get("bot_message")
    provided_context = context.get("relevant_chunks")

    if (
        not provided_context
        or not isinstance(provided_context, str)
        or not provided_context.strip()
    ):
        log.error(
            "Could not run Patronus Lynx. `relevant_chunks` must be passed as a non-empty string."
        )
        return {"hallucination": False, "reasoning": None}

    check_output_hallucination_prompt = llm_task_manager.render_task_prompt(
        task=Task.PATRONUS_LYNX_CHECK_OUTPUT_HALLUCINATION,
        context={
            "user_input": user_input,
            "bot_response": bot_response,
            "provided_context": provided_context,
        },
    )

    stop = llm_task_manager.get_stop_tokens(
        task=Task.PATRONUS_LYNX_CHECK_OUTPUT_HALLUCINATION
    )

    # Initialize the LLMCallInfo object
    llm_call_info_var.set(
        LLMCallInfo(task=Task.PATRONUS_LYNX_CHECK_OUTPUT_HALLUCINATION.value)
    )

    result = await llm_call(
        patronus_lynx_llm,
        check_output_hallucination_prompt,
        stop=stop,
        llm_params={"temperature": 0.0},
    )

    hallucination, reasoning = parse_patronus_lynx_response(result)
    return {"hallucination": hallucination, "reasoning": reasoning}


def check_guardrail_pass(
    response: Optional[dict], success_strategy: Literal["all_pass", "any_pass"]
) -> bool:
    """
    Check if evaluations in the Patronus API response pass based on the success strategy.
    "all_pass" requires all evaluators to pass for success.
    "any_pass" requires only one evaluator to pass for success.
    """
    if not response or "results" not in response:
        return False

    evaluations = response["results"]

    if success_strategy == "all_pass":
        return all(
            "evaluation_result" in result
            and isinstance(result["evaluation_result"], dict)
            and result["evaluation_result"].get("pass", False)
            for result in evaluations
        )
    return any(
        "evaluation_result" in result
        and isinstance(result["evaluation_result"], dict)
        and result["evaluation_result"].get("pass", False)
        for result in evaluations
    )


async def patronus_evaluate_request(
    api_params: dict,
    user_input: Optional[str] = None,
    bot_response: Optional[str] = None,
    provided_context: Optional[Union[str, List[str]]] = None,
) -> Optional[dict]:
    """
    Make a call to the Patronus Evaluate API.

    Returns a dictionary of the API response JSON if successful, or None if a server error occurs.
        * Server errors will cause the guardrail to block the bot response

    Raises a ValueError for client errors (400-499), as these indicate invalid requests.
    """
    api_key = os.environ.get("PATRONUS_API_KEY")

    if api_key is None:
        raise ValueError("PATRONUS_API_KEY environment variable not set.")

    if "evaluators" not in api_params:
        raise ValueError(
            "The Patronus Evaluate API parameters must contain an 'evaluators' field"
        )
    evaluators = api_params["evaluators"]
    if not isinstance(evaluators, list):
        raise ValueError(
            "The Patronus Evaluate API parameter 'evaluators' must be a list"
        )

    for evaluator in evaluators:
        if not isinstance(evaluator, dict):
            raise ValueError(
                "Each object in the 'evaluators' list must be a dictionary"
            )
        if "evaluator" not in evaluator:
            raise ValueError(
                "Each dictionary in the 'evaluators' list must contain the 'evaluator' field"
            )

    data = {
        **api_params,
        "evaluated_model_input": user_input,
        "evaluated_model_output": bot_response,
        "evaluated_model_retrieved_context": provided_context,
    }

    url = "https://api.patronus.ai/v1/evaluate"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url=url,
            headers=headers,
            json=data,
        ) as response:
            if 400 <= response.status < 500:
                raise ValueError(
                    f"The Patronus Evaluate API call failed with status code {response.status}. "
                    f"Details: {await response.text()}"
                )

            if response.status != 200:
                log.error(
                    "The Patronus Evaluate API call failed with status code %s. Details: %s",
                    response.status,
                    await response.text(),
                )
                return None

            response_json = await response.json()
            return response_json


def patronus_api_check_output_mapping(result: dict) -> bool:
    """
    Mapping for patronus_api_check_output.

    Expects result to be a dict with:
        "pass": a boolean where True means the output passed the check.

    Block (return True) if "pass" is False.
    """
    # Default to True (pass) if the key is missing
    passed = result.get("pass", True)
    return not passed


@action(
    name="patronus_api_check_output", output_mapping=patronus_api_check_output_mapping
)
async def patronus_api_check_output(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
) -> dict:
    """
    Check the user message, bot response, and/or provided context
    for issues based on the Patronus Evaluate API
    """
    user_input = context.get("user_message")
    bot_response = context.get("bot_message")
    provided_context = context.get("relevant_chunks")

    patronus_config = llm_task_manager.config.rails.config.patronus.output
    evaluate_config = getattr(patronus_config, "evaluate_config", {})
    success_strategy: Literal["all_pass", "any_pass"] = getattr(
        evaluate_config, "success_strategy", "all_pass"
    )
    api_params = getattr(evaluate_config, "params", {})
    response = await patronus_evaluate_request(
        api_params=api_params,
        user_input=user_input,
        bot_response=bot_response,
        provided_context=provided_context,
    )
    return {
        "pass": check_guardrail_pass(
            response=response, success_strategy=success_strategy
        )
    }

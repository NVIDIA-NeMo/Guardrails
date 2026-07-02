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
import os
import re
from collections.abc import Mapping
from typing import List, Literal, Optional, Tuple, Union

from nemoguardrails.actions import action
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.http import HTTPClient, http_call, resolve_http_client
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.llm.types import Task
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.types import LLMModel

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


def _patronus_lynx_outcome(hallucination: bool, reasoning: Union[List[str], None]) -> RailOutcome:
    metadata = {"hallucination": hallucination, "reasoning": reasoning}
    if hallucination:
        return RailOutcome.block(metadata=metadata)
    return RailOutcome.allow(metadata=metadata)


def _get_patronus_lynx_model(
    llms: Mapping[str, LLMModel],
    model_name: str,
) -> LLMModel:
    model = llms.get(model_name)
    if model is None:
        raise ValueError(f"Patronus Lynx model {model_name!r} is unavailable in the shared model registry.")
    return model


@action()
async def patronus_lynx_check_output_hallucination(
    llm_task_manager: LLMTaskManager,
    llms: Mapping[str, LLMModel],
    model_name: str,
    context: Optional[dict] = None,
    **kwargs,
) -> RailOutcome:
    """
    Check the bot response for hallucinations based on the given chunks
    using the configured Patronus Lynx model.
    """
    user_input = context.get("user_message")
    bot_response = context.get("bot_message")
    provided_context = context.get("relevant_chunks")

    if not provided_context or not isinstance(provided_context, str) or not provided_context.strip():
        log.error("Could not run Patronus Lynx. `relevant_chunks` must be passed as a non-empty string.")
        return _patronus_lynx_outcome(False, None)

    llm = _get_patronus_lynx_model(llms, model_name)
    check_output_hallucination_prompt = llm_task_manager.render_task_prompt(
        task=Task.PATRONUS_LYNX_CHECK_OUTPUT_HALLUCINATION,
        context={
            "user_input": user_input,
            "bot_response": bot_response,
            "provided_context": provided_context,
        },
    )

    stop = llm_task_manager.get_stop_tokens(task=Task.PATRONUS_LYNX_CHECK_OUTPUT_HALLUCINATION)

    # Initialize the LLMCallInfo object
    llm_call_info_var.set(LLMCallInfo(task=Task.PATRONUS_LYNX_CHECK_OUTPUT_HALLUCINATION.value))

    result = (
        await llm_call(
            llm,
            check_output_hallucination_prompt,
            stop=stop,
            llm_params={"temperature": 0.0},
        )
    ).content

    hallucination, reasoning = parse_patronus_lynx_response(result)
    return _patronus_lynx_outcome(hallucination, reasoning)


def check_guardrail_pass(response: Optional[dict], success_strategy: Literal["all_pass", "any_pass"]) -> bool:
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
    http_client: Optional[HTTPClient] = None,
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
        raise ValueError("The Patronus Evaluate API parameters must contain an 'evaluators' field")
    evaluators = api_params["evaluators"]
    if not isinstance(evaluators, list):
        raise ValueError("The Patronus Evaluate API parameter 'evaluators' must be a list")

    for evaluator in evaluators:
        if not isinstance(evaluator, dict):
            raise ValueError("Each object in the 'evaluators' list must be a dictionary")
        if "evaluator" not in evaluator:
            raise ValueError("Each dictionary in the 'evaluators' list must contain the 'evaluator' field")

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

    async with resolve_http_client(http_client) as client:
        response = await http_call(
            client,
            "POST",
            url,
            headers=headers,
            json=data,
            raise_for_status=False,
        )
    if 400 <= response.status_code < 500:
        raise ValueError(
            f"The Patronus Evaluate API call failed with status code {response.status_code}. Details: {response.text}"
        )

    if response.status_code != 200:
        log.error(
            "The Patronus Evaluate API call failed with status code %s. Details: %s",
            response.status_code,
            response.text,
        )
        return None

    return response.json()


@action(name="patronus_api_check_output")
async def patronus_api_check_output(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    http_client: Optional[HTTPClient] = None,
) -> RailOutcome:
    """
    Check the user message, bot response, and/or provided context
    for issues based on the Patronus Evaluate API
    """
    user_input = context.get("user_message")
    bot_response = context.get("bot_message")
    provided_context = context.get("relevant_chunks")

    patronus_config = llm_task_manager.config.rails.config.patronus.output
    evaluate_config = getattr(patronus_config, "evaluate_config", {})
    success_strategy: Literal["all_pass", "any_pass"] = getattr(evaluate_config, "success_strategy", "all_pass")
    api_params = getattr(evaluate_config, "params", {})
    request_kwargs = {"http_client": http_client} if http_client is not None else {}
    response = await patronus_evaluate_request(
        api_params=api_params,
        user_input=user_input,
        bot_response=bot_response,
        provided_context=provided_context,
        **request_kwargs,
    )
    passed = check_guardrail_pass(response=response, success_strategy=success_strategy)
    metadata = {"pass": passed}
    if passed:
        return RailOutcome.allow(metadata=metadata)
    return RailOutcome.block(metadata=metadata)

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

import logging
import warnings
from collections.abc import Collection
from typing import Any, Dict, List, Optional

from nemoguardrails.actions.llm.utils import llm_call, warn_if_truncated
from nemoguardrails.colang.v1_0.runtime.flows import _get_flow_params, _normalize_flow_id
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.types import LLMModel

log = logging.getLogger(__name__)

SELF_CHECK_INPUT_FLOW = "self check input"
SELF_CHECK_OUTPUT_FLOW = "self check output"
SELF_CHECK_VARIANT_PARAM = "variant"
SELF_CHECK_INPUT_VARIANT_PARAM = SELF_CHECK_VARIANT_PARAM
SELF_CHECK_OUTPUT_VARIANT_PARAM = SELF_CHECK_VARIANT_PARAM
SELF_CHECK_INPUT_DEFAULT_TASK = "self_check_input"
SELF_CHECK_OUTPUT_DEFAULT_TASK = "self_check_output"


def get_self_check_task_from_rail(flow: Any, flow_id: str, variant_param: str, default_task: str) -> Optional[str]:
    if not isinstance(flow, str) or _normalize_flow_id(flow) != flow_id:
        return None

    return _get_flow_params(flow).get(variant_param) or default_task


def get_self_check_prompt_task(
    task: str, default_task: str, available_prompt_tasks: Optional[Collection[str]] = None
) -> str:
    """Resolve the prompt task name while supporting deprecated bare custom tasks."""
    if task == default_task:
        return task

    prompt_task = f"{default_task} ${SELF_CHECK_VARIANT_PARAM}={task}"
    if (
        available_prompt_tasks is not None
        and prompt_task not in available_prompt_tasks
        and task in available_prompt_tasks
    ):
        warnings.warn(
            f"The `{task}` self-check prompt task is deprecated; rename it to `{prompt_task}`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return task

    return prompt_task


def resolve_self_check_task(
    variant: Optional[str],
    context: Optional[dict],
    events: Optional[List[dict]],
    triggered_rail_key: str,
    start_rail_event_type: str,
    flow_id: str,
    variant_param: str,
    default_task: str,
) -> str:
    if variant and not variant.startswith("$"):
        return variant

    context = context or {}
    context_task = get_self_check_task_from_rail(
        context.get(triggered_rail_key),
        flow_id=flow_id,
        variant_param=variant_param,
        default_task=default_task,
    )
    if context_task:
        return context_task

    for event in reversed(events or []):
        if event.get("type") == "start_flow" and event.get("flow_id") == flow_id:
            event_params = event.get("params") or {}
            return event_params.get(variant_param) or default_task

        if event.get("type") == start_rail_event_type:
            event_task = get_self_check_task_from_rail(
                event.get("flow_id"),
                flow_id=flow_id,
                variant_param=variant_param,
                default_task=default_task,
            )
            if event_task:
                return event_task

    return default_task


def get_self_check_llm(
    llms: Dict[str, LLMModel],
    task: str,
    default_task: str,
    main_llm: Optional[LLMModel],
) -> LLMModel:
    if task in llms:
        return llms[task]

    if default_task in llms:
        log.debug("No model found with type=%s, falling back to default %s model", task, default_task)
        return llms[default_task]

    if main_llm is not None:
        log.debug("No model found with type=%s or type=%s, falling back to main model", task, default_task)
        return main_llm

    raise ValueError(
        f"No matching model for task={task} found. "
        f"Please configure a model with type={task}, type={default_task}, or type=main"
    )


def parse_self_check_output(llm_task_manager: Any, task: str, response: str):
    if llm_task_manager.has_output_parser(task):
        return llm_task_manager.parse_task_output(task, output=response)

    return llm_task_manager.parse_task_output(task, output=response, forced_output_parser="is_content_safe")


async def run_self_check_task(
    task: str,
    prompt_context: Dict[str, Any],
    llms: Dict[str, LLMModel],
    default_task: str,
    main_llm: Optional[LLMModel],
    llm_task_manager: Any,
    lowest_temperature: float,
    max_tokens: int = 1024,
) -> tuple[bool, str]:
    llm = get_self_check_llm(llms, task, default_task=default_task, main_llm=main_llm)
    available_prompt_tasks = {prompt.task for prompt in llm_task_manager.config.prompts or []}
    prompt_task = get_self_check_prompt_task(task, default_task, available_prompt_tasks)

    prompt = llm_task_manager.render_task_prompt(
        task=prompt_task,
        context=prompt_context,
    )
    stop = llm_task_manager.get_stop_tokens(task=prompt_task)
    task_max_tokens = llm_task_manager.get_max_tokens(task=prompt_task) or max_tokens

    llm_call_info_var.set(LLMCallInfo(task=prompt_task))

    llm_response = await llm_call(
        llm,
        prompt,
        stop=stop,
        llm_params={
            "temperature": lowest_temperature,
            "max_tokens": task_max_tokens,
        },
    )
    warn_if_truncated(llm_response, prompt_task)
    response = llm_response.content

    result = parse_self_check_output(llm_task_manager, prompt_task, response)
    return bool(result[0]), response

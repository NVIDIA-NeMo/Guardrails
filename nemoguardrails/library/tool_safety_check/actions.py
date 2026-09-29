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

import json
from typing import Dict, Optional

from nemoguardrails.actions.actions import action
from nemoguardrails.actions.llm_judge import require_llm, run_llm_judged_check
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.guardrails.tool_schema import Tool, ToolResult, scope_arguments, tool_output_validation
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.types import LLMModel, ToolCall

_DEFAULT_TEMPERATURE = 1e-20


@action(is_system_action=True)
@tool_output_validation
async def tool_safety_check_output(
    llms: Dict[str, LLMModel],
    llm_task_manager: LLMTaskManager,
    tool_call: ToolCall,
    tool_definition: Optional[Tool],
    model_name: Optional[str] = None,
    variant: Optional[str] = None,
    argument_name: Optional[str] = None,
    **kwargs,
) -> RailOutcome:
    """Checks a tool call's arguments against per-tool safety instructions using a specified LLM.

    Args:
        tool_call: The tool call to check.
        tool_definition: The declared tool, used only by @tool_output_validation.
        model_name: The judge model, from `$model=` on the flow.
        variant: Selects which `config.prompts` task to render, from `$variant=` on the flow.
        argument_name: Narrows the arguments sent to the judge to one named argument, from
            `$argument=` on the flow. Absent, the full arguments dict is sent.
    """
    llm = require_llm(llms, model_name, rail_name="tool safety check output")

    tool_name = tool_call.function.name or tool_call.type
    task = f"tool_safety_check_output $model={model_name} $variant={variant}"
    arguments = scope_arguments(tool_call.function.arguments, argument_name)
    prompt = llm_task_manager.render_task_prompt(
        task=task,
        context={"tool_name": tool_name, "tool_call_arguments": json.dumps(arguments)},
    )
    return await run_llm_judged_check(llm, llm_task_manager, task, prompt, _DEFAULT_TEMPERATURE)


@action(is_system_action=True)
async def tool_safety_check_input(
    llms: Dict[str, LLMModel],
    llm_task_manager: LLMTaskManager,
    tool_call: ToolCall,
    tool_result: ToolResult,
    model_name: Optional[str] = None,
    variant: Optional[str] = None,
    **kwargs,
) -> RailOutcome:
    """Checks a tool result's content against per-tool safety instructions using a specified LLM.

    Args:
        tool_call: The prior call this result answers, used to resolve the tool name.
        tool_result: The tool result to check.
        model_name: The judge model, from `$model=` on the flow.
        variant: Selects which `config.prompts` task to render, from `$variant=` on the flow.
    """
    llm = require_llm(llms, model_name, rail_name="tool safety check input")

    tool_name = tool_call.function.name or tool_call.type
    content = tool_result.content
    text = content if isinstance(content, str) else json.dumps(content)
    task = f"tool_safety_check_input $model={model_name} $variant={variant}"
    prompt = llm_task_manager.render_task_prompt(
        task=task,
        context={"tool_name": tool_name, "tool_result_content": text},
    )
    return await run_llm_judged_check(llm, llm_task_manager, task, prompt, _DEFAULT_TEMPERATURE)

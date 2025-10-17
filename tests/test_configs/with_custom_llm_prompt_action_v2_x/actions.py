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

from ast import literal_eval
from typing import Optional

from langchain_core.language_models.llms import BaseLLM

from nemoguardrails.actions import action
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.logging.explain import LLMCallInfo


@action(name="CustomLlmRequestAction", is_system_action=True, execute_async=True)
async def custom_llm_request(
    llm_task_manager: LLMTaskManager,
    prompt_template_name: str,
    context: Optional[dict] = None,
    llm: Optional[BaseLLM] = None,
    **kwargs,
):
    # user_input = context.get("user_message")

    prompt = llm_task_manager.render_task_prompt(
        task=prompt_template_name,
        context=kwargs,
    )
    stop = llm_task_manager.get_stop_tokens(prompt_template_name)

    # Initialize the LLMCallInfo object
    llm_call_info_var.set(LLMCallInfo(task=prompt_template_name))

    result = await llm_call(llm, prompt, stop=stop, llm_params={"temperature": 0.5})

    result = llm_task_manager.parse_task_output(prompt_template_name, output=result)
    result = result.text

    # Any additional parsing of the output
    value = result.strip().split("\n")[0]
    if value.endswith(";"):
        value = value[:-1]

    return literal_eval(value)

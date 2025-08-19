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
from typing import Dict, List, Optional

from langchain_core.language_models.llms import BaseLLM

from nemoguardrails.actions.actions import action
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.llm.filters import to_chat_messages
from nemoguardrails.llm.params import llm_params
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.logging.explain import LLMCallInfo

log = logging.getLogger(__name__)


@action()
async def topic_safety_check_input(
    llms: Dict[str, BaseLLM],
    llm_task_manager: LLMTaskManager,
    model_name: Optional[str] = None,
    context: Optional[dict] = None,
    events: Optional[List[dict]] = None,
    **kwargs,
) -> dict:
    _MAX_TOKENS = 10
    user_input: str = ""

    if context is not None:
        user_input = context.get("user_message", "")
        model_name = model_name or context.get("model", None)

    if events is not None:
        # convert InternalEvent objects to dictionary format for compatibility with to_chat_messages
        dict_events = []
        for event in events:
            if hasattr(event, "name") and hasattr(event, "arguments"):
                dict_event = {"type": event.name}
                dict_event.update(event.arguments)
                dict_events.append(dict_event)
            else:
                dict_events.append(event)

        conversation_history = to_chat_messages(dict_events)

    if model_name is None:
        error_msg = (
            "Model name is required for topic safety check, "
            "please provide it as an argument in the config.yml. "
            "e.g. topic safety check input $model=llama_topic_guard"
        )
        raise ValueError(error_msg)

    llm = llms.get(model_name, None)

    if llm is None:
        error_msg = (
            f"Model {model_name} not found in the list of available models for topic safety check. "
            "Please provide a valid model name."
        )
        raise ValueError(error_msg)

    task = f"topic_safety_check_input $model={model_name}"

    system_prompt = llm_task_manager.render_task_prompt(
        task=task,
    )

    TOPIC_SAFETY_OUTPUT_RESTRICTION = (
        'If any of the above conditions are violated, please respond with "off-topic". '
        'Otherwise, respond with "on-topic". '
        'You must respond with "on-topic" or "off-topic".'
    )

    system_prompt = system_prompt.strip()
    if not system_prompt.endswith(TOPIC_SAFETY_OUTPUT_RESTRICTION):
        system_prompt = f"{system_prompt}\n\n{TOPIC_SAFETY_OUTPUT_RESTRICTION}"

    stop = llm_task_manager.get_stop_tokens(task=task)
    max_tokens = llm_task_manager.get_max_tokens(task=task)

    llm_call_info_var.set(LLMCallInfo(task=task))

    max_tokens = max_tokens or _MAX_TOKENS

    messages = []
    messages.append({"type": "system", "content": system_prompt})
    messages.extend(conversation_history)
    messages.append({"type": "user", "content": user_input})

    with llm_params(llm, temperature=0.01):
        result = await llm_call(llm, messages, stop=stop)

    if result.lower().strip() == "off-topic":
        on_topic = False
    else:
        on_topic = True

    return {"on_topic": on_topic}

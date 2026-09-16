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

"""Shared helpers for library rail actions that use an LLM as a judge.

Not Colang-specific, unlike `nemoguardrails.actions.llm`.
"""

from typing import Dict, Optional

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.llm.call import llm_call, warn_if_truncated
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.types import LLMModel

DEFAULT_MAX_TOKENS = 1024


def require_llm(llms: Dict[str, LLMModel], model_name: Optional[str], *, rail_name: str) -> LLMModel:
    """Resolve `model_name` in `llms`, raising a clear error if it's missing or unknown."""
    if model_name is None:
        raise ValueError(
            f"Model name is required for {rail_name}, please provide it as an argument "
            "in the config.yml, e.g. $model=<name>."
        )
    llm = llms.get(model_name)
    if llm is None:
        raise ValueError(
            f"Model {model_name} not found in the list of available models for {rail_name}. "
            "Please provide a valid model name."
        )
    return llm


async def run_llm_judged_check(
    llm: LLMModel,
    llm_task_manager: LLMTaskManager,
    task: str,
    prompt: str,
    temperature: float,
) -> RailOutcome:
    """Call the judge `llm` with the already-rendered `prompt` and parse the response.

    Uses the prompt's own `output_parser` if it declares one, otherwise falls back to
    `is_content_safe`: a yes/no/safe/unsafe judgment, optionally followed by violation words.
    """
    stop = llm_task_manager.get_stop_tokens(task=task)
    max_tokens = llm_task_manager.get_max_tokens(task=task) or DEFAULT_MAX_TOKENS

    llm_call_info_var.set(LLMCallInfo(task=task))

    llm_response = await llm_call(
        llm,
        prompt,
        stop=stop,
        llm_params={"temperature": temperature, "max_tokens": max_tokens},
    )
    warn_if_truncated(llm_response, task)
    forced_output_parser = None if llm_task_manager.has_output_parser(task) else "is_content_safe"
    result = llm_task_manager.parse_task_output(
        task, output=llm_response.content, forced_output_parser=forced_output_parser
    )
    # An unregistered output_parser name makes parse_task_output return the raw completion
    # string unparsed. Unpacking that as [is_safe, *violations] would read its first character
    # as a truthy is_safe, silently allowing an "unsafe: ..." verdict. Fail closed instead.
    if not isinstance(result, (list, tuple)) or not result or not isinstance(result[0], bool):
        return RailOutcome.block(reason="the judge model's response could not be parsed into a safety verdict")
    is_safe, *violations = result
    if is_safe:
        return RailOutcome.allow(metadata={"violations": violations})
    return RailOutcome.block(reason=", ".join(violations) or None, metadata={"violations": violations})

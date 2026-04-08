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

"""Topic safety rail action for IORails."""

from typing import Any, Optional

from nemoguardrails.guardrails.guardrails_types import LLMMessages, RailResult
from nemoguardrails.guardrails.rail_action import RailAction
from nemoguardrails.library.topic_safety.actions import (
    TOPIC_SAFETY_MAX_TOKENS,
    TOPIC_SAFETY_OUTPUT_RESTRICTION,
    TOPIC_SAFETY_TEMPERATURE,
)


class TopicSafetyInputAction(RailAction):
    """Check whether user input is on-topic per configured guidelines."""

    action_name = "topic safety check input"
    requires_model = True

    def _extract_messages(self, messages: LLMMessages, bot_response: Optional[str]) -> dict[str, Any]:
        return {"messages": messages}

    def _create_prompt(self, model_type: Optional[str], extracted: dict[str, Any]) -> list[dict]:
        task_key = f"topic_safety_check_input $model={model_type}"

        system_prompt = self.task_manager.render_task_prompt(task=task_key)
        if isinstance(system_prompt, list):
            raise RuntimeError(f"Topic safety prompt must be a string template, got messages: {task_key}")

        system_prompt = system_prompt.strip()
        if not system_prompt.endswith(TOPIC_SAFETY_OUTPUT_RESTRICTION):
            system_prompt = f"{system_prompt}\n\n{TOPIC_SAFETY_OUTPUT_RESTRICTION}"

        return [{"role": "system", "content": system_prompt}, *extracted["messages"]]

    async def _get_response(self, model_type: Optional[str], prompt: Any) -> str:
        task_key = f"topic_safety_check_input $model={model_type}"

        stop = self.task_manager.get_stop_tokens(task=task_key)
        max_tokens = self.task_manager.get_max_tokens(task=task_key) or TOPIC_SAFETY_MAX_TOKENS
        kwargs: dict = {"temperature": TOPIC_SAFETY_TEMPERATURE, "max_tokens": max_tokens}
        if stop:
            kwargs["stop"] = stop

        return await self._get_llm_response(model_type, prompt, **kwargs)

    def _parse_response(self, response: Any) -> RailResult:
        if response.lower().strip() == "off-topic":
            return RailResult(is_safe=False, reason="Topic safety: off-topic")
        return RailResult(is_safe=True)

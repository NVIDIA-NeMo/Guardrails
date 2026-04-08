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

"""Content safety rail actions for IORails."""

from typing import Any, Optional

from nemoguardrails.guardrails.guardrails_types import LLMMessages, RailResult
from nemoguardrails.guardrails.rail_action import RailAction

_MAX_TOKENS = 3
_TEMPERATURE = 1e-20


class ContentSafetyInputAction(RailAction):
    """Check user input for content safety violations."""

    action_name = "content safety check input"
    requires_model = True

    def _extract_messages(self, messages: LLMMessages, bot_response: Optional[str]) -> dict[str, Any]:
        return {"user_input": self._last_user_content(messages)}

    def _create_prompt(self, model_type: Optional[str], extracted: dict[str, Any]) -> list[dict]:
        prompt_task_key = f"content_safety_check_input $model={model_type}"
        content_safety_config = self.task_manager.config.rails.config.content_safety
        if content_safety_config is None:
            raise RuntimeError("content_safety config is required for content safety rail")
        reasoning_enabled = content_safety_config.reasoning.enabled

        prompt = self.task_manager.render_task_prompt(
            task=prompt_task_key,
            context={"user_input": extracted["user_input"], "reasoning_enabled": reasoning_enabled},
        )
        return self._prompt_to_messages(prompt)

    async def _get_response(self, model_type: Optional[str], prompt: Any) -> str:
        prompt_task_key = f"content_safety_check_input $model={model_type}"

        stop = self.task_manager.get_stop_tokens(task=prompt_task_key)
        max_tokens = self.task_manager.get_max_tokens(task=prompt_task_key) or _MAX_TOKENS
        kwargs: dict = {"temperature": _TEMPERATURE, "max_tokens": max_tokens}
        if stop:
            kwargs["stop"] = stop

        response_text = await self._get_llm_response(model_type, prompt, **kwargs)

        # Parse via LLMTaskManager's registered output parser
        return self.task_manager.parse_task_output(task=prompt_task_key, output=response_text)  # type: ignore[arg-type]

    def _parse_response(self, response: Any) -> RailResult:
        return _content_safety_to_rail_result(response)


class ContentSafetyOutputAction(RailAction):
    """Check bot response for content safety violations."""

    action_name = "content safety check output"

    def _extract_messages(self, messages: LLMMessages, bot_response: Optional[str]) -> dict[str, Any]:
        if not bot_response:
            raise RuntimeError("bot_response is required for content safety output check")
        return {
            "user_input": self._last_user_content(messages),
            "bot_response": bot_response,
        }

    def _create_prompt(self, model_type: Optional[str], extracted: dict[str, Any]) -> list[dict]:
        prompt_task_key = f"content_safety_check_output $model={model_type}"
        content_safety_config = self.task_manager.config.rails.config.content_safety
        if content_safety_config is None:
            raise RuntimeError("content_safety config is required for content safety rail")
        reasoning_enabled = content_safety_config.reasoning.enabled

        prompt = self.task_manager.render_task_prompt(
            task=prompt_task_key,
            context={
                "user_input": extracted["user_input"],
                "bot_response": extracted["bot_response"],
                "reasoning_enabled": reasoning_enabled,
            },
        )
        return self._prompt_to_messages(prompt)

    async def _get_response(self, model_type: Optional[str], prompt: Any) -> str:
        prompt_task_key = f"content_safety_check_output $model={model_type}"

        stop = self.task_manager.get_stop_tokens(task=prompt_task_key)
        max_tokens = self.task_manager.get_max_tokens(task=prompt_task_key) or _MAX_TOKENS
        kwargs: dict = {"temperature": _TEMPERATURE, "max_tokens": max_tokens}
        if stop:
            kwargs["stop"] = stop

        response_text = await self._get_llm_response(model_type, prompt, **kwargs)
        return self.task_manager.parse_task_output(task=prompt_task_key, output=response_text)  # type: ignore[arg-type]

    def _parse_response(self, response: Any) -> RailResult:
        return _content_safety_to_rail_result(response)


def _content_safety_to_rail_result(parsed: object) -> RailResult:
    """Convert nemoguard parser output to RailResult.

    nemoguard_parse_prompt_safety / nemoguard_parse_response_safety return:
      [True]                        -> safe
      [False, "S1: Violence", ...]  -> unsafe with categories
    """
    if isinstance(parsed, (list, tuple)):
        if parsed and parsed[0] is True:
            return RailResult(is_safe=True)
        if parsed and parsed[0] is False:
            if len(parsed) > 1:
                categories = ", ".join(str(c) for c in parsed[1:])
                return RailResult(is_safe=False, reason=f"Safety categories: {categories}")
            return RailResult(is_safe=False, reason="Unknown")
    raise RuntimeError(f"Unexpected content safety parse result: {parsed}")

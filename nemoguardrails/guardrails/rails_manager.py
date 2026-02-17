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

"""Rails manager for IORails engine.

Orchestrates input/output safety checks by calling ModelManager.
Rails run sequentially; the first failing rail short-circuits the check.
"""

import logging
from typing import Sequence, cast

from jinja2.sandbox import SandboxedEnvironment

from nemoguardrails.guardrails.guardrails_types import LLMMessages, RailResult
from nemoguardrails.guardrails.model_manager import ModelManager
from nemoguardrails.llm.output_parsers import nemoguard_parse_prompt_safety, nemoguard_parse_response_safety
from nemoguardrails.rails.llm.config import RailsConfig, TaskPrompt

log = logging.getLogger(__name__)


class RailsManager:
    """Orchestrates input and output safety checks for IORails.

    Reads the rails configuration to determine which checks are enabled,
    then runs them using ModelManager for all LLM/safety calls.
    """

    def __init__(self, config: RailsConfig, model_manager: ModelManager) -> None:
        self.config = config
        self.model_manager = model_manager

        # Store prompts keyed by task name for easy lookup
        self.prompts: dict[str, TaskPrompt] = {}
        if config.prompts:
            self.prompts = {prompt.task: prompt for prompt in config.prompts}

        # Determine which input/output rails are enabled
        self.input_flows: list[str] = list(config.rails.input.flows)
        self.output_flows: list[str] = list(config.rails.output.flows)

        # Create jinja2 rendering environment
        self._jinja2_env = SandboxedEnvironment(autoescape=False)

        log.info("RailsManager initialized: input_flows=%s, output_flows=%s", self.input_flows, self.output_flows)

    async def is_input_safe(self, messages: list[dict]) -> RailResult:
        """Run input-rails and return whether all rails pass

        Args:
            messages: The conversation messages to check.

        Returns:
            RailResult(is_safe=True) if all rails pass, or RailResult(is_safe=False, reason=...) on failure.
        """
        if not self.input_flows:
            return RailResult(is_safe=True)

        for flow in self.input_flows:
            result = await self._run_input_rail(flow, messages)
            if not result.is_safe:
                return result

        return RailResult(is_safe=True)

    async def is_output_safe(self, messages: list[dict], response: str) -> RailResult:
        """Run all enabled output rails.

        Args:
            messages: The original conversation messages (input context).
            response: The LLM-generated response to check.

        Returns:
            RailResult(is_safe=True) if all rails pass, or RailResult(is_safe=False, reason=...) on failure.
        """
        if not self.output_flows:
            return RailResult(is_safe=True)

        for flow in self.output_flows:
            result = await self._run_output_rail(flow, messages, response)
            if not result.is_safe:
                return result

        return RailResult(is_safe=True)

    async def _run_input_rail(self, flow: str, messages: list[dict]) -> RailResult:
        """Run an input rail flow if it's supported. If not raise an exception"""
        # Extract the base flow name (strip any $model=... parameter)
        base_flow = self._flow_name(flow)

        if base_flow == "content safety check input":
            return await self._check_content_safety_input(flow, messages)
        else:
            raise RuntimeError(f"Input rail flow `{base_flow}` not supported")

    async def _run_output_rail(self, flow: str, messages: list[dict], response: str) -> RailResult:
        """Run an output rail flow if it's supported. If not raise an exception"""
        base_flow = self._flow_name(flow)

        if base_flow == "content safety check output":
            return await self._check_content_safety_output(flow, messages, response)
        else:
            raise RuntimeError(f"Output rail flow `{base_flow}` not supported")

    async def _check_content_safety_input(self, flow: str, messages: list[dict]) -> RailResult:
        """Check input content safety via the content_safety model."""

        model_type = self._flow_model_type(flow)
        last_user_content = self._last_user_content(messages)
        prompt_key = self._flow_to_prompt_key(flow)
        prompt_content = self._render_prompt(prompt_key, user_input=last_user_content)

        try:
            response_text = await self.model_manager.generate_async(
                model_type, [{"role": "user", "content": prompt_content}]
            )

            result = self._parse_content_safety_input_response(response_text)
            return result

        except Exception as e:
            log.error("Content safety input check failed: %s", e)
            return RailResult(is_safe=False, reason=f"Content safety input check error: {e}")

    async def _check_content_safety_output(self, flow: str, messages: list[dict], response: str) -> RailResult:
        """Check output content safety via the content_safety model."""
        model_type = self._flow_model_type(flow)
        last_user_content = self._last_user_content(messages)
        prompt_key = self._flow_to_prompt_key(flow)
        prompt_content = self._render_prompt(prompt_key, user_input=last_user_content, bot_response=response)

        try:
            response_text = await self.model_manager.generate_async(
                model_type, [{"role": "user", "content": prompt_content}]
            )
            result = self._parse_content_safety_output_response(response_text)
            return result

        except Exception as e:
            log.error("Content safety output check failed: %s", e)
            return RailResult(is_safe=False, reason=f"Content safety output check error: {e}")

    def _render_prompt(
        self,
        prompt_key: str,
        user_input: str = "",
        bot_response: str = "",
    ) -> str:
        """Look up a prompt template by task key and render the prompt."""
        prompt_template = self.prompts.get(prompt_key)
        if not prompt_template or not prompt_template.content:
            raise RuntimeError(f"No prompt template found for key {prompt_key}")

        content = prompt_template.content
        template = self._jinja2_env.from_string(content)
        content = template.render(user_input=user_input, bot_response=bot_response)
        return content

    @staticmethod
    def _flow_to_prompt_key(flow: str) -> str:
        """Convert a flow name to the corresponding prompt task key.

        Flow names use spaces, prompt task keys use underscores:
          'content safety check input $model=content_safety'
          -> 'content_safety_check_input $model=content_safety'
        """
        if "$" in flow:
            base, param = flow.split("$", 1)
            return base.strip().replace(" ", "_") + " $" + param
        return flow.replace(" ", "_")

    @staticmethod
    def _flow_name(flow: str) -> str:
        """Extract the base flow name, stripping any $model=... parameter.
        For example:
           'content safety check input $model=content_safety' -> 'content safety check input'
           'self check input' -> 'self check input'
        """
        if "$model=" in flow:
            return flow.split("$model=")[0].strip()
        return flow

    @staticmethod
    def _flow_model_type(flow: str) -> str:
        """Extract the model type from a flow name like 'content safety check input $model=content_safety'."""
        if "$model=" in flow:
            return flow.split("$model=")[1].strip()
        raise RuntimeError(f"Flow {flow} doesn't contain a model type")

    @staticmethod
    def _last_content_by_role(messages: LLMMessages, role: str) -> str:
        """Get the last content from the provided role"""
        for message in reversed(messages):
            message_role = message.get("role")
            if message_role and message_role == role:
                message_content = message.get("content")
                if message_content:
                    return message_content

        raise RuntimeError(f"No {role}-role content in messages: {messages}")

    def _last_user_content(self, messages: LLMMessages) -> str:
        """Return the last entry in messages list with role set to `user`"""
        return self._last_content_by_role(messages, "user")

    def _parse_content_safety_input_response(self, response: str) -> RailResult:
        """Use the existing `nemoguard_parse_prompt_safety` method and convert to RailResult."""

        result = nemoguard_parse_prompt_safety(response)
        rail_result = self._parse_content_safety_result(result)
        return rail_result

    def _parse_content_safety_output_response(self, response: str) -> RailResult:
        """Use the existing `nemoguard_parse_response_safety` method and convert to RailResult."""

        result = nemoguard_parse_response_safety(response)
        rail_result = self._parse_content_safety_result(result)
        return rail_result

    def _parse_content_safety_result(self, result: Sequence[bool | str]) -> RailResult:
        """Convert return format of nemoguard_parse_prompt_safety and nemoguard_parse_response_safety
           to RailResult

        This is a list of either:
        - SAFE: [True]
        - UNSAFE: [False, "S1: Violence", "S17: Malware"]
        """

        if len(result) == 1 and result[0]:
            return RailResult(is_safe=True)

        if len(result) > 1 and not result[0]:
            unsafe_list: list[str] = cast(list[str], result[1:])
            unsafe_categories = ",".join(unsafe_list)
            return RailResult(is_safe=False, reason=f"Safety categories: {unsafe_categories}")

        raise RuntimeError(f"Content safety response invalid: {result}")

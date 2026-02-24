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
Rails run sequentially by default; the first failing rail short-circuits.
When parallel mode is enabled, all rails run concurrently and the first
unsafe result cancels remaining rails immediately.
"""

import asyncio
import logging
from collections.abc import Coroutine, Mapping, Sequence
from typing import Any, cast

from jinja2.sandbox import SandboxedEnvironment

from nemoguardrails.guardrails.guardrails_types import (
    LLMMessages,
    RailDirection,
    RailResult,
    get_request_id,
    truncate,
)
from nemoguardrails.guardrails.model_manager import ModelManager
from nemoguardrails.library.topic_safety.actions import (
    TOPIC_SAFETY_MAX_TOKENS,
    TOPIC_SAFETY_OUTPUT_RESTRICTION,
    TOPIC_SAFETY_TEMPERATURE,
)
from nemoguardrails.llm.output_parsers import nemoguard_parse_prompt_safety, nemoguard_parse_response_safety
from nemoguardrails.rails.llm.config import RailsConfig, TaskPrompt, _get_flow_model, _get_flow_name

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

        # Parallel execution flags (Optional[bool] in config, coerce to bool)
        self.input_parallel: bool = config.rails.input.parallel or False
        self.output_parallel: bool = config.rails.output.parallel or False

        log.info(
            "RailsManager initialized: input_flows=%s, output_flows=%s, input_parallel=%s, output_parallel=%s",
            self.input_flows,
            self.output_flows,
            self.input_parallel,
            self.output_parallel,
        )
        # Create jinja2 rendering environment
        self._jinja2_env = SandboxedEnvironment(autoescape=False)

    async def is_input_safe(self, messages: list[dict]) -> RailResult:
        """Run all enabled input rails, short-circuiting on the first failure.

        When parallel mode is enabled, all rails run concurrently and the first
        unsafe result cancels remaining rails.
        """
        if not self.input_flows:
            return RailResult(is_safe=True)

        rails = {flow: self._run_input_rail(flow, messages) for flow in self.input_flows}
        if self.input_parallel:
            return await self._run_rails_parallel(rails, RailDirection.INPUT)
        return await self._run_rails_sequential(rails, RailDirection.INPUT)

    async def is_output_safe(self, messages: list[dict], response: str) -> RailResult:
        """Run all enabled output rails, short-circuiting on the first failure.

        When parallel mode is enabled, all rails run concurrently and the first
        unsafe result cancels remaining rails.
        """
        if not self.output_flows:
            return RailResult(is_safe=True)

        rails = {flow: self._run_output_rail(flow, messages, response) for flow in self.output_flows}
        if self.output_parallel:
            return await self._run_rails_parallel(rails, RailDirection.OUTPUT)
        return await self._run_rails_sequential(rails, RailDirection.OUTPUT)

    async def _run_rails_sequential(
        self,
        rails: Mapping[str, Coroutine[Any, Any, RailResult]],
        direction: RailDirection,
    ) -> RailResult:
        """Run rail coroutines sequentially, short-circuiting on first unsafe result."""
        req_id = get_request_id()
        remaining = iter(rails.items())
        try:
            for flow, coro in remaining:
                result = await coro
                log.debug("[%s] %s flow %s result %s", req_id, direction.value, flow, result)
                if not result.is_safe:
                    log.info("[%s] %s flow %s blocked", req_id, direction.value, flow)
                    return result
            return RailResult(is_safe=True)
        finally:
            for _, coro in remaining:
                coro.close()

    async def _run_rails_parallel(
        self,
        rails: Mapping[str, Coroutine[Any, Any, RailResult]],
        direction: RailDirection,
    ) -> RailResult:
        """Run rail coroutines concurrently, cancelling remaining on first unsafe result."""
        req_id = get_request_id()
        task_to_flow: dict[asyncio.Task, str] = {asyncio.create_task(coro): flow for flow, coro in rails.items()}
        tasks = list(task_to_flow.keys())
        task_order = {task: i for i, task in enumerate(tasks)}
        pending_tasks: set[asyncio.Task] = set(tasks)

        try:
            while pending_tasks:
                done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in sorted(done, key=lambda t: task_order[t]):
                    result = task.result()
                    flow = task_to_flow[task]
                    log.debug("[%s] %s flow %s result %s", req_id, direction.value, flow, result)
                    if not result.is_safe:
                        log.info(
                            "[%s] %s flow %s blocked (cancelling %d remaining)",
                            req_id,
                            direction.value,
                            flow,
                            len(pending_tasks),
                        )
                        for t in pending_tasks:
                            t.cancel()
                        if pending_tasks:
                            await asyncio.wait(pending_tasks)
                        return result
            return RailResult(is_safe=True)
        except BaseException:
            for t in tasks:
                if not t.done():
                    t.cancel()
            alive = [t for t in tasks if not t.done()]
            if alive:
                await asyncio.wait(alive)
            raise

    async def _run_input_rail(self, flow: str, messages: list[dict]) -> RailResult:
        """Run an input rail flow if it's supported. If not raise an exception"""
        # Extract the base flow name (strip any $model=... parameter)
        base_flow = _get_flow_name(flow)

        if base_flow == "content safety check input":
            return await self._check_content_safety_input(flow, messages)
        elif base_flow == "topic safety check input":
            return await self._check_topic_safety_input(flow, messages)
        elif base_flow == "jailbreak detection model":
            return await self._check_jailbreak_detection(messages)
        else:
            raise RuntimeError(f"Input rail flow `{base_flow}` not supported")

    async def _run_output_rail(self, flow: str, messages: list[dict], response: str) -> RailResult:
        """Run an output rail flow if it's supported. If not raise an exception"""
        base_flow = _get_flow_name(flow)

        if base_flow == "content safety check output":
            return await self._check_content_safety_output(flow, messages, response)
        else:
            raise RuntimeError(f"Output rail flow `{base_flow}` not supported")

    async def _check_content_safety_input(self, flow: str, messages: list[dict]) -> RailResult:
        """Check input content safety via the content_safety model."""

        model_type = _get_flow_model(flow)
        if not model_type:
            raise RuntimeError(f"Model not specified for content-safety input rail: {flow}")

        req_id = get_request_id()
        log.info("[%s] Checking content safety input via model '%s'", req_id, model_type)

        last_user_content = self._last_user_content(messages)
        prompt_key = self._flow_to_prompt_key(flow)
        prompt_content = self._render_prompt(prompt_key, user_input=last_user_content)
        log.debug("[%s] Content safety input prompt: %s", req_id, truncate(prompt_content))

        try:
            response_text = await self.model_manager.generate_async(
                model_type, [{"role": "user", "content": prompt_content}]
            )
            log.debug("[%s] Content safety input response: %s", req_id, truncate(response_text))

            result = self._parse_content_safety_input_response(response_text)
            return result

        except Exception as e:
            log.error("[%s] Content safety input check failed: %s", req_id, e)
            return RailResult(is_safe=False, reason=f"Content safety input check error: {e}")

    async def _check_content_safety_output(self, flow: str, messages: list[dict], response: str) -> RailResult:
        """Check output content safety via the content_safety model."""
        model_type = _get_flow_model(flow)
        if not model_type:
            raise RuntimeError(f"Model not specified for content-safety output rail: {flow}")

        req_id = get_request_id()
        log.info("[%s] Checking content safety output via model '%s'", req_id, model_type)

        last_user_content = self._last_user_content(messages)
        prompt_key = self._flow_to_prompt_key(flow)
        prompt_content = self._render_prompt(prompt_key, user_input=last_user_content, bot_response=response)
        log.debug("[%s] Content safety output prompt: %s", req_id, truncate(prompt_content))

        try:
            response_text = await self.model_manager.generate_async(
                model_type, [{"role": "user", "content": prompt_content}]
            )
            log.debug("[%s] Content safety output response: %s", req_id, truncate(response_text))

            result = self._parse_content_safety_output_response(response_text)
            return result

        except Exception as e:
            log.error("[%s] Content safety output check failed: %s", req_id, e)
            return RailResult(is_safe=False, reason=f"Content safety output check error: {e}")

    async def _check_topic_safety_input(self, flow: str, messages: list[dict]) -> RailResult:
        """Check topic safety via the topic_control model.

        Unlike content safety which sends a single rendered prompt, topic control
        sends a system message (guidelines) plus the full conversation history.
        This matches the library action behavior which includes all prior turns
        so the model has context for follow-up messages.
        """
        model_type = _get_flow_model(flow)
        if not model_type:
            raise RuntimeError(f"Model not specified for topic-safety input rail: {flow}")

        req_id = get_request_id()
        log.info("[%s] Checking topic safety input via model '%s'", req_id, model_type)

        last_user_content = self._last_user_content(messages)
        prompt_key = self._flow_to_prompt_key(flow)
        system_prompt = self._render_topic_safety_prompt(prompt_key)
        log.debug("[%s] Topic safety input user content: %s", req_id, truncate(last_user_content))

        try:
            response_text = await self.model_manager.generate_async(
                model_type,
                [
                    {"role": "system", "content": system_prompt},
                    *messages,
                ],
                temperature=TOPIC_SAFETY_TEMPERATURE,
                max_tokens=TOPIC_SAFETY_MAX_TOKENS,
            )
            log.debug("[%s] Topic safety input response: %s", req_id, truncate(response_text))
            return self._parse_topic_safety_response(response_text)

        except Exception as e:
            log.error("[%s] Topic safety input check failed: %s", req_id, e)
            return RailResult(is_safe=False, reason=f"Topic safety input check error: {e}")

    async def _check_jailbreak_detection(self, messages: list[dict]) -> RailResult:
        """Check for jailbreak attempts by calling the jailbreak detection APIEngine."""
        req_id = get_request_id()
        log.info("[%s] Checking jailbreak detection", req_id)

        last_user_content = self._last_user_content(messages)
        log.debug("[%s] Jailbreak detection input: %s", req_id, truncate(last_user_content))

        try:
            response = await self.model_manager.api_call("jailbreak_detection", {"input": last_user_content})
            log.debug("[%s] Jailbreak detection response: %s", req_id, truncate(response))
            return self._parse_jailbreak_response(response)

        except Exception as e:
            log.error("[%s] Jailbreak detection check failed: %s", req_id, e)
            return RailResult(is_safe=False, reason=f"Jailbreak detection check error: {e}")

    @staticmethod
    def _parse_jailbreak_response(response: dict) -> RailResult:
        """Convert a {"jailbreak": bool} API response to a RailResult.
        Response looks like: {"jailbreak": true, "score": 0.6599113682063298}
        """
        if "jailbreak" not in response:
            raise RuntimeError(f"Jailbreak detection response missing 'jailbreak' field: {response}")

        jailbreak_detected = response["jailbreak"]
        score = response.get("score", "unknown")
        if jailbreak_detected:
            return RailResult(is_safe=False, reason=f"Score: {score}")
        return RailResult(is_safe=True, reason=f"Score: {score}")

    def _render_topic_safety_prompt(self, prompt_key: str) -> str:
        """Look up a topic safety prompt and append the output restriction suffix.

        The topic safety prompt template is the system message containing policy
        guidelines.  Unlike content safety prompts it does NOT contain
        ``{{ user_input }}`` — the user input is sent as a separate message.
        """
        prompt_template = self.prompts.get(prompt_key)
        if not prompt_template or not prompt_template.content:
            raise RuntimeError(f"No prompt template found for key {prompt_key}")

        system_prompt = prompt_template.content.strip()
        if not system_prompt.endswith(TOPIC_SAFETY_OUTPUT_RESTRICTION):
            system_prompt = f"{system_prompt}\n\n{TOPIC_SAFETY_OUTPUT_RESTRICTION}"
        return system_prompt

    @staticmethod
    def _parse_topic_safety_response(response: str) -> RailResult:
        """LLM response of "off-topic" is unsafe, anything else is safe. Return RailsResult."""
        if response.lower().strip() == "off-topic":
            return RailResult(is_safe=False, reason="Topic safety: off-topic")
        return RailResult(is_safe=True)

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

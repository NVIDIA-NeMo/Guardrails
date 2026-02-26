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

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

if TYPE_CHECKING:
    from langgraph.runtime import Runtime as LangGraphRuntime
from nemoguardrails.integrations.langchain.exceptions import GuardrailViolation
from nemoguardrails.integrations.langchain.message_utils import (
    create_ai_message,
    is_ai_message,
    is_human_message,
    messages_to_dicts,
)
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.options import RailsResult, RailStatus, RailType
from nemoguardrails.utils import get_or_create_event_loop

log = logging.getLogger(__name__)


class GuardrailsMiddleware(AgentMiddleware):
    def __init__(
        self,
        config_path: Optional[str] = None,
        config_yaml: Optional[str] = None,
        raise_on_violation: bool = False,
        blocked_input_message: str = "I cannot process this request due to content policy.",
        blocked_output_message: str = "I cannot provide this response due to content policy.",
        enable_input_rails: bool = True,
        enable_output_rails: bool = True,
    ):
        if config_path is not None:
            config = RailsConfig.from_path(config_path)
        elif config_yaml is not None:
            config = RailsConfig.from_content(config_yaml)
        else:
            raise ValueError("Either 'config_path' or 'config_yaml' must be provided to GuardrailsMiddleware")

        self.rails = LLMRails(config=config)
        self.raise_on_violation = raise_on_violation
        self.blocked_input_message = blocked_input_message
        self.blocked_output_message = blocked_output_message
        self.enable_input_rails = enable_input_rails
        self.enable_output_rails = enable_output_rails

    def _has_input_rails(self) -> bool:
        return len(self.rails.config.rails.input.flows) > 0

    def _has_output_rails(self) -> bool:
        return len(self.rails.config.rails.output.flows) > 0

    def _convert_to_rails_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        return messages_to_dicts(messages)

    def _get_last_user_message(self, messages: List[BaseMessage]) -> Optional[HumanMessage]:
        for msg in reversed(messages):
            if is_human_message(msg):
                return msg
        return None

    def _get_last_ai_message(self, messages: List[BaseMessage]) -> Optional[AIMessage]:
        for msg in reversed(messages):
            if is_ai_message(msg):
                return msg
        return None

    def _handle_guardrail_failure(
        self,
        result: RailsResult,
        rail_type: str,
        blocked_message: str,
    ) -> None:
        if result.status == RailStatus.BLOCKED:
            failure_message = f"{rail_type.capitalize()} blocked by {result.rail or 'unknown rail'}"

            if self.raise_on_violation:
                raise GuardrailViolation(
                    message=failure_message,
                    result=result,
                    rail_type=rail_type,
                )

            log.warning(failure_message)

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        if not self.enable_input_rails or not self._has_input_rails():
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        rails_messages = self._convert_to_rails_messages(messages)

        try:
            result = await self.rails.check_async(rails_messages, rail_types=[RailType.INPUT])

            if result.status == RailStatus.BLOCKED:
                self._handle_guardrail_failure(
                    result=result,
                    rail_type="input",
                    blocked_message=self.blocked_input_message,
                )
                blocked_msg = create_ai_message(self.blocked_input_message)
                return {"messages": messages + [blocked_msg], "jump_to": "end"}

            return None

        except GuardrailViolation:
            raise
        except Exception as e:
            log.error(f"Error checking input rails: {e}", exc_info=True)

            if self.raise_on_violation:
                raise GuardrailViolation(
                    message=f"Input rail execution error: {str(e)}",
                    rail_type="input",
                )

            blocked_msg = create_ai_message(self.blocked_input_message)
            return {"messages": messages + [blocked_msg], "jump_to": "end"}

    def _replace_last_ai_message(self, messages: list, replacement: AIMessage) -> list:
        for i in range(len(messages) - 1, -1, -1):
            if is_ai_message(messages[i]):
                return messages[:i] + [replacement] + messages[i + 1 :]
        return messages + [replacement]

    async def aafter_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        if not self.enable_output_rails or not self._has_output_rails():
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last_ai_message = self._get_last_ai_message(messages)
        if not last_ai_message:
            return None

        rails_messages = self._convert_to_rails_messages(messages)

        try:
            result = await self.rails.check_async(rails_messages, rail_types=[RailType.OUTPUT])

            if result.status == RailStatus.BLOCKED:
                self._handle_guardrail_failure(
                    result=result,
                    rail_type="output",
                    blocked_message=self.blocked_output_message,
                )
                blocked_msg = create_ai_message(self.blocked_output_message)
                return {"messages": self._replace_last_ai_message(messages, blocked_msg)}

            return None

        except GuardrailViolation:
            raise
        except Exception as e:
            log.error(f"Error checking output rails: {e}", exc_info=True)

            if self.raise_on_violation:
                raise GuardrailViolation(
                    message=f"Output rail execution error: {str(e)}",
                    rail_type="output",
                )

            blocked_msg = create_ai_message(self.blocked_output_message)
            return {"messages": self._replace_last_ai_message(messages, blocked_msg)}

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        if not self.enable_input_rails or not self._has_input_rails():
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        loop = get_or_create_event_loop()
        return loop.run_until_complete(self.abefore_model(state, runtime))

    def after_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        if not self.enable_output_rails or not self._has_output_rails():
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last_ai_message = self._get_last_ai_message(messages)
        if not last_ai_message:
            return None

        loop = get_or_create_event_loop()
        return loop.run_until_complete(self.aafter_model(state, runtime))


class InputRailsMiddleware(GuardrailsMiddleware):
    def __init__(
        self,
        config_path: Optional[str] = None,
        config_yaml: Optional[str] = None,
        raise_on_violation: bool = False,
        blocked_input_message: str = "I cannot process this request due to content policy.",
    ):
        super().__init__(
            config_path=config_path,
            config_yaml=config_yaml,
            raise_on_violation=raise_on_violation,
            blocked_input_message=blocked_input_message,
            blocked_output_message="",
            enable_input_rails=True,
            enable_output_rails=False,
        )

    async def aafter_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        return None

    def after_agent(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        return None


class OutputRailsMiddleware(GuardrailsMiddleware):
    def __init__(
        self,
        config_path: Optional[str] = None,
        config_yaml: Optional[str] = None,
        raise_on_violation: bool = False,
        blocked_output_message: str = "I cannot provide this response due to content policy.",
    ):
        super().__init__(
            config_path=config_path,
            config_yaml=config_yaml,
            raise_on_violation=raise_on_violation,
            blocked_input_message="",
            blocked_output_message=blocked_output_message,
            enable_input_rails=False,
            enable_output_rails=True,
        )

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        return None

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        return None

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
import logging
from typing import Any, Optional

from nemoguardrails.guardrails.guardrails_types import LLMMessages, RailResult, get_request_id
from nemoguardrails.guardrails.rail_action import RailAction, _rail_llm_call_var
from nemoguardrails.guardrails.telemetry import action_span, record_span_error
from nemoguardrails.llm.clients._errors import _redact_secrets
from nemoguardrails.manifests.surface_reference import parse_configured_surface
from nemoguardrails.types import ToolCall

log = logging.getLogger(__name__)


def _parse_verdict(text: str) -> bool:
    """Return True (safe/ALLOW) if last non-empty line is ALLOW; False (block) for BLOCK.

    Strips trailing punctuation and whitespace before comparing so responses
    like "BLOCK." or "ALLOW " are handled correctly. Fails closed: returns
    False when no clear verdict is found.
    """
    for line in reversed(text.strip().splitlines()):
        token = line.strip().rstrip(".!: ").upper()
        if token == "ALLOW":
            return True
        if token == "BLOCK":
            return False
    return False


class PerToolCheckAction(RailAction):
    """LLM-based per-tool policy check for IORails.

    Evaluates a single tool call or tool result against a named prompt task.
    Flow name format: ``check tool call $check=<task_name>`` with optional
    ``$model=<model_type>`` (defaults to ``"main"``).
    """

    action_name = "check tool call"
    fallback_model = "main"
    requires_model = False

    def _extract_messages(self, messages: LLMMessages, bot_response: Optional[str]) -> dict[str, Any]:
        raise NotImplementedError

    def _create_prompt(self, model_type: Optional[str], extracted: dict[str, Any]) -> Any:
        raise NotImplementedError

    async def _get_response(self, model_type: Optional[str], prompt: Any) -> Any:
        raise NotImplementedError

    def _parse_response(self, response: Any) -> RailResult:
        text = (response or "").strip()
        is_safe = _parse_verdict(text)
        if is_safe:
            return RailResult(is_safe=True)
        return RailResult(is_safe=False, reason=f"check tool call blocked: {text}")

    # run() is overridden directly because the base class signature does not
    # support per-tool kwargs (tool_call, tool_name, tool_result); the four
    # abstract pipeline methods are unused stubs.
    async def run(
        self,
        flow: str,
        messages: LLMMessages,
        bot_response: Optional[str] = None,
        *,
        tool_call: Optional[ToolCall] = None,
        tool_name: Optional[str] = None,
        tool_result: Optional[str] = None,
    ) -> RailResult:
        """Evaluate a tool call or result against the named prompt task."""
        _rail_llm_call_var.set(None)
        with action_span(self._tracer, self.action_name) as span:
            req_id = get_request_id()

            _, params = parse_configured_surface(flow)
            check_name = params.get("check")
            if not check_name:
                return RailResult(
                    is_safe=False,
                    reason=f"'check tool call' requires a $check= parameter in flow '{flow}'",
                )

            model_type = params.get("model") or self.fallback_model

            tool_call_json = ""
            if tool_call is not None:
                tool_call_json = json.dumps(tool_call.to_dict())

            user_message = self._last_user_content_or_empty(messages)

            context: dict[str, Any] = {
                "tool_call": tool_call_json,
                "tool_name": tool_name or "",
                "tool_result": tool_result or "",
                "user_message": user_message,
            }

            try:
                prompt = self.task_manager.render_task_prompt(task=check_name, context=context)
            except Exception as e:
                log.error("[%s] check tool call: prompt render failed for task '%s': %s", req_id, check_name, e)
                return RailResult(is_safe=False, reason=_redact_secrets(f"prompt render error: {e}"))

            prompt_messages = self._prompt_to_messages(prompt)
            log.debug("[%s] check tool call: task=%s model=%s", req_id, check_name, model_type)

            try:
                response = await self._get_llm_response(model_type, prompt_messages)
                text = (response.content or "").strip()
                log.debug("[%s] check tool call: response=%r", req_id, text)
                return self._parse_response(text)
            except Exception as e:
                record_span_error(span, e)
                log.error("[%s] check tool call failed: %s", req_id, e)
                return RailResult(is_safe=False, reason=_redact_secrets(f"check tool call error: {e}"))

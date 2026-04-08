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

"""Base class for IORails rail actions.

Defines the template-method pipeline: extract → prompt → respond → parse.
Subclasses override individual steps. The base provides three concrete response
helpers for the common call patterns (LLM, API, local).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from nemoguardrails.guardrails.guardrails_types import (
    LLMMessages,
    RailResult,
    get_request_id,
    truncate,
)
from nemoguardrails.guardrails.model_manager import ModelManager
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import _get_flow_model, _get_flow_name

log = logging.getLogger(__name__)


class RailAction(ABC):
    """Base class for all IORails rail actions.

    Subclasses implement the abstract ``_``-prefixed hooks to customise each
    stage of the pipeline.  The public entry point is :meth:`run`.

    Subclasses must define these class attributes:
      - action_name: The base flow name as it appears in RailsConfig
        (e.g. ``"content safety check input"``).
      - fallback_model: Model to use when the flow has no ``$model=`` parameter.
        ``None`` means no fallback.
      - requires_model: Whether a resolved model_type is mandatory.  When True
        (default) and no model can be resolved, ``run()`` raises immediately.
    """

    action_name: str
    fallback_model: Optional[str] = None
    requires_model: bool = True

    def __init__(self, model_manager: ModelManager, task_manager: LLMTaskManager) -> None:
        self.model_manager = model_manager
        self.task_manager = task_manager

    async def run(
        self,
        flow: str,
        messages: LLMMessages,
        bot_response: Optional[str] = None,
    ) -> RailResult:
        """Execute the full rail pipeline and return a safety result."""
        req_id = get_request_id()
        base_flow = _get_flow_name(flow)
        self._validate_flow_name(base_flow)

        model_type = self._get_model_type(flow)
        if self.requires_model and not model_type:
            raise RuntimeError(f"No $model= specified for '{base_flow}' and no fallback_model defined")

        extracted = self._extract_messages(messages, bot_response)
        log.debug("[%s] %s extracted: %s", req_id, base_flow, truncate(extracted))

        prompt = self._create_prompt(model_type, extracted)
        if prompt is not None:
            log.debug("[%s] %s prompt: %s", req_id, base_flow, truncate(prompt))

        try:
            response = await self._get_response(model_type, prompt)
            log.debug("[%s] %s response: %s", req_id, base_flow, truncate(response))
            return self._parse_response(response)
        except Exception as e:
            log.error("[%s] %s failed: %s", req_id, base_flow, e)
            return RailResult(is_safe=False, reason=f"{base_flow} error: {e}")

    def _get_model_type(self, flow: str) -> Optional[str]:
        """Extract model from the flow's ``$model=`` parameter, falling back to :attr:`fallback_model`."""
        return _get_flow_model(flow) or self.fallback_model

    @abstractmethod
    def _extract_messages(
        self,
        messages: LLMMessages,
        bot_response: Optional[str],
    ) -> dict[str, Any]:
        """Extract the relevant fields from messages into a dict.

        Returns a dict of extracted values that will be passed to _create_prompt.
        """

    @abstractmethod
    def _create_prompt(
        self,
        model_type: Optional[str],
        extracted: dict[str, Any],
    ) -> Any:
        """Build the prompt / request payload from extracted data.

        Returns whatever _get_response needs: a message list, a dict body, etc.
        May return None if the response step doesn't need a prompt (e.g. API calls
        that build their own payload).
        """

    @abstractmethod
    async def _get_response(
        self,
        model_type: Optional[str],
        prompt: Any,
    ) -> Any:
        """Call the model/API/local engine and return the raw response."""

    @abstractmethod
    def _parse_response(self, response: Any) -> RailResult:
        """Convert the raw response into a RailResult."""

    async def _get_llm_response(
        self,
        model_type: Optional[str],
        messages: list[dict],
        **kwargs: Any,
    ) -> str:
        """Call an LLM via ModelManager and return the response text."""
        if not model_type:
            raise RuntimeError("model_type is required for LLM calls")
        return await self.model_manager.generate_async(model_type, messages, **kwargs)

    async def _get_api_response(
        self,
        api_name: str,
        body: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call an API endpoint via ModelManager and return the response dict."""
        return await self.model_manager.api_call(api_name, body, **kwargs)

    async def _get_local_response(self, **kwargs: Any) -> Any:
        """Run a local/in-process check. Override in subclasses that need it."""
        raise NotImplementedError("Subclass must override _get_local_response")

    def _validate_flow_name(self, base_flow: str | None) -> None:
        """Verify the flow's base name matches this action's action_name."""
        if not base_flow:
            raise RuntimeError("No flow name found")

        if base_flow != self.action_name:
            raise RuntimeError(f"Flow '{base_flow}' does not match expected action_name '{self.action_name}'")

    @staticmethod
    def _last_user_content(messages: LLMMessages) -> str:
        """Return the content of the last user message."""
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                return msg["content"]
        raise RuntimeError(f"No user message found in: {messages}")

    @staticmethod
    def _prompt_to_messages(prompt: Union[str, list[dict]]) -> list[dict]:
        """Convert LLMTaskManager render output to role/content message format."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return [{"role": m["type"], "content": m["content"]} for m in prompt]

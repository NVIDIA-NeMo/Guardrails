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
import time
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

from nemoguardrails.guardrails.api_engine import APIEngineError
from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.guardrails_types import (
    LLMMessages,
    RailResult,
    get_request_id,
    serialize_prompt,
    truncate,
)
from nemoguardrails.guardrails.model_engine import ModelEngineError
from nemoguardrails.guardrails.telemetry import action_span, record_span_error
from nemoguardrails.llm.clients._errors import _sanitize
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import _get_flow_model, _get_flow_name
from nemoguardrails.types import LLMResponse, UsageInfo

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RailLLMCall:
    """Usage/model/timing captured for the call a rail made, for GenerationLog.

    Set by :meth:`RailAction._get_llm_response` (LLM rails) or
    :meth:`RailAction._get_api_response` (API rails, e.g. jailbreak — usage/model None),
    and read by RailsManager right after the rail runs (same async task), so the caller
    can build the per-rail ``RailCallRecord``. ``started_at``/``finished_at`` are
    wall-clock (``time.time()``) timestamps; ``duration`` is a monotonic delta.
    """

    usage: Optional[UsageInfo]
    llm_model_name: Optional[str]
    request_id: Optional[str]
    provider_name: Optional[str]
    prompt: Optional[str]
    completion: Optional[str]
    started_at: float
    finished_at: float
    duration: float


# Request-scoped: the last LLM call a rail made. ``RailAction.run`` clears it at the
# start of each rail so a rail that makes no model call leaves it None.
_rail_llm_call_var: ContextVar[Optional[RailLLMCall]] = ContextVar("rail_llm_call", default=None)


def get_and_clear_rail_llm_call_contextvar() -> Optional[RailLLMCall]:
    """Return and clear the LLM call captured for the rail that just ran."""
    call = _rail_llm_call_var.get()
    _rail_llm_call_var.set(None)
    return call


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

    def __init__(
        self,
        engine_registry: EngineRegistry,
        task_manager: LLMTaskManager,
        tracer: Optional["Tracer"] = None,
    ) -> None:
        """Store the engine registry, task manager, and optional tracer for subclass hooks."""
        self.engine_registry = engine_registry
        self.task_manager = task_manager
        self._tracer = tracer

    async def run(
        self,
        flow: str,
        messages: LLMMessages,
        bot_response: Optional[str] = None,
    ) -> RailResult:
        """Execute the full rail pipeline and return a safety result."""
        # Clear any capture from a prior rail on this task; a rail that makes no model
        # call then leaves it None and produces a record with no LLM call.
        _rail_llm_call_var.set(None)
        with action_span(self._tracer, self.action_name) as span:
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
            except (ModelEngineError, APIEngineError) as e:
                # Record an error on the OTEL span
                record_span_error(span, e)
                if e.status is not None:
                    log.error("[%s] %s failed (HTTP %d): %s", req_id, base_flow, e.status, e)
                    raise
                log.error("[%s] %s failed: %s", req_id, base_flow, e)
                # The reason reaches the client through the streaming violation
                # payload, so scrub upstream URLs as well as secrets.
                return RailResult(is_safe=False, reason=_sanitize(f"{base_flow} error: {e}"))
            except Exception as e:
                # Record an error on the OTEL span
                record_span_error(span, e)
                log.error("[%s] %s failed: %s", req_id, base_flow, e)
                return RailResult(is_safe=False, reason=_sanitize(f"{base_flow} error: {e}"))

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
    ) -> LLMResponse:
        """Call an LLM via EngineRegistry and return the structured response.

        Captures usage/model/timing into a request-scoped contextvar so RailsManager can
        record this call in the GenerationLog after the rail runs. The capture happens in a
        ``finally`` so a call that raises is still recorded as an attempt (usage/model/
        completion left None), matching LLMRails counting a failed call.
        """
        if not model_type:
            raise RuntimeError("model_type is required for LLM calls")
        started_at = time.time()
        t0 = time.monotonic()
        usage: Optional[UsageInfo] = None
        model_name: Optional[str] = None
        request_id: Optional[str] = None
        completion: Optional[str] = None
        try:
            response = await self.engine_registry.model_call(model_type, messages, **kwargs)
            usage = response.usage
            model_name = response.model
            request_id = response.request_id
            completion = response.content
            return response
        finally:
            _rail_llm_call_var.set(
                RailLLMCall(
                    usage=usage,
                    llm_model_name=model_name,
                    request_id=request_id,
                    provider_name=self.engine_registry.provider_name(model_type),
                    prompt=serialize_prompt(messages),
                    completion=completion,
                    started_at=started_at,
                    finished_at=time.time(),
                    duration=time.monotonic() - t0,
                )
            )

    async def _get_api_response(
        self,
        api_name: str,
        body: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call an API endpoint via EngineRegistry and return the response dict.

        Records the call (with no token usage or model name) so API-backed rails such as
        jailbreak detection still appear in the GenerationLog's ``llm_calls`` and counts,
        matching LLMRails. Recorded in a ``finally`` so a call that raises is still counted
        as an attempt.
        """
        started_at = time.time()
        t0 = time.monotonic()
        try:
            return await self.engine_registry.api_call(api_name, body, **kwargs)
        finally:
            _rail_llm_call_var.set(
                RailLLMCall(
                    usage=None,
                    llm_model_name=None,
                    request_id=None,
                    provider_name=None,
                    prompt=None,
                    completion=None,
                    started_at=started_at,
                    finished_at=time.time(),
                    duration=time.monotonic() - t0,
                )
            )

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
    def _last_user_content_or_empty(messages: LLMMessages) -> str:
        """Return the content of the last user message, or "" when there is none.

        Output checks evaluate the bot response; the user prompt only adds context
        and may legitimately be absent (for example an output-only ``check`` on an
        assistant message). Unlike :meth:`_last_user_content`, this does not raise.
        """
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                return msg["content"]
        return ""

    @staticmethod
    def _prompt_to_messages(prompt: Union[str, list[dict]]) -> list[dict]:
        """Convert LLMTaskManager render output to role/content message format."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return [{"role": m["type"], "content": m["content"]} for m in prompt]

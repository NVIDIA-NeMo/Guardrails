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

Orchestrates input/output safety checks by delegating to RailAction instances.
Rails run sequentially by default; the first failing rail short-circuits.
When parallel mode is enabled, all rails run concurrently and the first
unsafe result cancels remaining rails immediately.
"""

import asyncio
import logging
from collections.abc import Coroutine, Mapping
from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union

from nemoguardrails.guardrails.actions.content_safety_action import (
    ContentSafetyInputAction,
    ContentSafetyOutputAction,
)
from nemoguardrails.guardrails.actions.jailbreak_detection_action import JailbreakDetectionAction
from nemoguardrails.guardrails.actions.tool_call_action import ToolCallRailAction
from nemoguardrails.guardrails.actions.tool_result_action import ToolResultRailAction
from nemoguardrails.guardrails.actions.topic_safety_action import TopicSafetyInputAction
from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.guardrails_types import (
    RailDirection,
    RailResult,
    get_request_id,
)
from nemoguardrails.guardrails.rail_action import RailAction
from nemoguardrails.guardrails.telemetry import mark_rail_stop, rail_span, set_rail_content
from nemoguardrails.guardrails.tool_rail_action import ToolRailAction
from nemoguardrails.guardrails.tool_schema import ToolExchange, Toolset
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import _get_flow_name
from nemoguardrails.types import ToolCall

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

log = logging.getLogger(__name__)

# All known RailAction subclasses, keyed by their action_name.
_ACTION_CLASSES: dict[str, type[RailAction]] = {
    cls.action_name: cls
    for cls in [
        ContentSafetyInputAction,
        ContentSafetyOutputAction,
        TopicSafetyInputAction,
        JailbreakDetectionAction,
    ]
}

# All known ToolRailAction subclasses, keyed by their action_name. Tool rails are
# local structural/schema validators (model-free) and so are registered separately
# from the LLM/API-call-shaped RailAction classes above.
_TOOL_ACTION_CLASSES: dict[str, type[ToolRailAction]] = {
    cls.action_name: cls
    for cls in [
        ToolCallRailAction,
        ToolResultRailAction,
    ]
}

_ToolActionT = TypeVar("_ToolActionT", bound=ToolRailAction)


class RailsManager:
    """Orchestrates input and output safety checks for IORails.

    Reads the rails configuration to determine which checks are enabled,
    instantiates the corresponding RailAction for each flow, then runs
    them sequentially or in parallel.
    """

    def __init__(
        self,
        *,
        engine_registry: EngineRegistry,
        task_manager: LLMTaskManager,
        input_flows: list[str],
        output_flows: list[str],
        input_parallel: bool = False,
        output_parallel: bool = False,
        tool_call_flows: Optional[list[str]] = None,
        tool_result_flows: Optional[list[str]] = None,
        tracer: Optional["Tracer"] = None,
        content_capture_enabled: bool = False,
    ) -> None:
        """Build RailAction instances for each configured input and output flow.

        When *tracer* is provided, rail and action executions produce OTEL
        spans; when ``None`` the span helpers become no-ops.

        When *content_capture_enabled* is True, rail spans carry the
        rail's input messages (``guardrails.rail.input``) and the block
        reason (``guardrails.rail.reason``) when the rail rejects the
        request.  Defaults to False; only meaningful when ``tracer`` is
        also set.
        """
        self.engine_registry = engine_registry
        self.task_manager = task_manager
        self._tracer = tracer
        self._content_capture_enabled = content_capture_enabled

        self.input_flows: list[str] = list(input_flows)
        self.output_flows: list[str] = list(output_flows)

        self.input_parallel: bool = input_parallel
        self.output_parallel: bool = output_parallel

        self.tool_call_flows: list[str] = list(tool_call_flows or [])
        self.tool_result_flows: list[str] = list(tool_result_flows or [])

        # Build action instances for each configured flow
        self._actions: dict[str, RailAction] = {}
        for flow in self.input_flows + self.output_flows:
            base_name = _get_flow_name(flow) or flow
            self._actions[flow] = self._create_action(base_name)

        # Tool Call Actions run on tool invocations from the main LLM response
        # Tool Result Actions run on the results of executing Tool Calls in the harness
        self._tool_call_actions = self._build_tool_actions(self.tool_call_flows, ToolCallRailAction)
        self._tool_result_actions = self._build_tool_actions(self.tool_result_flows, ToolResultRailAction)

        log.info(
            "RailsManager initialized: input_flows=%s, output_flows=%s, tool_call_flows=%s, "
            "tool_result_flows=%s, input_parallel=%s, output_parallel=%s",
            self.input_flows,
            self.output_flows,
            self.tool_call_flows,
            self.tool_result_flows,
            self.input_parallel,
            self.output_parallel,
        )

    def _create_action(self, base_name: str) -> RailAction:
        """Instantiate the RailAction for a given flow base name."""
        action_cls = _ACTION_CLASSES.get(base_name)
        if action_cls is None:
            available = sorted(_ACTION_CLASSES.keys())
            raise RuntimeError(f"Rail flow '{base_name}' not supported. Available: {available}")
        return action_cls(self.engine_registry, self.task_manager, tracer=self._tracer)

    def _build_tool_actions(self, flows: list[str], expected_cls: type[_ToolActionT]) -> dict[str, _ToolActionT]:
        """Instantiate the tool rails for *flows*, checking each resolves to *expected_cls*.

        Raises ``RuntimeError`` on a duplicate flow, an unknown flow, or a flow that
        resolves to the wrong direction. Duplicates are rejected because the dispatch
        keys its coroutine map by flow, so a repeated flow would silently drop a run.
        """
        actions: dict[str, _ToolActionT] = {}
        for flow in flows:
            if flow in actions:
                raise RuntimeError(f"Duplicate tool rail flow '{flow}' is not supported")
            base_name = _get_flow_name(flow) or flow
            action_cls = _TOOL_ACTION_CLASSES.get(base_name)
            if action_cls is None:
                available = sorted(_TOOL_ACTION_CLASSES.keys())
                raise RuntimeError(f"Tool rail flow '{base_name}' not supported. Available: {available}")
            action = action_cls(tracer=self._tracer)
            if not isinstance(action, expected_cls):
                raise RuntimeError(
                    f"Tool rail flow '{flow}' resolved to {type(action).__name__}, expected {expected_cls.__name__}"
                )
            actions[flow] = action
        return actions

    async def is_input_safe(self, messages: list[dict], *, enabled: Union[bool, list[str]] = True) -> RailResult:
        """Run the enabled input rails, short-circuiting on the first failure.

        The per-request *enabled* toggle selects which configured input rails run:
        ``True`` (the default) runs all, ``False`` runs none, and a list runs only the
        named flows (matched on the normalized flow name). When parallel mode is enabled,
        all selected rails run concurrently and the first unsafe result cancels the rest.
        """
        active = self._enabled_flows(self.input_flows, enabled)
        if not active:
            return RailResult(is_safe=True)

        rails = {flow: self._run_rail(flow, RailDirection.INPUT, messages) for flow in active}
        if self.input_parallel:
            return await self._run_rails_parallel(rails, RailDirection.INPUT)
        return await self._run_rails_sequential(rails, RailDirection.INPUT)

    async def is_output_safe(
        self, messages: list[dict], response: str, *, enabled: Union[bool, list[str]] = True
    ) -> RailResult:
        """Run the enabled output rails, short-circuiting on the first failure.

        The per-request *enabled* toggle selects which configured output rails run:
        ``True`` (the default) runs all, ``False`` runs none, and a list runs only the
        named flows (matched on the normalized flow name). When parallel mode is enabled,
        all selected rails run concurrently and the first unsafe result cancels the rest.
        """
        active = self._enabled_flows(self.output_flows, enabled)
        if not active:
            return RailResult(is_safe=True)

        rails = {flow: self._run_rail(flow, RailDirection.OUTPUT, messages, bot_response=response) for flow in active}
        if self.output_parallel:
            return await self._run_rails_parallel(rails, RailDirection.OUTPUT)
        return await self._run_rails_sequential(rails, RailDirection.OUTPUT)

    async def are_tool_calls_safe(
        self,
        tool_calls: list[ToolCall],
        llm_params: Optional[dict],
        *,
        enabled: Union[bool, list[str]] = True,
        model_type: str = "main",
    ) -> RailResult:
        """Validate the model's emitted tool calls (OUTPUT-direction tool rail).

        The tool-call counterpart to :meth:`is_output_safe`: takes the model's output
        (``tool_calls``) plus the request's declared tools (``llm_params``) and returns
        a ``RailResult``.
        """
        active = self._enabled_flows(list(self._tool_call_actions), enabled)
        if not active or not tool_calls:
            return RailResult(is_safe=True)
        try:
            toolset = self.engine_registry.parse_tools(model_type, llm_params)
        except Exception as e:
            log.warning("[%s] tool parsing failed; blocking tool calls: %s", get_request_id(), e)
            return RailResult(is_safe=False, reason=f"tool parsing failed: {e}")

        rails = {flow: self._run_tool_call_rail(flow, tool_calls, toolset) for flow in active}
        return await self._run_rails_sequential(rails, RailDirection.OUTPUT)

    async def are_tool_results_safe(
        self,
        messages: list[dict],
        *,
        enabled: Union[bool, list[str]] = True,
        model_type: str = "main",
    ) -> RailResult:
        """Validate incoming tool results (INPUT-direction tool rail).

        The tool-result counterpart to :meth:`is_input_safe`: takes the conversation
        ``messages`` and returns a ``RailResult``. Groups the conversation into per-turn
        ``(calls, results)`` exchanges via the engine adapter and validates each result
        against its own turn's calls, so call ids reused across turns (spec-allowed) are
        not flagged as ambiguous duplicates.
        """
        active = self._enabled_flows(list(self._tool_result_actions), enabled)
        if not active:
            return RailResult(is_safe=True)
        try:
            exchanges = self.engine_registry.extract_tool_exchanges(model_type, messages)
        except Exception as e:
            log.warning("[%s] tool exchange extraction failed; blocking: %s", get_request_id(), e)
            return RailResult(is_safe=False, reason=f"tool exchange extraction failed: {e}")
        if not any(exchange.results for exchange in exchanges):
            return RailResult(is_safe=True)

        rails = {flow: self._run_tool_result_rail(flow, exchanges) for flow in active}
        return await self._run_rails_sequential(rails, RailDirection.INPUT)

    @staticmethod
    def _enabled_flows(configured: list[str], enabled: Union[bool, list[str]]) -> list[str]:
        """Resolve the per-request enable toggle into the configured flows to run.

        ``True`` (the default) runs every configured flow; ``False`` runs none; a list
        runs only the named flows that are configured, preserving configured order and
        ignoring unknown names. The two booleans are spelled out as separate cases so a
        non-empty list is never mistaken for ``True``.

        List membership is compared on the normalized flow name (``_get_flow_name``),
        the same way ``_create_action``, ``_build_tool_actions`` and ``unsupported_reason``
        do, so a request toggle carrying the canonical rail name matches a configured flow
        that carries a ``$model=``/``(...)`` suffix instead of silently dropping it
        (fail-open). Shared by the input, output, and tool rail families.
        """
        if enabled is True:
            return list(configured)
        if enabled is False:
            return []
        requested = {_get_flow_name(name) or name for name in enabled}
        return [flow for flow in configured if (_get_flow_name(flow) or flow) in requested]

    async def _run_rail(
        self,
        flow: str,
        direction: RailDirection,
        messages: list[dict],
        bot_response: Optional[str] = None,
    ) -> RailResult:
        """Dispatch a single rail flow to its RailAction instance."""
        with rail_span(self._tracer, flow, direction) as span:
            action = self._actions[flow]
            result = await action.run(flow, messages, bot_response)
            mark_rail_stop(span, result.is_safe)
            # Capture rail input + block reason after the action runs.
            # RailAction.run() catches its own exceptions and returns
            # RailResult(is_safe=False, reason=...), so this branch is
            # reached even on action errors and the error reason gets
            # recorded as the block reason.
            if self._content_capture_enabled:
                set_rail_content(
                    span,
                    {"messages": messages, "bot_response": bot_response},
                    reason=result.reason if not result.is_safe else None,
                )
            return result

    async def _run_tool_call_rail(self, flow: str, tool_calls: list[ToolCall], toolset: Toolset) -> RailResult:
        """Dispatch a single tool-call rail to its action, wrapped in an OUTPUT rail span."""
        with rail_span(self._tracer, flow, RailDirection.OUTPUT) as span:
            result = await self._tool_call_actions[flow].run(toolset, tool_calls)
            mark_rail_stop(span, result.is_safe)
            if self._content_capture_enabled:
                set_rail_content(
                    span,
                    {"tool_calls": [tc.to_dict() for tc in tool_calls]},
                    reason=result.reason if not result.is_safe else None,
                )
            return result

    async def _run_tool_result_rail(self, flow: str, exchanges: list[ToolExchange]) -> RailResult:
        """Validate each turn's results against that turn's calls, wrapped in an INPUT rail span.

        Each exchange is validated independently so ``call_id`` linkage stays turn-local;
        the first unsafe exchange short-circuits.
        """
        action = self._tool_result_actions[flow]
        with rail_span(self._tracer, flow, RailDirection.INPUT) as span:
            result = RailResult(is_safe=True)
            for exchange in exchanges:
                result = await action.run(exchange.results, exchange.calls)
                if not result.is_safe:
                    break
            mark_rail_stop(span, result.is_safe)
            if self._content_capture_enabled:
                all_results = [r for exchange in exchanges for r in exchange.results]
                set_rail_content(
                    span,
                    {
                        "tool_results": [
                            {"call_id": r.call_id, "name": r.name, "is_error": r.is_error} for r in all_results
                        ]
                    },
                    reason=result.reason if not result.is_safe else None,
                )
            return result

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

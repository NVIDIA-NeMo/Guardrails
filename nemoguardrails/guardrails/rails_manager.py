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
from typing import TYPE_CHECKING, Any, Optional

from nemoguardrails.guardrails.actions.content_safety_action import (
    ContentSafetyInputAction,
    ContentSafetyOutputAction,
)
from nemoguardrails.guardrails.actions.jailbreak_detection_action import JailbreakDetectionAction
from nemoguardrails.guardrails.actions.topic_safety_action import TopicSafetyInputAction
from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.guardrails_types import (
    RailDirection,
    RailResult,
    get_request_id,
)
from nemoguardrails.guardrails.rail_action import RailAction
from nemoguardrails.guardrails.telemetry import mark_rail_stop, rail_span, set_rail_content
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import _get_flow_name

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

        # Build action instances for each configured flow
        self._actions: dict[str, RailAction] = {}
        for flow in self.input_flows + self.output_flows:
            base_name = _get_flow_name(flow) or flow
            self._actions[flow] = self._create_action(base_name)

        log.info(
            "RailsManager initialized: input_flows=%s, output_flows=%s, input_parallel=%s, output_parallel=%s",
            self.input_flows,
            self.output_flows,
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

    async def is_input_safe(self, messages: list[dict]) -> RailResult:
        """Run all enabled input rails, short-circuiting on the first failure.

        When parallel mode is enabled, all rails run concurrently and the first
        unsafe result cancels remaining rails.
        """
        if not self.input_flows:
            return RailResult(is_safe=True)

        rails = {flow: self._run_rail(flow, RailDirection.INPUT, messages) for flow in self.input_flows}
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

        rails = {
            flow: self._run_rail(flow, RailDirection.OUTPUT, messages, bot_response=response)
            for flow in self.output_flows
        }
        if self.output_parallel:
            return await self._run_rails_parallel(rails, RailDirection.OUTPUT)
        return await self._run_rails_sequential(rails, RailDirection.OUTPUT)

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

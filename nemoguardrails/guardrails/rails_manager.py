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

"""Rails manager for IORails: runs input/output checks through compiled, manifest-driven rails."""

import asyncio
import logging
import warnings
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union

from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.guardrails.actions.tool_call_action import ToolCallRailAction
from nemoguardrails.guardrails.actions.tool_result_action import ToolResultRailAction
from nemoguardrails.guardrails.compiled_rail import CompiledRail, RailDependencies, compile_rail
from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.guardrails_types import (
    RailCallRecord,
    RailDirection,
    RailResult,
    display_reason,
    get_request_id,
    last_user_content,
    rewrite_user_message,
)
from nemoguardrails.guardrails.telemetry import mark_rail_stop, rail_span, set_rail_content
from nemoguardrails.guardrails.tool_rail_action import ToolRailAction
from nemoguardrails.guardrails.tool_schema import ToolExchange, Toolset
from nemoguardrails.http.runtime import create_http_client
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.manifests import RailDirection as SurfaceDirection
from nemoguardrails.rails.llm.config import _get_flow_model, _get_flow_name
from nemoguardrails.types import ToolCall, UsageInfo

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

    from nemoguardrails.logging.explain import LLMCallInfo

log = logging.getLogger(__name__)

# IORails labels its directions for logging; the manifest catalog keys surfaces by its own.
_SURFACE_DIRECTIONS = {
    RailDirection.INPUT: SurfaceDirection.INPUT,
    RailDirection.OUTPUT: SurfaceDirection.OUTPUT,
}

# All known ToolRailAction subclasses, keyed by their action_name. Tool rails are
# local structural/schema validators (model-free) and so are registered separately
# from the manifest-driven rails compiled above.
_TOOL_ACTION_CLASSES: dict[str, type[ToolRailAction]] = {
    cls.action_name: cls
    for cls in [
        ToolCallRailAction,
        ToolResultRailAction,
    ]
}

_ToolActionT = TypeVar("_ToolActionT", bound=ToolRailAction)

# The conversation variable each direction's rails may rewrite; anything else is refused.
_REWRITABLE_TARGET = {
    RailDirection.INPUT: TransformTarget.USER_MESSAGE,
    RailDirection.OUTPUT: TransformTarget.BOT_MESSAGE,
}


def _rewriting_flows(
    rails: Mapping[tuple[RailDirection, str], CompiledRail],
    direction: RailDirection,
    flows: Sequence[str],
) -> tuple[str, ...]:
    """The configured flows whose surface declares they may rewrite the text they check."""
    return tuple(flow for flow in flows if rails[(direction, flow)].transform_target is not None)


def _transforms_first(rewriting: Sequence[str], flows: Sequence[str]) -> list[str]:
    """Order rails that may rewrite ahead of rails that only judge, so a mask reaches the check."""
    ordered = list(rewriting)
    already_ordered = set(rewriting)
    ordered.extend(flow for flow in flows if flow not in already_ordered)
    return ordered


def _rail_result(outcome: RailOutcome) -> RailResult:
    """Map an engine-neutral rail verdict onto IORails' rail result."""
    return RailResult(outcome)


def _tool_rail_result(outcome: RailOutcome, flow: str) -> RailResult:
    """Map a tool rail's verdict, refusing a rewrite it has nowhere to apply."""
    if outcome.is_transform:
        # Reading it as "not blocked" would allow the request and drop the rewrite unseen.
        raise NotImplementedError(f"tool rail {flow!r} returned a rewrite, which IORails cannot apply")
    return RailResult(outcome)


def _rewritten_text(outcome: RailOutcome, direction: RailDirection, flow: str) -> str:
    """Return what a rail rewrote this direction's variable to, ignoring any it also named.

    A rail guarding both sides of a turn names both, and each direction takes its own, as the
    Colang flows do with the same verdict.
    """
    target = _REWRITABLE_TARGET[direction]
    rewrites = outcome.transform_text
    if target.value not in rewrites:
        # An action naming none of its direction's variable contradicts its own manifest.
        raise NotImplementedError(
            f"{flow!r} rewrote {sorted(rewrites)}, which a {direction.value.lower()} rail cannot apply; "
            f"it may rewrite {target.value!r}"
        )
    return rewrites[target.value]


def _refuse_concurrent_rewrite(result: RailResult, flow: str) -> None:
    """Refuse a rewrite from a concurrent rail: peers read the arriving text, so it cannot compose."""
    if result.outcome.is_transform:
        raise NotImplementedError(f"{flow!r} returned a rewrite while running in parallel, which cannot be applied")


def _checked_text(direction: RailDirection, messages: list[dict], bot_response: Optional[str]) -> str:
    """Return the text this direction's rails check, which is the text a rewrite replaces."""
    if direction is RailDirection.INPUT:
        return last_user_content(messages)
    return bot_response or ""


def _result_after_rewrites(
    direction: RailDirection,
    original_text: str,
    final_text: str,
    records: tuple[RailCallRecord, ...],
) -> RailResult:
    """Report the text that survived every rail, so a rewrite undone by a later rail is no rewrite."""
    if final_text == original_text:
        return RailResult.allow(records=records)
    return RailResult(RailOutcome.transform([(_REWRITABLE_TARGET[direction], final_text)]), records=records)


def _model_free_record(flow: str, rail_type: str, result: RailResult) -> RailCallRecord:
    """Build the per-rail GenerationLog record for a rail that reached no model."""
    base_name = _get_flow_name(flow) or flow
    model = _get_flow_model(flow)
    action_name = base_name.replace(" ", "_")
    return RailCallRecord(
        flow=flow,
        rail_type=rail_type,
        is_safe=result.is_safe,
        made_call=False,
        action_name=action_name,
        return_value=result.return_value,
        task=f"{action_name} $model={model}" if model else action_name,
    )


def _merge_llm_call_info(record: RailCallRecord, call: "LLMCallInfo") -> RailCallRecord:
    """Merge a captured model call's usage, naming and timing into *record*."""
    return replace(
        record,
        made_call=True,
        request_id=call.request_id,
        usage=UsageInfo(
            input_tokens=call.prompt_tokens or 0,
            output_tokens=call.completion_tokens or 0,
            total_tokens=call.total_tokens or 0,
        ),
        llm_model_name=call.llm_model_name,
        llm_provider_name=call.llm_provider_name,
        prompt=call.prompt,
        completion=call.completion,
        started_at=call.started_at,
        finished_at=call.finished_at,
        duration=call.duration,
    )


def _rail_call_record(
    flow: str, rail_type: str, result: RailResult, calls: Sequence["LLMCallInfo"] = ()
) -> RailCallRecord:
    """Build the per-rail GenerationLog record from a rail's result and the calls it made."""
    record = _model_free_record(flow, rail_type, result)
    if not calls:
        return record
    if len(calls) > 1:
        # RailCallRecord holds one call. No in-scope rail makes two, so say so rather than drop silently.
        log.warning("[%s] rail %s made %d model calls; recording only the last", get_request_id(), flow, len(calls))
    return _merge_llm_call_info(record, calls[-1])


class RailsManager:
    """Compiles a manifest-driven rail per configured flow and runs them sequentially or in parallel."""

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
        """Compile a manifest-driven rail for each configured input and output flow."""
        # Both telemetry settings are no-ops without a tracer: the span helpers do nothing, and
        # content capture writes the rail's input and block reason onto spans that do not exist.
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

        deps = self._rail_dependencies()
        # Keyed by direction as well as flow: compilation is direction-specific, so a surface
        # the catalog offers in both directions and a config lists in both would otherwise
        # collide on one key and run whichever compiled last.
        self._rails: dict[tuple[RailDirection, str], CompiledRail] = {}
        configured = ((RailDirection.INPUT, self.input_flows), (RailDirection.OUTPUT, self.output_flows))
        for direction, flows in configured:
            for flow in flows:
                self._rails[(direction, flow)] = compile_rail(flow, _SURFACE_DIRECTIONS[direction], deps)

        # Which rails may rewrite decides whether this config can run its rails concurrently at
        # all, which is settled once, here. The order they run in is not: it follows from the
        # configured flows, which stay the single source of truth for what runs.
        self.transform_flows: dict[RailDirection, tuple[str, ...]] = {
            direction: _rewriting_flows(self._rails, direction, flows) for direction, flows in configured
        }
        if any(self.transform_flows.values()):
            self._disable_parallel_execution()

        # Tool Call Actions run on tool invocations from the main LLM response
        # Tool Result Actions run on the results of executing Tool Calls in the harness
        self._tool_call_actions = self._build_tool_actions(self.tool_call_flows, ToolCallRailAction)
        self._tool_result_actions = self._build_tool_actions(self.tool_result_flows, ToolResultRailAction)

        self._http_client = create_http_client()
        runtime_deps = replace(deps, http_client=self._http_client)
        self._rails = {key: rail.with_runtime_dependencies(runtime_deps) for key, rail in self._rails.items()}

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

    def _disable_parallel_execution(self) -> None:
        """Turn both directions sequential, because concurrent rails cannot carry a rewrite."""
        if not (self.input_parallel or self.output_parallel):
            return
        rewriting = sorted(flow for flows in self.transform_flows.values() for flow in flows)
        warnings.warn(
            f"rails.input.parallel / rails.output.parallel are not honored alongside a rail that rewrites "
            f"content ({', '.join(rewriting)}); input and output rails run sequentially so each rewrite "
            f"reaches the rails behind it.",
            stacklevel=3,
        )
        self.input_parallel = False
        self.output_parallel = False

    def _rail_dependencies(self) -> RailDependencies:
        """Bundle the collaborators a compiled rail's action may declare as parameters."""
        return RailDependencies(
            llms=self.engine_registry.llms,
            llm_task_manager=self.task_manager,
            config=self.task_manager.config,
            tracer=self._tracer,
        )

    async def stop(self) -> None:
        """Close the runtime-owned pooled HTTP client."""
        await self._http_client.close()

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
        active = self._flows_to_run(RailDirection.INPUT, self.input_flows, enabled)
        if not active:
            return RailResult.allow()

        if self.input_parallel:
            rails = {flow: self._run_rail(flow, RailDirection.INPUT, messages) for flow in active}
            return await self._run_rails_parallel(rails, RailDirection.INPUT)
        return await self._run_rails_sequential(active, RailDirection.INPUT, messages)

    async def is_output_safe(
        self, messages: list[dict], response: str, *, enabled: Union[bool, list[str]] = True
    ) -> RailResult:
        """Run the enabled output rails, short-circuiting on the first failure.

        The per-request *enabled* toggle selects which configured output rails run:
        ``True`` (the default) runs all, ``False`` runs none, and a list runs only the
        named flows (matched on the normalized flow name). When parallel mode is enabled,
        all selected rails run concurrently and the first unsafe result cancels the rest.
        """
        active = self._flows_to_run(RailDirection.OUTPUT, self.output_flows, enabled)
        if not active:
            return RailResult.allow()

        if self.output_parallel:
            rails = {flow: self._run_rail(flow, RailDirection.OUTPUT, messages, response) for flow in active}
            return await self._run_rails_parallel(rails, RailDirection.OUTPUT)
        return await self._run_rails_sequential(active, RailDirection.OUTPUT, messages, response)

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
            return RailResult.allow()
        try:
            toolset = self.engine_registry.parse_tools(model_type, llm_params)
        except Exception as e:
            log.warning("[%s] tool parsing failed; blocking tool calls: %s", get_request_id(), e)
            return RailResult.block(reason=f"tool parsing failed: {e}")

        rails = {flow: self._run_tool_call_rail(flow, tool_calls, toolset) for flow in active}
        return await self._run_tool_rails_sequential(rails, RailDirection.OUTPUT)

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
            return RailResult.allow()
        try:
            exchanges = self.engine_registry.extract_tool_exchanges(model_type, messages)
        except Exception as e:
            log.warning("[%s] tool exchange extraction failed; blocking: %s", get_request_id(), e)
            return RailResult.block(reason=f"tool exchange extraction failed: {e}")
        if not any(exchange.results for exchange in exchanges):
            return RailResult.allow()

        rails = {flow: self._run_tool_result_rail(flow, exchanges) for flow in active}
        return await self._run_tool_rails_sequential(rails, RailDirection.INPUT)

    def _flows_to_run(
        self, direction: RailDirection, configured: list[str], enabled: Union[bool, list[str]]
    ) -> list[str]:
        """The flows this request runs, in the order it runs them: rewriting rails first."""
        active = self._enabled_flows(configured, enabled)
        return _transforms_first(_rewriting_flows(self._rails, direction, active), active)

    @staticmethod
    def _enabled_flows(configured: list[str], enabled: Union[bool, list[str]]) -> list[str]:
        """Resolve the per-request toggle into the configured flows to run: True all, False none."""
        # Membership is compared on the normalized flow name, as compile_rail and
        # unsupported_reason do, so a toggle naming the canonical rail still matches a configured
        # flow carrying a $model= suffix rather than silently dropping it (fail-open).
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
        """Dispatch a single rail flow to its compiled rail and record what it did."""
        with rail_span(self._tracer, flow, direction) as span:
            rail_execution = await self._rails[(direction, flow)].execute(messages, bot_response)
            result = _rail_result(rail_execution.outcome)
            if not result.is_safe:
                result = replace(result, triggered_rail=_get_flow_name(flow) or flow)
            records = (_rail_call_record(flow, direction.value.lower(), result, rail_execution.llm_calls),)
            result = replace(result, records=records)
            mark_rail_stop(span, result.is_safe)
            # CompiledRail converts an action error into a blocking outcome, so this branch is
            # reached on failures too and the redacted error text becomes the block reason.
            if self._content_capture_enabled:
                set_rail_content(
                    span,
                    {"messages": messages, "bot_response": bot_response},
                    reason=display_reason(result) if not result.is_safe else None,
                )
            return result

    async def _run_tool_call_rail(self, flow: str, tool_calls: list[ToolCall], toolset: Toolset) -> RailResult:
        """Dispatch a single tool-call rail to its action, wrapped in an OUTPUT rail span."""
        with rail_span(self._tracer, flow, RailDirection.OUTPUT) as span:
            result = _tool_rail_result(await self._tool_call_actions[flow].run(toolset, tool_calls), flow)
            result = replace(result, records=(_rail_call_record(flow, "tool_output", result),))
            mark_rail_stop(span, result.is_safe)
            if self._content_capture_enabled:
                set_rail_content(
                    span,
                    {"tool_calls": [tc.to_dict() for tc in tool_calls]},
                    reason=display_reason(result) if not result.is_safe else None,
                )
            return result

    async def _run_tool_result_rail(self, flow: str, exchanges: list[ToolExchange]) -> RailResult:
        """Validate each turn's results against that turn's calls, wrapped in an INPUT rail span.

        Each exchange is validated independently so ``call_id`` linkage stays turn-local;
        the first unsafe exchange short-circuits.
        """
        action = self._tool_result_actions[flow]
        with rail_span(self._tracer, flow, RailDirection.INPUT) as span:
            outcome = RailOutcome.allow()
            for exchange in exchanges:
                outcome = await action.run(exchange.results, exchange.calls)
                if outcome.is_blocked:
                    break
            result = _tool_rail_result(outcome, flow)
            result = replace(result, records=(_rail_call_record(flow, "tool_input", result),))
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
                    reason=display_reason(result) if not result.is_safe else None,
                )
            return result

    async def _run_rails_sequential(
        self,
        flows: Sequence[str],
        direction: RailDirection,
        messages: list[dict],
        bot_response: Optional[str] = None,
    ) -> RailResult:
        """Run a direction's rails in turn, threading each rewrite into the rails behind it."""
        req_id = get_request_id()
        collected: list[RailCallRecord] = []
        original_text = _checked_text(direction, messages, bot_response)
        final_text = original_text
        for flow in flows:
            result = await self._run_rail(flow, direction, messages, bot_response)
            collected.extend(result.records)
            log.debug("[%s] %s flow %s result %s", req_id, direction.value, flow, result)
            if not result.is_safe:
                log.info("[%s] %s flow %s blocked", req_id, direction.value, flow)
                return replace(result, records=tuple(collected))
            if result.outcome.is_transform:
                final_text = _rewritten_text(result.outcome, direction, flow)
                log.info("[%s] %s flow %s rewrote the text it checked", req_id, direction.value, flow)
                if direction is RailDirection.INPUT:
                    try:
                        messages = rewrite_user_message(messages, final_text)
                    except ValueError:
                        # Blocking keeps a misbehaving rail inside the fail-closed envelope,
                        # rather than failing the request as a server error.
                        log.error(
                            "[%s] %s flow %s rewrote a turn this request does not have", req_id, direction.value, flow
                        )
                        return RailResult.block(
                            reason="a rail rewrote a message this request does not have",
                            triggered_rail=_get_flow_name(flow) or flow,
                            records=tuple(collected),
                        )
                else:
                    bot_response = final_text
        return _result_after_rewrites(direction, original_text, final_text, tuple(collected))

    async def _run_tool_rails_sequential(
        self,
        rails: Mapping[str, Coroutine[Any, Any, RailResult]],
        direction: RailDirection,
    ) -> RailResult:
        """Run tool rail coroutines in turn, short-circuiting on the first unsafe result.

        Separate from the input/output loop because tool rails carry no text between them, so
        their coroutines are built up front and the unreached ones need closing.
        """
        req_id = get_request_id()
        remaining = iter(rails.items())
        collected: list[RailCallRecord] = []
        try:
            for flow, coro in remaining:
                result = await coro
                collected.extend(result.records)
                log.debug("[%s] %s flow %s result %s", req_id, direction.value, flow, result)
                if not result.is_safe:
                    log.info("[%s] %s flow %s blocked", req_id, direction.value, flow)
                    return replace(result, records=tuple(collected))
            return RailResult.allow(records=tuple(collected))
        finally:
            for _, coro in remaining:
                coro.close()

    async def _run_rails_parallel(
        self,
        rails: Mapping[str, Coroutine[Any, Any, RailResult]],
        direction: RailDirection,
    ) -> RailResult:
        """Run rail coroutines concurrently; on the first unsafe result, finish draining its
        completion batch (so rails that finished alongside it still contribute their records)
        before cancelling the rest."""
        req_id = get_request_id()
        task_to_flow: dict[asyncio.Task, str] = {asyncio.create_task(coro): flow for flow, coro in rails.items()}
        tasks = list(task_to_flow.keys())
        task_order = {task: i for i, task in enumerate(tasks)}
        pending_tasks: set[asyncio.Task] = set(tasks)
        collected: list[RailCallRecord] = []

        try:
            while pending_tasks:
                done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                first_unsafe: Optional[RailResult] = None
                for task in sorted(done, key=lambda t: task_order[t]):
                    result = task.result()
                    flow = task_to_flow[task]
                    _refuse_concurrent_rewrite(result, flow)
                    collected.extend(result.records)
                    log.debug("[%s] %s flow %s result %s", req_id, direction.value, flow, result)
                    if not result.is_safe and first_unsafe is None:
                        first_unsafe = result
                        log.info("[%s] %s flow %s blocked", req_id, direction.value, flow)
                if first_unsafe is not None:
                    if pending_tasks:
                        log.info("[%s] %s cancelling %d remaining", req_id, direction.value, len(pending_tasks))
                        for t in pending_tasks:
                            t.cancel()
                        await asyncio.wait(pending_tasks)
                    return replace(first_unsafe, records=tuple(collected))
            return RailResult.allow(records=tuple(collected))
        except BaseException:
            for t in tasks:
                if not t.done():
                    t.cancel()
            alive = [t for t in tasks if not t.done()]
            if alive:
                await asyncio.wait(alive)
            raise

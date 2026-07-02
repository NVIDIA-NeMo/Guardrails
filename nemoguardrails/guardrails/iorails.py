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

"""Optimized IORails Engine for specific guardrail configurations.

This module provides an optimized inference path for guardrail configurations that
only use specific supported flows (input/output content safety). For configurations
outside this supported set, the standard LLMRails engine should be used instead.
"""

import asyncio
import json
import logging
import time
import warnings
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import nullcontext, suppress
from typing import TYPE_CHECKING, Any, Optional, Union

from nemoguardrails.actions.llm.utils import _extract_and_remove_think_tags
from nemoguardrails.base_guardrails import BaseGuardrails
from nemoguardrails.exceptions import StreamingNotSupportedError
from nemoguardrails.guardrails.async_work_queue import AsyncWorkQueue
from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.guardrails_types import (
    LLMMessage,
    LLMMessages,
    RailDirection,
    get_request_id,
    truncate,
)
from nemoguardrails.guardrails.rails_manager import RailsManager
from nemoguardrails.guardrails.telemetry import (
    are_metrics_enabled,
    get_tracer,
    is_content_capture_enabled,
    is_tracing_enabled,
    record_nonstream_rejected,
    record_request_blocked,
    record_request_error,
    record_span_error,
    record_stream_rejected,
    register_nonstream_saturation_gauges,
    request_metrics,
    set_request_content,
    set_speculative_span_attrs,
    stream_active_metric,
    traced_request,
)
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.buffer import get_buffer_strategy
from nemoguardrails.rails.llm.config import RailsConfig, _get_flow_name
from nemoguardrails.rails.llm.options import GenerationOptions
from nemoguardrails.streaming import END_OF_STREAM, StreamingHandler
from nemoguardrails.tracing.constants import GuardrailsAttributes
from nemoguardrails.types import LLMModel, LLMResponse, ToolCall

if TYPE_CHECKING:
    from opentelemetry.trace import Span

log = logging.getLogger(__name__)

REFUSAL_MESSAGE = "I'm sorry, I can't respond to that."

# Concurrency budgets for the non-streaming AsyncWorkQueue:
# NONSTREAM_QUEUE_DEPTH      — max pending items before submit raises QueueFull
# NONSTREAM_MAX_CONCURRENCY  — max concurrent worker tasks draining the queue
NONSTREAM_QUEUE_DEPTH = 256
NONSTREAM_MAX_CONCURRENCY = 256

# Concurrency budget for streaming requests (separate from the non-streaming
# AsyncWorkQueue — streams have no admission buffer, just fail-fast on the
# semaphore).
STREAM_MAX_CONCURRENCY = 256

# Error type used by _generation_task when pushing error JSON into the stream
_GENERATION_ERROR_TYPE = "generation_error"


def _is_stream_error_chunk(chunk: Union[str, dict]) -> bool:
    """True when a streamed chunk is an error/violation payload.

    Covers both the ``generation_error`` payload pushed on a generation failure
    and the ``guardrails_violation`` payload emitted when output rails block.
    Handles plain-string chunks and the ``{"text": ...}`` frames produced when
    ``include_metadata=True``. The cheap ``"error"`` substring guard keeps the
    per-chunk hot path from JSON-parsing ordinary text tokens.
    """
    text = chunk.get("text") if isinstance(chunk, dict) else chunk
    if not isinstance(text, str) or '"error"' not in text:
        return False
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(parsed, dict) and "error" in parsed


def _serialize_tool_calls(tool_calls: list[ToolCall]) -> list[dict]:
    """Serialize ToolCall objects to OpenAI /chat/completions shape.

    ``function.arguments`` is emitted as a JSON string (OpenAI-native) rather
    than the canonical dict carried internally, so the output round-trips
    through OpenAI-compatible clients.
    """
    return [
        {
            "id": tool_call.id,
            "type": tool_call.type,
            "function": {
                "name": tool_call.function.name,
                "arguments": json.dumps(tool_call.function.arguments),
            },
        }
        for tool_call in tool_calls
    ]


def _frame_for_stream(payload: str, include_metadata: Optional[bool]) -> Union[str, dict]:
    """Frame a directly-yielded payload to match the surrounding stream's chunk shape.

    Returns a ``{"text": payload}`` dict under ``include_metadata``, the raw string
    otherwise — the same wrapping the StreamingHandler applies to ``push_chunk``'d
    strings, so terminal and block chunks that bypass the handler stay shape-consistent.
    """
    return {"text": payload} if include_metadata else payload


def _terminal_tool_call_chunk(
    tool_calls: list[ToolCall], include_metadata: Optional[bool]
) -> tuple[str, Union[str, dict]]:
    """Frame assembled tool calls as the stream's terminal chunk.

    Returns ``(payload, framed)``: ``payload`` is the OpenAI-native
    ``{"tool_calls": ...}`` JSON string used for content capture, and
    ``framed`` is what to yield — a ``{"text": payload}`` dict under
    ``include_metadata``, a raw string otherwise — matching the shape of
    the surrounding stream.
    """
    payload = json.dumps({"tool_calls": _serialize_tool_calls(tool_calls)})
    return payload, _frame_for_stream(payload, include_metadata)


def _build_assistant_message(content: str, tool_calls: Optional[list[ToolCall]]) -> LLMMessage:
    """Build the assistant message returned by ``generate``.

    Without tool calls this is the existing ``{"role", "content"}`` shape. With
    tool calls present, the calls are serialized to OpenAI shape and ``content``
    is set to ``None`` when empty, matching the OpenAI assistant-message contract.
    """
    if not tool_calls:
        return {"role": "assistant", "content": content}
    return {
        "role": "assistant",
        "content": content or None,
        "tool_calls": _serialize_tool_calls(tool_calls),
    }


def _coerce_generation_options(options: Optional[Union[dict, GenerationOptions]]) -> Optional[GenerationOptions]:
    """Normalize the request ``options`` argument into a ``GenerationOptions`` or None."""
    if isinstance(options, GenerationOptions):
        return options
    if isinstance(options, dict):
        return GenerationOptions(**options)
    return None


def _unsupported_flows_reason(flows: list[str], supported: frozenset[str], label: str) -> Optional[str]:
    """Return a fallback reason when any flow in *flows* is outside *supported*, else None.

    Each flow id is normalized (call args / ``$model=`` suffix stripped) before the
    membership check, so ``"content safety check input $model=x"`` matches the bare
    flow name. A flow whose name normalizes to empty carries no recognizable rail name
    and is ignored. *label* names the rail family in the message (e.g. ``"input"``,
    ``"tool output"``); offending names are reported sorted and de-duplicated.
    """
    unsupported = set()
    for flow in flows:
        name = _get_flow_name(flow)
        if name and name not in supported:
            unsupported.add(name)
    if not unsupported:
        return None
    return f"config has unsupported {label} flows: {sorted(unsupported)}"


def _duplicate_flows_reason(flows: list[str], label: str) -> Optional[str]:
    """Return a fallback reason when *flows* contains a duplicate flow, else None.

    A duplicate tool flow raises ``RuntimeError`` in RailsManager at construction, so
    surfacing it here lets the config route to LLMRails cleanly instead of failing init.
    Flow ids are normalized (call args / ``$model=`` suffix stripped) before comparison
    -- matching :func:`_unsupported_flows_reason` -- so two entries that differ only by a
    suffix the tool rails ignore are still caught as duplicates rather than running twice.
    A flow whose name normalizes to empty carries no recognizable rail name and is skipped.
    """
    seen = set()
    for flow in flows:
        name = _get_flow_name(flow)
        if not name:
            continue
        if name in seen:
            return f"config has duplicate {label} flows: {flows}"
        seen.add(name)
    return None


class IORails(BaseGuardrails):
    """Workflow engine for accelerated Input/Output rails inference."""

    # Rail sections and flows that this engine can handle. Configs using anything
    # outside these sets fall back to LLMRails.
    SUPPORTED_RAILS = frozenset({"input", "output", "config", "tool_input", "tool_output"})
    SUPPORTED_INPUT_FLOWS = frozenset(
        {"content safety check input", "topic safety check input", "jailbreak detection model"}
    )
    SUPPORTED_OUTPUT_FLOWS = frozenset({"content safety check output"})
    # Tool-rail flows are direction-specific: tool_output may only carry the
    # tool-call validator and tool_input only the tool-result validator. The
    # supported sets double as the direction check so a misdirected flow falls
    # back to LLMRails rather than raising in RailsManager at construction.
    SUPPORTED_TOOL_OUTPUT_FLOWS = frozenset({"tool call validation"})
    SUPPORTED_TOOL_INPUT_FLOWS = frozenset({"tool result validation"})

    @classmethod
    def unsupported_reason(cls, config: RailsConfig, llm: Optional[LLMModel] = None) -> Optional[str]:
        """Return None if IORails can handle (config, llm), else a human-readable reason."""
        if llm is not None:
            return "an `llm` argument was provided; IORails does not accept a custom LLM"

        if config.colang_version != "1.0":
            return f"IORails supports Colang 1.0 only; config uses Colang {config.colang_version}"

        unsupported_rails = sorted(config.rails.model_fields_set - cls.SUPPORTED_RAILS)
        if unsupported_rails:
            return f"config has rails outside the IORails-supported set: {unsupported_rails}"

        # Each rail family accepts only its own direction-specific flows, so an unknown
        # or misdirected flow routes the config to LLMRails. The supported sets double
        # as the direction check (tool_output allows only the call validator, etc.).
        flow_checks = (
            ("input", config.rails.input.flows, cls.SUPPORTED_INPUT_FLOWS),
            ("output", config.rails.output.flows, cls.SUPPORTED_OUTPUT_FLOWS),
            ("tool output", config.rails.tool_output.flows, cls.SUPPORTED_TOOL_OUTPUT_FLOWS),
            ("tool input", config.rails.tool_input.flows, cls.SUPPORTED_TOOL_INPUT_FLOWS),
        )
        for label, flows, supported in flow_checks:
            reason = _unsupported_flows_reason(flows, supported, label)
            if reason is not None:
                return reason

        # A duplicate tool flow raises RuntimeError in RailsManager at construction;
        # surface it here as a fallback reason so the config routes to LLMRails
        # cleanly instead of failing IORails init (matching how unsupported flows
        # are handled).
        for label, tool_flows in (
            ("tool output", config.rails.tool_output.flows),
            ("tool input", config.rails.tool_input.flows),
        ):
            reason = _duplicate_flows_reason(tool_flows, label)
            if reason is not None:
                return reason

        return None

    @classmethod
    def can_handle(cls, config: RailsConfig, llm: Optional[LLMModel] = None) -> bool:
        """Return True iff IORails can handle the given config and llm argument."""
        return cls.unsupported_reason(config, llm) is None

    def __init__(self, config: RailsConfig, *, _report_usage: bool = True) -> None:
        """Build the engine registry and rails manager from the given config."""
        self._running = False
        self.config = config

        # Create the OTEL tracer (if enabled in config).
        # Pass to EngineRegistry and RailsManager to keep all spans consistent under parent
        self._tracing_enabled = is_tracing_enabled(config.tracing)
        self._tracer = get_tracer() if self._tracing_enabled else None
        self._metrics_enabled = are_metrics_enabled(config.metrics)
        # Content capture only makes sense when tracing is on — there's no
        # point recording prompts/responses onto spans that won't be exported.
        # The flag itself is resolved from config + env var by the helper.
        self._content_capture_enabled = self._tracing_enabled and is_content_capture_enabled(config.tracing)

        self.engine_registry = EngineRegistry(
            config.models,
            config.rails.config,
            tracer=self._tracer,
            metrics_enabled=self._metrics_enabled,
            content_capture_enabled=self._content_capture_enabled,
        )
        # Tool rails are CPU-bound, run sequentially since we're not waiting on IO to complete
        if config.rails.tool_output.parallel or config.rails.tool_input.parallel:
            warnings.warn(
                "rails.tool_output.parallel / rails.tool_input.parallel are not honored by IORails; "
                "tool rails run sequentially.",
                stacklevel=2,
            )

        self.rails_manager = RailsManager(
            engine_registry=self.engine_registry,
            task_manager=LLMTaskManager(config),
            input_flows=config.rails.input.flows,
            output_flows=config.rails.output.flows,
            input_parallel=config.rails.input.parallel or False,
            output_parallel=config.rails.output.parallel or False,
            tool_call_flows=config.rails.tool_output.flows,
            tool_result_flows=config.rails.tool_input.flows,
            tracer=self._tracer,
            content_capture_enabled=self._content_capture_enabled,
        )
        self._speculative_generation = config.rails.input.speculative_generation or False
        self._speculative_max_buffered_tokens = config.rails.input.speculative_max_buffered_tokens

        # Non-streaming admission queue + worker pool (owned by IORails so
        # all request-path concurrency controls sit under one roof).  The
        # queue auto-starts lazily on first submit(); ``start()`` below
        # starts it explicitly alongside the engine registry.
        self._generate_async_queue = AsyncWorkQueue(
            name="iorails_generate_queue",
            max_queue_size=NONSTREAM_QUEUE_DEPTH,
            max_concurrency=NONSTREAM_MAX_CONCURRENCY,
            reject_on_full=True,
        )

        # Semaphore for streaming concurrency control / load shedding
        self._stream_semaphore = asyncio.Semaphore(STREAM_MAX_CONCURRENCY)

        # ObservableGauges are created lazily on first ``start()`` because
        # they need a reference to an AsyncWorkQueue which has been started.
        self._gauges_registered = False

        if _report_usage:
            from nemoguardrails.telemetry import RailsEngineEnum, report_usage

            report_usage(config, deployment_type="library", rails_engine=RailsEngineEnum.IORAILS.value)

    @property
    def _has_streaming_output_rails(self) -> bool:
        """True when output rails are configured and streaming is enabled for them."""
        streaming = self.config.rails.output.streaming
        return streaming is not None and streaming.enabled and len(self.config.rails.output.flows) > 0

    async def start(self) -> None:
        """Start the IORails engine. Call this during service startup."""
        if self._running:
            return

        #  The EngineRegistry cleans up all its Engines if there's an exception on startup
        #  so no need to catch exceptions and clean up here
        await self.engine_registry.start()
        try:
            await self._generate_async_queue.start()
            try:
                # Queue is now live; register the state-observing ObservableGauges.
                # ``lambda: self._running`` is checked at collect time so the gauges
                # report empty lists once the engine has been stopped.
                if self._metrics_enabled and not self._gauges_registered:
                    register_nonstream_saturation_gauges(
                        self._generate_async_queue,
                        is_running=lambda: self._running,
                    )
                    self._gauges_registered = True
            except BaseException:
                # Gauge registration failed after the queue was started — roll
                # the queue back so a retry of start() comes from a clean state
                # rather than leaving the queue running with ``_running=False``
                # (which would make stop() a no-op and leak worker tasks).
                try:
                    await self._generate_async_queue.stop()
                except BaseException:
                    log.exception("queue rollback failed during IORails.start()")
                raise
        except BaseException:
            # Log but suppress rollback failures so we propagate the original
            # queue-start (or gauge-registration) error as the actionable root cause.
            try:
                await self.engine_registry.stop()
            except BaseException:
                log.exception("engine_registry rollback failed during IORails.start()")
            raise

        self._running = True

    async def stop(self) -> None:
        """Stop the IORails engine. Call this during service shutdown."""
        if not self._running:
            return

        # Each shutdown step runs independently so a failure in one does not
        # leak the other. _running is cleared regardless so a retry of stop()
        # is a no-op and we don't leak worker tasks.
        try:
            try:
                await self._generate_async_queue.stop()
            finally:
                await self.engine_registry.stop()
        finally:
            self._running = False

    async def __aenter__(self):
        """Context manager (used for testing rather than long-lived instance)"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager (used for testing rather than long-lived instance)"""
        await self.stop()

    def generate(self, messages: LLMMessages, **kwargs) -> LLMMessage:
        """Synchronous version of generate_async.

        Telemetry is disabled for the ephemeral IORails object used for
        the ``generate()`` call. For production use, use the asynchronous
        `generate_async()` and `stream_async()` methods for non-streaming
        and streaming requests respectively.
        """

        # Disable tracing and metrics for synchronous generation calls
        sync_config = self.config.model_copy(deep=True)
        if sync_config.tracing is not None:
            sync_config.tracing.enabled = False
        if sync_config.metrics is not None:
            sync_config.metrics.enabled = False

        async def _run_sync_iorails():
            """Spin up a short-lived IORails engine for one synchronous generate call."""
            # Avoid counting this sync-API bridge as a separate user-created IORails instance.
            async with IORails(sync_config, _report_usage=False) as iorails_engine:
                return await iorails_engine.generate_async(messages, **kwargs)

        return asyncio.run(_run_sync_iorails())

    async def generate_async(self, messages: LLMMessages, **kwargs) -> LLMMessage:
        """Public entry: submit the request to the internal work queue.

        The queue enforces non-streaming concurrency limits
        (``NONSTREAM_MAX_CONCURRENCY`` workers draining up to
        ``NONSTREAM_QUEUE_DEPTH`` pending items).  Callers receive
        ``asyncio.QueueFull`` when the admission buffer is full and
        ``guardrails.nonstream.rejections`` increments if metrics are enabled.

        Request-level metrics (``guardrails.requests``,
        ``guardrails.request.duration``, ``guardrails.requests.errors``)
        wrap the queue submission, so duration includes queue-wait time
        (OTEL HTTP semconv).  A ``QueueFull`` rejection shows up in BOTH
        ``requests.errors{error.type=QueueFull}`` and
        ``nonstream.rejections`` — honest dual-signal reporting.
        """
        await self.start()
        metrics_ctx = request_metrics() if self._metrics_enabled else nullcontext()
        with metrics_ctx:
            try:
                return await self._generate_async_queue.submit(self._run_generate, messages, **kwargs)
            except asyncio.QueueFull:
                if self._metrics_enabled:
                    record_nonstream_rejected()
                raise

    async def _run_generate(self, messages: LLMMessages, **kwargs) -> LLMMessage:
        """Runs inside a queue worker task.  Wraps the pipeline in
        ``traced_request`` so each request gets its own span + request ID,
        then delegates to ``_do_generate`` for the actual input rails →
        LLM → output rails flow.  Metrics are emitted at the outer
        lifecycle scope by ``generate_async``, not here.
        """
        tracer = self._tracer if self._tracing_enabled else None
        with traced_request(tracer) as (request_span, req_id):
            t0 = time.monotonic()
            try:
                result = await self._do_generate(messages, req_id, request_span, **kwargs)
            except Exception:
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.error("[%s] generate_async failed time=%.1fms", req_id, elapsed_ms, exc_info=True)
                raise
            # Capture content once here at the traced_request boundary so any
            # future early-return added to _do_generate is covered automatically.
            if self._content_capture_enabled:
                set_request_content(request_span, messages, result.get("content"))
            elapsed_ms = (time.monotonic() - t0) * 1000
            log.info("[%s] generate_async completed time=%.1fms", req_id, elapsed_ms)
            return result

    @staticmethod
    def _guardrails_violation_payload(message: str, param: str) -> str:
        """Build the JSON error payload emitted when a streaming rail blocks the request.

        Shared by every streaming block path so they all surface the same
        ``guardrails_violation`` / ``content_blocked`` shape; ``param`` distinguishes which
        rail family blocked (``input_rails`` / ``tool_input_rails`` / ``tool_output_rails`` /
        ``output_rails``).
        """
        return json.dumps(
            {
                "error": {
                    "message": message,
                    "type": "guardrails_violation",
                    "param": param,
                    "code": "content_blocked",
                }
            }
        )

    async def _do_generate(
        self, messages: LLMMessages, req_id: str, request_span: Optional["Span"] = None, **kwargs
    ) -> LLMMessage:
        """Core pipeline: tool-result rails -> input rails -> LLM call -> tool-call + output rails."""
        log.info("[%s] generate_async called", req_id)
        log.debug("[%s] generate_async messages=%s", req_id, truncate(messages))

        options = _coerce_generation_options(kwargs.get("options"))
        # Pass llm_params (including tool definitions) unchanged to the LLM call.
        llm_kwargs = options.llm_params if (options and options.llm_params) else {}
        input_enabled = options.rails.input if options else True
        output_enabled = options.rails.output if options else True
        tool_input_enabled = options.rails.tool_input if options else True
        tool_output_enabled = options.rails.tool_output if options else True

        # Agent/client executes tool-calls and sends results to Main LLM with prior conversation history.
        # Symmetric with INPUT rails
        log.info("[%s] Running tool result rails", req_id)
        tool_result = await self.rails_manager.are_tool_results_safe(messages, enabled=tool_input_enabled)
        if not tool_result.is_safe:
            log.info("[%s] Tool result blocked: %s", req_id, tool_result.reason)
            if self._metrics_enabled:
                record_request_blocked(RailDirection.INPUT)
            return {"role": "assistant", "content": REFUSAL_MESSAGE}

        if self._speculative_generation:
            response = await self._do_generate_speculative(
                messages, req_id, llm_kwargs, request_span, input_enabled=input_enabled
            )
        else:
            response = await self._do_generate_sequential(messages, req_id, llm_kwargs, input_enabled=input_enabled)

        if response is None:
            return {"role": "assistant", "content": REFUSAL_MESSAGE}

        # Log raw content before reasoning extraction and think-token removal
        log.debug("[%s] Raw LLM response: %s", req_id, truncate(response.content))

        # Reasoning extraction prefers LLMResponse `reasoning` field if the provider
        # supports it, falling back to extracting <think>...</think> tags otherwise.
        # The fallback mutates response.content to remove reasoning content.
        reasoning_content = response.reasoning or _extract_and_remove_think_tags(response)
        response_text = response.content

        # Main LLM returns function calls to make based on available tools and conversation
        # Symmetric with OUTPUT rails
        if response.tool_calls:
            tool_call = await self.rails_manager.are_tool_calls_safe(
                response.tool_calls, llm_kwargs, enabled=tool_output_enabled
            )
            if not tool_call.is_safe:
                log.info("[%s] Tool call blocked: %s", req_id, tool_call.reason)
                if self._metrics_enabled:
                    record_request_blocked(RailDirection.OUTPUT)
                return {"role": "assistant", "content": REFUSAL_MESSAGE}

        # Output rails check the final answer, not reasoning traces.
        # Reasoning is re-attached as <think> tags only below so reasoning intentionally bypasses output
        # rails, matching LLMRails.
        # A tool-call-only response skips output rails (no text to check)
        # Tool calls have their own `ToolOutputRails` set of rails separate to `OutputRails`
        is_tool_call_only = bool(response.tool_calls) and not response_text
        if not is_tool_call_only:
            log.info("[%s] Running output rails", req_id)
            output_result = await self.rails_manager.is_output_safe(messages, response_text, enabled=output_enabled)
            if not output_result.is_safe:
                log.info("[%s] Output blocked: %s", req_id, output_result.reason)
                if self._metrics_enabled:
                    record_request_blocked(RailDirection.OUTPUT)
                return {"role": "assistant", "content": REFUSAL_MESSAGE}

        # TODO: Support returning GenerationResponse `reasoning_content` to match LLMRails
        # For now, embed the reasoning on the content with think-tags
        if reasoning_content:
            response_text = f"<think>{reasoning_content}</think>\n" + response_text

        return _build_assistant_message(response_text, response.tool_calls)

    async def _do_generate_sequential(
        self, messages: LLMMessages, req_id: str, llm_kwargs: dict, *, input_enabled: Union[bool, list[str]] = True
    ) -> Optional[LLMResponse]:
        """Sequential path: input rails block before LLM generation starts."""
        log.info("[%s] Running input rails", req_id)
        input_result = await self.rails_manager.is_input_safe(messages, enabled=input_enabled)
        if not input_result.is_safe:
            log.info("[%s] Input blocked: %s", req_id, input_result.reason)
            if self._metrics_enabled:
                record_request_blocked(RailDirection.INPUT)
            return None

        log.info("[%s] Calling main LLM", req_id)
        return await self.engine_registry.model_call("main", messages, **llm_kwargs)

    async def _do_generate_speculative(
        self,
        messages: LLMMessages,
        req_id: str,
        llm_kwargs: dict,
        request_span: Optional["Span"] = None,
        *,
        input_enabled: Union[bool, list[str]] = True,
    ) -> Optional[LLMResponse]:
        """Speculative path: input rails and LLM generation race concurrently."""
        log.info("[%s] Speculative generation: launching input rails + LLM concurrently", req_id)

        rails_task = asyncio.create_task(self.rails_manager.is_input_safe(messages, enabled=input_enabled))
        gen_task = asyncio.create_task(self.engine_registry.model_call("main", messages, **llm_kwargs))

        try:
            response = await self._parallel_input_rail_and_response_generation(
                rails_task, gen_task, req_id, request_span
            )
        except BaseException as outer_exc:
            for t in (rails_task, gen_task):
                if not t.done():
                    t.cancel()
            # Drain all tasks (including done) to retrieve their exceptions and
            # avoid asyncio "Task exception was never retrieved" warnings, then
            # log any genuine errors that get swallowed here (i.e. not the
            # exception being re-raised and not cancellations from above).
            rails_exc, gen_exc = await asyncio.gather(rails_task, gen_task, return_exceptions=True)
            for name, exc in (("input_rails", rails_exc), ("generation", gen_exc)):
                if (
                    isinstance(exc, BaseException)
                    and not isinstance(exc, asyncio.CancelledError)
                    and exc is not outer_exc
                ):
                    log.warning(
                        "[%s] %s task error discarded during cleanup: %r",
                        req_id,
                        name,
                        exc,
                    )
            raise

        return response

    async def _parallel_input_rail_and_response_generation(
        self,
        rails_task: asyncio.Task,
        gen_task: asyncio.Task,
        req_id: str,
        request_span: Optional["Span"] = None,
    ) -> Optional[LLMResponse]:
        """Race input rails against LLM generation, return LLMResponse or None (rejected)."""
        done, _ = await asyncio.wait({rails_task, gen_task}, return_when=asyncio.FIRST_COMPLETED)

        first_completed = (
            GuardrailsAttributes.SPECULATIVE_FIRST_COMPLETED_INPUT_RAILS
            if rails_task in done
            else GuardrailsAttributes.SPECULATIVE_FIRST_COMPLETED_GENERATION
        )

        if rails_task in done:
            input_result = rails_task.result()

            if not input_result.is_safe:
                log.info("[%s] Input blocked (speculative): %s", req_id, input_result.reason)
                gen_task.cancel()
                # Use gather(return_exceptions=True) instead of bare await: when both
                # tasks finish simultaneously, gen_task may hold a stored exception that
                # would leak through suppress(CancelledError). gather drains it safely.
                gen_result = (await asyncio.gather(gen_task, return_exceptions=True))[0]
                if isinstance(gen_result, BaseException) and not isinstance(gen_result, asyncio.CancelledError):
                    log.warning("[%s] LLM generation error suppressed: %s", req_id, gen_result)
                if self._metrics_enabled:
                    record_request_blocked(RailDirection.INPUT)
                set_speculative_span_attrs(
                    request_span, first_completed, GuardrailsAttributes.SPECULATIVE_FIRST_COMPLETED_INPUT_RAILS
                )
                return None

            # Rails passed — wait for generation to finish
            response = await gen_task
            set_speculative_span_attrs(request_span, first_completed, "none")
        else:
            # Generation finished first — wait for rails verdict
            response = gen_task.result()

            input_result = await rails_task

            if not input_result.is_safe:
                log.info("[%s] Input blocked (speculative, gen-first): %s", req_id, input_result.reason)
                if self._metrics_enabled:
                    record_request_blocked(RailDirection.INPUT)
                set_speculative_span_attrs(
                    request_span, first_completed, GuardrailsAttributes.SPECULATIVE_FIRST_COMPLETED_INPUT_RAILS
                )
                return None

            set_speculative_span_attrs(request_span, first_completed, "none")

        log.debug("[%s] Main LLM response: %s", req_id, truncate(response.content))
        return response

    def _validate_streaming_with_output_rails(self) -> None:
        """Raise if output rails exist but streaming is not enabled for them."""
        if len(self.config.rails.output.flows) > 0 and not self._has_streaming_output_rails:
            raise StreamingNotSupportedError(
                "stream_async() cannot be used when output rails are configured but "
                "rails.output.streaming.enabled is False. Either set "
                "rails.output.streaming.enabled to True in your configuration, or use "
                "generate_async() instead of stream_async()."
            )

    def stream_async(
        self,
        messages: LLMMessages,
        options: Optional[Union[dict, GenerationOptions]] = None,
        include_metadata: Optional[bool] = False,
    ) -> AsyncIterator[Union[str, dict]]:
        """Stream LLM response tokens with input/output rails applied.

        Returns an async iterator that yields string chunks (or dicts when
        ``include_metadata=True``).  Input rails run before any tokens are
        streamed.  If output rails are configured and streaming is enabled,
        tokens are buffered and checked using the same ``RollingBuffer`` /
        ``stream_first`` semantics as LLMRails.

        Args:
            messages: Conversation messages in OpenAI format.
            options: Optional GenerationOptions (llm_params are forwarded to
                the main LLM call).
            include_metadata: When True, chunks are dicts with ``text`` and
                ``metadata`` keys instead of plain strings.

        Returns:
            An async iterator of string chunks (or dicts).

        Raises:
            StreamingNotSupportedError: If output rails are present but
                ``rails.output.streaming.enabled`` is False.
            ValueError: If ``include_metadata=True`` with output rails
                streaming enabled (BufferStrategy requires plain string chunks).
            asyncio.QueueFull: If the streaming concurrency limit is
                reached (load shedding).
        """
        self._validate_streaming_with_output_rails()

        # Speculative streaming (SG2): input rails race the LLM instead of blocking
        # before it.  Only check-first is supported — during the speculation window
        # tokens cannot reach the client, so stream_first is overridden to
        # check-first for speculative requests.
        #
        # NOTE: this precondition warning fires per stream_async() call (i.e. at
        # request time), not once at engine startup.  Flagged here so developers
        # know the check-first override is decided per request, not globally.
        use_speculative = self._speculative_generation
        force_check_first = False
        if use_speculative and self._has_streaming_output_rails and self.config.rails.output.streaming.stream_first:
            warnings.warn(
                "speculative_generation with stream_first=True is not supported for streaming; "
                "forcing check-first behavior for this request",
                stacklevel=2,
            )
            force_check_first = True

        if include_metadata and self._has_streaming_output_rails:
            raise ValueError(
                "include_metadata=True is not supported when output rails streaming is enabled. "
                "BufferStrategy requires plain string chunks. Use include_metadata=False or "
                "disable output rails streaming."
            )

        # Normalize options once; the inner tasks below read both llm_params
        # (passed unchanged to the LLM call, including tool definitions) and the
        # per-request tool-rail toggles off the coerced GenerationOptions.
        options = _coerce_generation_options(options)
        llm_kwargs: dict = options.llm_params if (options and options.llm_params) else {}
        input_enabled = options.rails.input if options else True
        output_enabled = options.rails.output if options else True
        tool_input_enabled = options.rails.tool_input if options else True
        tool_output_enabled = options.rails.tool_output if options else True

        streaming_handler = StreamingHandler(include_metadata=include_metadata)
        # Tool calls assembled by the stream: _generation_task rebinds this (via
        # nonlocal) to the engine's finalized list and _wrapped_iterator reads it
        # after the content stream drains. The engine emits the complete list once
        # (see ModelEngine.stream_call), so a plain rebind is sufficient.
        accumulated_tool_calls: list[ToolCall] = []

        async def _generation_task(request_span, *, run_input_rails: bool = True, spec_stats: Optional[dict] = None):
            """Background task: input rails → stream LLM chunks → push to handler.

            ``request_span`` is the IORails request span (or ``None`` when
            tracing is disabled), captured by the caller from
            ``traced_request`` and passed in explicitly — never fetched via
            ``trace.get_current_span()`` which could return the host app's
            ambient span and pollute unrelated traces.

            When ``run_input_rails`` is False (speculative streaming), the
            tool-result and input rails are skipped here — the caller runs them
            in a concurrent task so LLM tokens start flowing immediately.  When
            ``spec_stats`` is provided, the task records its wall-clock duration
            into it (``generation_duration_ms``) for speculative telemetry.

            Inherits the request ID from the caller context via create_task().
            """
            nonlocal accumulated_tool_calls
            req_id = get_request_id()
            t0 = time.monotonic()
            try:
                if run_input_rails:
                    # Step 0: Tool-result rails. Client/agent harness executes tool calls and sends
                    # results of execution to Main LLM along with prior conversation history
                    # Symmetric with INPUT rails for dialog use-case
                    log.info("[%s] Running tool result rails", req_id)
                    tool_result = await self.rails_manager.are_tool_results_safe(messages, enabled=tool_input_enabled)
                    if not tool_result.is_safe:
                        log.info("[%s] Tool result blocked: %s", req_id, tool_result.reason)
                        if self._metrics_enabled:
                            record_request_blocked(RailDirection.INPUT)
                        await streaming_handler.push_chunk(
                            self._guardrails_violation_payload(
                                f"Blocked by tool input rails: {tool_result.reason}", "tool_input_rails"
                            )
                        )
                        await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore[arg-type]
                        return

                    # Step 1: Input rails (non-streaming)
                    log.info("[%s] Running input rails", req_id)
                    input_result = await self.rails_manager.is_input_safe(messages, enabled=input_enabled)
                    if not input_result.is_safe:
                        log.info("[%s] Input blocked: %s", req_id, input_result.reason)
                        if self._metrics_enabled:
                            record_request_blocked(RailDirection.INPUT)
                        await streaming_handler.push_chunk(
                            self._guardrails_violation_payload(
                                f"Blocked by input rails: {input_result.reason}", "input_rails"
                            )
                        )
                        await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore[arg-type]
                        return

                # Step 2: Stream main LLM content from structured response.
                # delta_content is forwarded as text chunks; delta_tool_calls are
                # accumulated and surfaced as a terminal JSON chunk after the text
                # stream ends. Reasoning is dropped for LLMRails compatibility.
                log.info("[%s] Streaming main LLM", req_id)
                content_parts: list[str] = []
                async for chunk in self.engine_registry.stream_model_call("main", messages, **llm_kwargs):
                    if chunk.delta_content:
                        content_parts.append(chunk.delta_content)
                        await streaming_handler.push_chunk(chunk.delta_content)
                    if chunk.delta_tool_calls:
                        # Engine emits the complete finalized list once (see
                        # ModelEngine.stream_call), so rebind rather than accumulate.
                        accumulated_tool_calls = chunk.delta_tool_calls

                # While LLMResponseChunk.delta_reasoning is dropped explicitly,
                # think-tags embedded in delta_content are not. Give a warning
                # to reflect this asymmetry (once-per-request).
                full_content = "".join(content_parts)
                if "<think>" in full_content or "</think>" in full_content:
                    log.warning(
                        "[%s] Streamed content contains <think> tags; model is leaking "
                        "reasoning via delta_content rather than delta_reasoning "
                        "(output rails will process reasoning tokens)",
                        req_id,
                    )

                if accumulated_tool_calls and not content_parts:
                    log.info("[%s] Tool-call-only stream: output rails skipped", req_id)

                await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore[arg-type]
            except Exception as e:
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.error(
                    "[%s] generation task failed time=%.1fms",
                    req_id,
                    elapsed_ms,
                    exc_info=True,
                )
                # Mark the request span ERROR; record_span_error no-ops when
                # request_span is None (tracing disabled), so no extra guard
                # is needed and there's no ambient-context lookup to worry about.
                record_span_error(request_span, e)
                # Bump guardrails.requests.errors explicitly: the exception is
                # about to be swallowed (converted to an error-payload chunk),
                # so request_metrics's except branch never fires for the
                # streaming path.
                if self._metrics_enabled:
                    record_request_error(e)
                error_payload = json.dumps(
                    {"error": {"message": str(e), "type": _GENERATION_ERROR_TYPE, "code": "generation_failed"}}
                )
                await streaming_handler.push_chunk(error_payload)
                await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore[arg-type]
            finally:
                elapsed_ms = (time.monotonic() - t0) * 1000
                if spec_stats is not None:
                    spec_stats["generation_duration_ms"] = elapsed_ms
                log.info("[%s] generation task completed time=%.1fms", req_id, elapsed_ms)

        async def _wrapped_iterator():
            """Wrap the base iterator with semaphore-based concurrency control.

            Request-level metrics (``guardrails.requests``,
            ``guardrails.request.duration``, ``guardrails.requests.errors``)
            wrap the entire stream lifecycle, so a ``QueueFull`` on the
            semaphore check bumps BOTH ``stream.rejections`` and
            ``requests.errors{error.type=QueueFull}`` — dual-signal
            semantics matching the non-streaming path.
            """
            # Ensure engines are running (idempotent if already started).
            # Kept outside ``request_metrics`` so duration matches the
            # non-streaming path (excludes one-time engine startup cost).
            await self.start()

            metrics_ctx = request_metrics() if self._metrics_enabled else nullcontext()
            with metrics_ctx:
                # Non-blocking acquire; raises immediately if all slots are taken.
                # locked() returns True when the semaphore value is 0.  Because there
                # is no await between the check and acquire(), no other coroutine can
                # interleave in asyncio's cooperative model, so this is race-free.
                if self._stream_semaphore.locked():
                    if self._metrics_enabled:
                        record_stream_rejected()
                    raise asyncio.QueueFull("Streaming concurrency limit reached")
                await self._stream_semaphore.acquire()

                tracer = self._tracer if self._tracing_enabled else None
                # Track this stream as active while it holds the semaphore
                # permit; the CM decrements in its finally, just before the
                # outer ``semaphore.release()`` below.
                stream_active_ctx = stream_active_metric() if self._metrics_enabled else nullcontext()
                try:
                    with stream_active_ctx:
                        # traced_request is entered inside the async generator so the
                        # request span is the current OTEL context when create_task()
                        # below snapshots contextvars — that's what makes rail / LLM
                        # spans raised inside _generation_task attach as children.
                        with traced_request(tracer) as (request_span, req_id):
                            t0 = time.monotonic()
                            # Accumulate chunks the consumer actually receives.
                            # Declared outside the try so the outer finally can
                            # always reference it, even if the try body raises
                            # before any chunk is yielded.  Captured at stream end
                            # on the request span so we record exactly what reached
                            # the caller (including any output-rails error JSON
                            # injected on block).
                            delivered: list[str] = []
                            # Set if an error / guardrails-violation payload reaches
                            # the consumer, so the terminal tool-call chunk is
                            # suppressed (never surface tool calls after a failure/block).
                            error_emitted = False
                            # Speculative-streaming state, declared before the try so
                            # the outer finally can always reference them (defined even
                            # if the try body raises before assignment).
                            spec_stats: Optional[dict[str, Any]] = None
                            input_task: Optional[asyncio.Task] = None
                            try:
                                log.info("[%s] stream_async called", req_id)
                                log.debug("[%s] stream_async messages=%s", req_id, truncate(messages))

                                # Speculative streaming (SG2): run input rails in a
                                # concurrent task so the LLM starts streaming right
                                # away, and skip input rails inside the generation
                                # task.  spec_stats collects telemetry filled by the
                                # input task, the generation task, and the gate.
                                if use_speculative:
                                    spec_stats = {
                                        "first_completed": None,
                                        "first_rejector": GuardrailsAttributes.SPECULATIVE_CANCELLATION_NONE,
                                        "safe": True,
                                        "output_rails_early_reject": False,
                                        "cancellation_event": GuardrailsAttributes.SPECULATIVE_CANCELLATION_NONE,
                                    }
                                    input_task = asyncio.create_task(
                                        self._check_speculative_input_safety(
                                            messages,
                                            input_enabled=input_enabled,
                                            tool_input_enabled=tool_input_enabled,
                                            spec_stats=spec_stats,
                                        )
                                    )
                                    task = asyncio.create_task(
                                        _generation_task(request_span, run_input_rails=False, spec_stats=spec_stats)
                                    )
                                else:
                                    task = asyncio.create_task(_generation_task(request_span))
                                try:
                                    # Determine the inner iterator: with or without output rails.
                                    if self._has_streaming_output_rails:
                                        inner_iterator = self._run_output_rails_in_streaming(
                                            streaming_handler=streaming_handler,
                                            messages=messages,
                                            enabled=output_enabled,
                                            include_metadata=include_metadata,
                                            force_check_first=force_check_first,
                                        )
                                    else:
                                        # SG2 buffer-and-release: with no output rails configured,
                                        # speculation still runs.  Raw LLM tokens flow straight from
                                        # the streaming handler; when speculating, the gate below holds
                                        # them in the bounded release buffer until input rails pass,
                                        # then flushes.
                                        inner_iterator = streaming_handler

                                    # Gate raw/validated chunks on the input rails verdict during the
                                    # speculation window; pass through unchanged when not speculating.
                                    if use_speculative:
                                        assert input_task is not None and spec_stats is not None
                                        base_iterator = self._gate_on_input(
                                            inner_iterator, input_task, spec_stats, include_metadata=include_metadata
                                        )
                                    else:
                                        base_iterator = inner_iterator

                                    async for chunk in base_iterator:
                                        if chunk is not None:
                                            if _is_stream_error_chunk(chunk):
                                                error_emitted = True
                                            if self._content_capture_enabled:
                                                # Plain strings are the normal path.
                                                # Dicts arrive when include_metadata=True;
                                                # skip empty-string text fields so
                                                # metadata-only frames don't pollute
                                                # the captured output.
                                                if isinstance(chunk, str):
                                                    delivered.append(chunk)
                                                elif isinstance(chunk, dict):
                                                    text = chunk.get("text")
                                                    if isinstance(text, str) and text:
                                                        delivered.append(text)
                                            yield chunk
                                    # Emit assembled tool calls as the terminal chunk once
                                    # text + output rails finish, but only on a clean stream:
                                    # suppress after an error/guardrails block so the caller
                                    # never receives a tool call following a failure.
                                    if accumulated_tool_calls and not error_emitted:
                                        # ToolCallRail checks tool calls from the main LLM (OUTPUT)
                                        tool_call = await self.rails_manager.are_tool_calls_safe(
                                            accumulated_tool_calls, llm_kwargs, enabled=tool_output_enabled
                                        )
                                        if not tool_call.is_safe:
                                            log.info(
                                                "[%s] Streamed tool call blocked: %s",
                                                req_id,
                                                tool_call.reason,
                                            )
                                            if self._metrics_enabled:
                                                record_request_blocked(RailDirection.OUTPUT)
                                            violation = self._guardrails_violation_payload(
                                                f"Blocked by tool output rails: {tool_call.reason}",
                                                "tool_output_rails",
                                            )
                                            if self._content_capture_enabled:
                                                delivered.append(violation)
                                            yield _frame_for_stream(violation, include_metadata)
                                        else:
                                            payload, framed = _terminal_tool_call_chunk(
                                                accumulated_tool_calls, include_metadata
                                            )
                                            if self._content_capture_enabled:
                                                delivered.append(payload)
                                            yield framed
                                finally:
                                    if not task.done():
                                        task.cancel()
                                    with suppress(asyncio.CancelledError):
                                        await task
                                    # Defensive: the gate cancels+drains input_task in
                                    # its own finally, but ensure it never leaks if the
                                    # gate was never fully iterated (early break/error).
                                    if input_task is not None:
                                        if not input_task.done():
                                            input_task.cancel()
                                        with suppress(asyncio.CancelledError, Exception):
                                            await input_task
                            except Exception:
                                elapsed_ms = (time.monotonic() - t0) * 1000
                                log.error("[%s] stream_async failed time=%.1fms", req_id, elapsed_ms, exc_info=True)
                                raise
                            finally:
                                elapsed_ms = (time.monotonic() - t0) * 1000
                                log.info("[%s] stream_async completed time=%.1fms", req_id, elapsed_ms)
                                # Capture input + accumulated output onto the request
                                # span before it closes.  Always runs (normal exit,
                                # error, or consumer cancellation) so an errored
                                # stream still records whatever reached the caller.
                                # Empty `delivered` -> output_text=None so we don't
                                # falsely claim an "" assistant message was produced.
                                if self._content_capture_enabled:
                                    output_text = "".join(delivered) if delivered else None
                                    set_request_content(request_span, messages, output_text)
                                # Stamp speculative-generation telemetry on the request
                                # span.  Runs after teardown so both task durations are
                                # recorded (the generation task's finally has run once it
                                # was awaited above).  overlap ≈ min(both durations) since
                                # both tasks start together; time_saved is the overlap for
                                # safe requests and 0 for rejected ones.
                                if use_speculative and spec_stats is not None:
                                    rails_ms = spec_stats.get("rails_duration_ms")
                                    gen_ms = spec_stats.get("generation_duration_ms")
                                    overlap_ms = (
                                        min(rails_ms, gen_ms) if rails_ms is not None and gen_ms is not None else None
                                    )
                                    time_saved_ms = (
                                        None if overlap_ms is None else (overlap_ms if spec_stats.get("safe") else 0.0)
                                    )
                                    set_speculative_span_attrs(
                                        request_span,
                                        spec_stats.get("first_completed")
                                        or GuardrailsAttributes.SPECULATIVE_FIRST_COMPLETED_GENERATION,
                                        spec_stats.get(
                                            "first_rejector", GuardrailsAttributes.SPECULATIVE_CANCELLATION_NONE
                                        ),
                                        rails_duration_ms=rails_ms,
                                        generation_duration_ms=gen_ms,
                                        overlap_ms=overlap_ms,
                                        time_saved_ms=time_saved_ms,
                                        cancellation_event=spec_stats.get("cancellation_event"),
                                        output_rails_early_reject=spec_stats.get("output_rails_early_reject"),
                                        output_rails_speculation_chunks=spec_stats.get(
                                            "output_rails_speculation_chunks"
                                        ),
                                        output_rails_wasted_chunks=spec_stats.get("output_rails_wasted_chunks"),
                                        release_queue_duration_ms=spec_stats.get("release_queue_duration_ms"),
                                        release_queue_token_count=spec_stats.get("release_queue_token_count"),
                                    )
                finally:
                    self._stream_semaphore.release()

        return _wrapped_iterator()

    async def _check_speculative_input_safety(
        self,
        messages: LLMMessages,
        *,
        input_enabled: Union[bool, list[str]] = True,
        tool_input_enabled: Union[bool, list[str]] = True,
        spec_stats: Optional[dict] = None,
    ):
        """Concurrent input-safety check for speculative streaming (SG2).

        Runs tool-result rails then input rails, returning ``(RailResult, param)``
        for the first failing rail (or the safe input result).  ``param``
        identifies the rail family for the client-facing violation payload
        (``tool_input_rails`` vs ``input_rails``), matching the non-speculative
        path.  Runs as its own task so LLM generation can stream concurrently
        during the speculation window.  Records its wall-clock duration into
        ``spec_stats['rails_duration_ms']`` for telemetry.
        """
        t0 = time.monotonic()
        try:
            tool_result = await self.rails_manager.are_tool_results_safe(messages, enabled=tool_input_enabled)
            if not tool_result.is_safe:
                return tool_result, "tool_input_rails"
            input_result = await self.rails_manager.is_input_safe(messages, enabled=input_enabled)
            return input_result, "input_rails"
        finally:
            if spec_stats is not None:
                spec_stats["rails_duration_ms"] = (time.monotonic() - t0) * 1000

    async def _gate_on_input(
        self,
        base_iterator: AsyncIterator[Union[str, dict]],
        input_task: "asyncio.Task",
        spec_stats: dict,
        *,
        include_metadata: Optional[bool] = False,
    ) -> AsyncGenerator[Union[str, dict], None]:
        """Hold streamed chunks until the input rails verdict is known (SG2).

        During the speculation window the LLM (and output rails, when
        configured) run concurrently with input rails.  Chunks that arrive
        before the input verdict are held in a bounded in-memory buffer rather
        than sent to the client — the input safety verdict is not yet known, so
        nothing may reach the caller.  On input PASS the held buffer is flushed
        and streaming continues normally.  On input REJECT (or an output-rails
        early reject / generation error surfaced as an error chunk) the held
        buffer is discarded and a refusal / the error payload is emitted.

        Cancellation (SG2): on input reject the generation task is torn down by
        the caller's ``finally``; on an output-rails early reject the still-
        running input rails task is cancelled here so the request aborts before
        the input verdict arrives.

        When the held buffer reaches ``speculative_max_buffered_tokens`` the gate
        stops consuming the base iterator and blocks on the input verdict, forcing
        an early resolve (release on pass, teardown on reject).  This bounds only
        the release buffer — the background generation task keeps producing into
        the stream queue — so total memory is bounded by the input-rail latency and
        the model's finite output, not by halting generation.  (An alternative
        overflow policy, aborting the request when the bound is exceeded, would cap
        total memory by cancelling the producer; backpressure is used here instead.)
        """
        req_id = get_request_id()
        input_rails = GuardrailsAttributes.SPECULATIVE_FIRST_COMPLETED_INPUT_RAILS
        generation = GuardrailsAttributes.SPECULATIVE_FIRST_COMPLETED_GENERATION
        output_rails = GuardrailsAttributes.SPECULATIVE_FIRST_COMPLETED_OUTPUT_RAILS
        none_value = GuardrailsAttributes.SPECULATIVE_CANCELLATION_NONE

        released = False
        held: list = []
        hold_start: Optional[float] = None
        spec_chunks = 0

        # Human-readable labels for the violation message, keyed by the payload
        # ``param``.  Keeps tool-result-rail rejections labeled as tool_input_rails
        # (matching the non-speculative path) instead of collapsing to input_rails.
        reject_labels = {"input_rails": "input rails", "tool_input_rails": "tool input rails"}

        def _mark_reject_input(reason, first_completed, cancellation_event, param):
            spec_stats["first_completed"] = first_completed
            spec_stats["first_rejector"] = input_rails
            spec_stats["cancellation_event"] = cancellation_event
            spec_stats["output_rails_wasted_chunks"] = len(held)
            spec_stats["safe"] = False
            label = reject_labels.get(param, "input rails")
            return _frame_for_stream(
                self._guardrails_violation_payload(f"Blocked by {label}: {reason}", param),
                include_metadata,
            )

        def _mark_release(first_completed):
            spec_stats["first_completed"] = first_completed
            spec_stats["safe"] = True
            spec_stats["release_queue_token_count"] = len(held)
            if hold_start is not None:
                spec_stats["release_queue_duration_ms"] = (time.monotonic() - hold_start) * 1000

        try:
            async for chunk in base_iterator:
                if released:
                    yield chunk
                    continue

                # An error chunk during the speculation window is either an
                # output-rails violation or a generation error surfaced by the
                # base iterator.  Either way we short-circuit: cancel the still-
                # running input rails task (output-reject early short-circuit) and
                # forward the payload.  Held (validated-but-unreleased) chunks are
                # discarded — the request is being refused.
                if _is_stream_error_chunk(chunk):
                    text = chunk.get("text") if isinstance(chunk, dict) else chunk
                    is_output_violation = False
                    if isinstance(text, str):
                        try:
                            parsed = json.loads(text)
                            is_output_violation = (
                                isinstance(parsed, dict) and parsed.get("error", {}).get("param") == "output_rails"
                            )
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if not input_task.done():
                        input_task.cancel()
                    if is_output_violation:
                        spec_stats["first_completed"] = output_rails
                        spec_stats["first_rejector"] = output_rails
                        spec_stats["output_rails_early_reject"] = True
                    else:
                        # Generation error surfaced as a chunk — not a rail rejection.
                        spec_stats["first_completed"] = generation
                        spec_stats["first_rejector"] = none_value
                    spec_stats["cancellation_event"] = GuardrailsAttributes.SPECULATIVE_CANCELLATION_INPUT_RAILS
                    spec_stats["output_rails_wasted_chunks"] = len(held)
                    spec_stats["safe"] = False
                    held.clear()
                    log.info("[%s] Speculative stream short-circuit (%s)", req_id, spec_stats["first_rejector"])
                    yield chunk
                    return

                spec_chunks += 1
                if hold_start is None:
                    hold_start = time.monotonic()
                held.append(chunk)

                if input_task.done():
                    input_result, input_param = input_task.result()
                    first_completed = input_rails
                elif len(held) >= self._speculative_max_buffered_tokens:
                    # Release buffer full — stop consuming the base iterator and block
                    # for the verdict (release on pass, teardown on reject).  This bounds
                    # `held`, not the upstream stream queue: the generation task keeps
                    # producing until the verdict resolves.  (To cap total memory instead,
                    # switch this to abort the request on overflow — cancelling the
                    # producer stops queue growth at the bound.)
                    log.info("[%s] Speculative release buffer full (%d); awaiting input verdict", req_id, len(held))
                    input_result, input_param = await input_task
                    first_completed = generation
                else:
                    # Still speculating — keep holding.
                    continue

                if not input_result.is_safe:
                    log.info("[%s] Input blocked (speculative streaming): %s", req_id, input_result.reason)
                    if self._metrics_enabled:
                        record_request_blocked(RailDirection.INPUT)
                    refusal = _mark_reject_input(
                        input_result.reason,
                        first_completed,
                        GuardrailsAttributes.SPECULATIVE_CANCELLATION_GENERATION,
                        input_param,
                    )
                    held.clear()
                    yield refusal
                    return

                _mark_release(first_completed)
                released = True
                for held_chunk in held:
                    yield held_chunk
                held.clear()

            # Stream ended before the input verdict was applied (generation
            # finished first, or an empty stream).  Await the verdict and either
            # flush the held buffer or refuse.
            if not released:
                input_result, input_param = await input_task
                if not input_result.is_safe:
                    log.info("[%s] Input blocked (speculative streaming, gen-first): %s", req_id, input_result.reason)
                    if self._metrics_enabled:
                        record_request_blocked(RailDirection.INPUT)
                    # Generation already completed, so nothing is cancelled here.
                    refusal = _mark_reject_input(input_result.reason, generation, none_value, input_param)
                    held.clear()
                    yield refusal
                    return
                _mark_release(generation)
                for held_chunk in held:
                    yield held_chunk
                held.clear()
        finally:
            spec_stats["output_rails_speculation_chunks"] = spec_chunks
            if not input_task.done():
                input_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await input_task

    async def _run_output_rails_in_streaming(
        self,
        streaming_handler: AsyncIterator[Union[str, dict]],
        messages: LLMMessages,
        *,
        enabled: Union[bool, list[str]] = True,
        include_metadata: Optional[bool] = False,
        force_check_first: bool = False,
    ) -> AsyncGenerator[Union[str, dict], None]:
        """Buffer streamed chunks and run output rails on each batch.

        Uses the same ``RollingBuffer`` and ``stream_first`` semantics as
        LLMRails:
        - ``stream_first=True``: yield chunks immediately, then run output
          rails.  If unsafe, inject an error and stop.
        - ``stream_first=False``: run output rails first, only yield chunks
          if safe.

        ``force_check_first`` overrides the configured ``stream_first`` to
        check-first behavior.  Speculative streaming (SG2) sets this so that
        validated chunks are produced (never pre-yielded) — during the
        speculation window tokens cannot reach the client before the input
        rails verdict is known.
        """

        # Unpack streaming config and get the buffer strategy
        output_streaming_config = self.config.rails.output.streaming
        stream_first = output_streaming_config.stream_first and not force_check_first
        buffer_strategy = get_buffer_strategy(output_streaming_config)

        async for chunk_batch in buffer_strategy(streaming_handler):
            user_output_chunks = chunk_batch.user_output_chunks
            bot_response_chunk = buffer_strategy.format_chunks(chunk_batch.processing_context)

            # If the batch contains a generation error from _generation_task,
            # yield it directly and stop — don't feed error JSON through output rails.
            for chunk in user_output_chunks:
                try:
                    parsed = json.loads(chunk)
                    if isinstance(parsed, dict) and parsed.get("error", {}).get("type") == _GENERATION_ERROR_TYPE:
                        yield chunk
                        return
                except (json.JSONDecodeError, TypeError):
                    pass

            if stream_first:
                for chunk in user_output_chunks:
                    yield chunk

            # Run output rails on the accumulated context. Skip when content is empty
            # (e.g. tool-call-only response) to avoid a pointless is_output_safe("") call.
            req_id = get_request_id()
            if not bot_response_chunk:
                if not stream_first:
                    for chunk in user_output_chunks:
                        yield chunk
                continue

            log.info("[%s] Running output rails", req_id)
            output_result = await self.rails_manager.is_output_safe(messages, bot_response_chunk, enabled=enabled)
            if not output_result.is_safe:
                log.info("[%s] Output blocked: %s", req_id, output_result.reason)
                if self._metrics_enabled:
                    record_request_blocked(RailDirection.OUTPUT)
                violation = self._guardrails_violation_payload(
                    f"Blocked by output rails: {output_result.reason}", "output_rails"
                )
                yield _frame_for_stream(violation, include_metadata)
                return

            if not stream_first:
                for chunk in user_output_chunks:
                    yield chunk

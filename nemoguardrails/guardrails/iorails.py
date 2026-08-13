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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from nemoguardrails.actions.rail_outcome import TransformTarget
from nemoguardrails.base_guardrails import BaseGuardrails
from nemoguardrails.exceptions import StreamingNotSupportedError
from nemoguardrails.guardrails.async_work_queue import AsyncWorkQueue
from nemoguardrails.guardrails.compiled_rail import (
    RailCompilationError,
    RailDependencies,
    compile_rail,
    unservable_reason,
)
from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.guardrails_types import (
    LLMMessage,
    LLMMessages,
    RailCallRecord,
    RailDirection,
    RailResult,
    TimedLLMResponse,
    client_reason,
    display_reason,
    get_request_id,
    rewrite_user_message,
    serialize_prompt,
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
from nemoguardrails.llm.call import _extract_and_remove_think_tags
from nemoguardrails.llm.clients._errors import (
    STREAM_ERROR_TYPES,
    build_streaming_error_payload,
)
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.manifests import RailDirection as SurfaceDirection
from nemoguardrails.patch_asyncio import check_sync_call_from_async_loop
from nemoguardrails.rails.llm.buffer import get_buffer_strategy
from nemoguardrails.rails.llm.config import RailsConfig, _get_flow_name
from nemoguardrails.rails.llm.options import (
    ActivatedRail,
    ExecutedAction,
    GenerationLog,
    GenerationOptions,
    GenerationResponse,
    GenerationStats,
    RailsResult,
    RailStatus,
    RailType,
)
from nemoguardrails.streaming import END_OF_STREAM, StreamingHandler
from nemoguardrails.tracing.constants import GuardrailsAttributes
from nemoguardrails.types import LLMModel, LLMResponse, LLMResponseChunk, ToolCall

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


def _is_stream_error_chunk(chunk: Union[str, dict]) -> bool:
    """True when a streamed chunk is an error/violation payload.

    Covers the ``generation_error`` / ``downstream_error`` payloads pushed on a
    generation failure and the ``guardrails_violation`` payload emitted when
    rails block. Handles plain-string chunks and the ``{"text": ...}`` frames
    produced when ``include_metadata=True``. The cheap ``"error"`` substring
    guard keeps the per-chunk hot path from JSON-parsing ordinary text tokens.

    The ``type`` must be one of the internal markers in ``STREAM_ERROR_TYPES``:
    matching any object that merely has an ``error`` key would let model output
    that looks like an OpenAI error truncate the stream and skip output rails.
    """
    text = chunk.get("text") if isinstance(chunk, dict) else chunk
    if not isinstance(text, str) or '"error"' not in text:
        return False
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(parsed, dict):
        return False
    error_obj = parsed.get("error")
    return isinstance(error_obj, dict) and error_obj.get("type") in STREAM_ERROR_TYPES


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


def _stream_chunk_metadata(chunk: LLMResponseChunk) -> Optional[dict]:
    """Extract per-chunk streaming metadata (usage + provider_metadata) for the StreamingHandler.

    Mirrors LLMRails' streaming metadata contract so ``include_metadata`` consumers see the same
    frames from both engines: token usage as a flat ``input/output/total_tokens`` dict and
    ``provider_metadata`` verbatim. Returns ``None`` when the chunk carries neither, so ordinary
    text tokens push no metadata and the StreamingHandler emits them as bare ``{"text": ...}``.
    """
    metadata: dict = {}
    if chunk.provider_metadata:
        metadata["provider_metadata"] = chunk.provider_metadata
    if chunk.usage:
        metadata["usage"] = {
            "input_tokens": chunk.usage.input_tokens,
            "output_tokens": chunk.usage.output_tokens,
            "total_tokens": chunk.usage.total_tokens,
        }
    return metadata or None


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


def _build_llm_metadata(response: LLMResponse) -> Optional[dict]:
    """Return the main call's ``provider_metadata`` verbatim (LLMRails mirror).

    A pure passthrough — token usage lives in ``log`` (``log.llm_calls`` / ``log.stats``),
    not here, matching LLMRails. Returns ``None`` when the main call carried no metadata.
    """
    return dict(response.provider_metadata or {}) or None


def _make_generation_record(
    timed: TimedLLMResponse,
    provider_name: Optional[str],
    prompt: Optional[str],
) -> RailCallRecord:
    """Represent the main generation call as a ``RailCallRecord`` (rail_type "generation").

    Unifies the main call with the rail calls so the log builder can treat every LLM call
    the same way. ``prompt`` is the serialized main-model messages; ``completion`` is the
    response content. Timing comes from ``timed``: ``started_at``/``finished_at`` are
    wall-clock timestamps and ``duration`` is a monotonic delta.
    """
    response = timed.response
    return RailCallRecord(
        flow="generation",
        rail_type="generation",
        is_safe=True,
        made_call=True,
        action_name="generate_bot_message",
        task="general",
        request_id=response.request_id,
        usage=response.usage,
        llm_model_name=response.model,
        llm_provider_name=provider_name,
        prompt=prompt,
        completion=response.content,
        started_at=timed.started_at,
        finished_at=timed.finished_at,
        duration=timed.duration,
    )


def _record_has_llm_call(record: RailCallRecord) -> bool:
    """True when a record reflects a model/API call (made a call, or carries usage/model)."""
    return record.made_call or record.usage is not None or record.llm_model_name is not None


def _call_info(record: RailCallRecord) -> LLMCallInfo:
    """Map a ``RailCallRecord`` to a ``LLMCallInfo``"""
    usage = record.usage
    return LLMCallInfo(
        task=record.task,
        duration=record.duration,
        total_tokens=usage.total_tokens if usage else None,
        prompt_tokens=usage.input_tokens if usage else None,
        completion_tokens=usage.output_tokens if usage else None,
        started_at=record.started_at,
        finished_at=record.finished_at,
        request_id=record.request_id,
        prompt=record.prompt,
        completion=record.completion,
        llm_model_name=record.llm_model_name or "unknown",
        llm_provider_name=record.llm_provider_name or "unknown",
    )


def _activated_rail(record: RailCallRecord) -> ActivatedRail:
    """Map a ``RailCallRecord`` to an ``ActivatedRail`` with one synthetic ``ExecutedAction``.

    IORails runs one model-backed check per rail (not a Colang action chain), so each rail
    gets a single ``ExecutedAction`` carrying the rail's structured verdict as
    ``return_value`` and its (at most one) LLM call.
    """
    action = ExecutedAction(
        action_name=record.action_name or record.flow,
        action_params={},
        return_value=record.return_value,
        llm_calls=[_call_info(record)] if _record_has_llm_call(record) else [],
        started_at=record.started_at,
        finished_at=record.finished_at,
        duration=record.duration,
    )
    return ActivatedRail(
        type=record.rail_type,
        name=record.flow,
        executed_actions=[action],
        stop=not record.is_safe,
        started_at=record.started_at,
        finished_at=record.finished_at,
        duration=record.duration,
    )


def _build_generation_stats(
    records: list[RailCallRecord], call_records: list[RailCallRecord], total_duration: Optional[float]
) -> GenerationStats:
    """Aggregate per-phase durations and token totals across every recorded call."""

    def _phase_duration(*rail_types: str) -> Optional[float]:
        total = sum((r.duration or 0.0) for r in records if r.rail_type in rail_types)
        return total or None

    return GenerationStats(
        input_rails_duration=_phase_duration("input", "tool_input"),
        output_rails_duration=_phase_duration("output", "tool_output"),
        generation_rails_duration=_phase_duration("generation"),
        total_duration=total_duration,
        llm_calls_duration=sum((r.duration or 0.0) for r in call_records) or None,
        llm_calls_count=len(call_records),
        llm_calls_total_prompt_tokens=sum(r.usage.input_tokens for r in call_records if r.usage),
        llm_calls_total_completion_tokens=sum(r.usage.output_tokens for r in call_records if r.usage),
        llm_calls_total_tokens=sum(r.usage.total_tokens for r in call_records if r.usage),
    )


def _build_generation_log(
    records: list[RailCallRecord], options: Optional[GenerationOptions], total_duration: Optional[float]
) -> Optional[GenerationLog]:
    """Synthesize a ``GenerationLog`` from collected rail + generation records.

    Returns ``None`` unless ``options.log`` requests ``activated_rails`` or ``llm_calls``.
    ``stats`` is always included when either is requested (matching LLMRails); the
    per-rail ``activated_rails`` and flat ``llm_calls`` are gated on their own flags.
    ``internal_events`` / ``colang_history`` are rejected earlier (Colang-only).
    """
    log_options = options.log if options else None
    if not log_options or not (log_options.activated_rails or log_options.llm_calls):
        return None

    call_records = [record for record in records if _record_has_llm_call(record)]
    generation_log = GenerationLog()
    generation_log.stats = _build_generation_stats(records, call_records, total_duration)
    if log_options.activated_rails:
        generation_log.activated_rails = [_activated_rail(record) for record in records]
    if log_options.llm_calls:
        generation_log.llm_calls = [_call_info(record) for record in call_records]
    return generation_log


def _build_generation_response(
    response_text: str,
    reasoning_content: Optional[str],
    response: LLMResponse,
    log: Optional[GenerationLog] = None,
) -> GenerationResponse:
    """Build the structured ``GenerationResponse`` returned when ``options`` are supplied.

    Reasoning goes to the ``reasoning_content`` field with clean message content (no
    inline ``<think>`` prefix — that is the bare-return shape). Tool calls use the
    canonical ``ToolCall.to_dict()`` shape (dict arguments), matching LLMRails'
    ``GenerationResponse.tool_calls`` rather than the OpenAI-wire JSON-string shape
    used for the bare message. ``llm_output`` stays ``None`` to match LLMRails, whose
    ``raw_response`` source is never populated. ``log`` is set when the caller requested
    log details via ``options.log``.
    """
    result = GenerationResponse(response=[{"role": "assistant", "content": response_text}])
    if reasoning_content:
        result.reasoning_content = reasoning_content
    if response.tool_calls:
        result.tool_calls = [tool_call.to_dict() for tool_call in response.tool_calls]
    llm_metadata = _build_llm_metadata(response)
    if llm_metadata:
        result.llm_metadata = llm_metadata
    if log is not None:
        result.log = log
    return result


def _finalize_refusal(structured: bool, log: Optional[GenerationLog] = None) -> Union[LLMMessage, GenerationResponse]:
    """Shape the refusal message for the active return contract (structured vs bare).

    On the structured path the collected ``log`` (rails that ran up to the block) is
    attached when the caller requested it.
    """
    message: LLMMessage = {"role": "assistant", "content": REFUSAL_MESSAGE}
    if not structured:
        return message
    result = GenerationResponse(response=[message])
    if log is not None:
        result.log = log
    return result


def _response_content_for_capture(result: Union[LLMMessage, GenerationResponse]) -> Optional[str]:
    """Extract the assistant content for content capture from either return shape."""
    if isinstance(result, GenerationResponse):
        response = result.response
        if isinstance(response, list) and response:
            last = response[-1]
            content = last.get("content") if isinstance(last, dict) else None
            return content if isinstance(content, str) else None
        return response if isinstance(response, str) else None
    return result.get("content")


def _raise_on_unsupported_options(options: Optional[GenerationOptions], state: object) -> None:
    """Raise for ``GenerationOptions``/``state`` features IORails cannot honor.

    ``state`` and ``output_vars``/``output_data`` require Colang runtime state that
    IORails does not have and raise ``ValueError``. ``log.internal_events`` /
    ``log.colang_history`` are Colang-runtime-only and raise ``NotImplementedError``;
    ``log.activated_rails`` / ``log.llm_calls`` are supported. These raise rather than
    silently returning empty data, mirroring how LLMRails rejects options it cannot fulfill.
    """
    if state is not None:
        raise ValueError("state is not supported by IORails; it is a stateless input/output rails engine")
    if options is None:
        return
    if options.output_vars:
        raise ValueError("output_vars/output_data is not supported by IORails; it has no Colang context to return")
    log_options = options.log
    if log_options and (log_options.internal_events or log_options.colang_history):
        raise NotImplementedError(
            "GenerationLog `internal_events` and `colang_history` are not supported by IORails "
            "(no Colang runtime); use `activated_rails` and/or `llm_calls`"
        )


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


# TODO: _determine_rails_from_messages and _get_last_content_by_role are duplicated
# from nemoguardrails.rails.llm.llmrails. They should move to a shared checks-helper
# module that both engines import, rather than IORails depending on the heavy LLMRails
# module. Tracked for a future refactor.
def _determine_rails_from_messages(messages: list[dict]) -> Optional[dict]:
    """Pick which rails to run from message roles.

    user-only -> input, assistant-only -> output, both -> input and output.
    Returns ``{"rails": [...]}`` or ``None`` when there is no user/assistant
    message to check.
    """
    roles = {msg.get("role") for msg in messages}
    has_user = "user" in roles
    has_assistant = "assistant" in roles

    if not has_user and not has_assistant:
        log.warning(
            "check() called with no user or assistant messages. "
            "Only system, context, or tool messages found. "
            "Returning passing result without running rails."
        )
        return None

    if has_user and has_assistant:
        return {"rails": ["input", "output"]}
    if has_user:
        return {"rails": ["input"]}
    return {"rails": ["output"]}


def _get_last_content_by_role(messages: list[dict], role: str) -> str:
    """Return the content of the last message with the given role, or "".

    Non-string content (e.g. ``None`` on an assistant tool-call message) is
    normalized to "" so it can flow into the ``str``-typed ``RailsResult.content``.
    """
    for msg in reversed(messages):
        if msg.get("role") == role:
            content = msg.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _rewritten_user_message(result: RailResult) -> Optional[str]:
    """What the input rails rewrote the user message to, or None when they left it as it came."""
    return result.outcome.transform_text.get(TransformTarget.USER_MESSAGE.value)


def _rewritten_bot_message(result: RailResult) -> Optional[str]:
    """What the output rails rewrote the response to, or None when they left it as generated."""
    return result.outcome.transform_text.get(TransformTarget.BOT_MESSAGE.value)


@dataclass(frozen=True, slots=True)
class _GeneratedTurn:
    """The main model's response, and the messages it was generated from.

    The messages travel back with the response because input rails may have rewritten them, and
    what the model actually read is what the output rails and the generation log must see.
    ``response`` is None when the input rails blocked, so no generation happened.
    """

    response: Optional[LLMResponse]
    messages: LLMMessages


def _compile_only_deps(config: RailsConfig) -> RailDependencies:
    """Dependencies for the gate's trial compile: declared types and config, no live collaborators.

    Compilation reads ``llms`` for its key names alone — to reject a rail naming a model type
    the configuration does not declare — and never touches a model, so naming the types is
    enough. They must be named, though: with an empty mapping the gate would refuse every
    rail carrying a model binding while ``RailsManager`` accepted it, and a config would be
    routed to LLMRails for a model it actually has.

    ``config`` is carried for the same reason. The dependency check reads it to tell an
    in-process backend from a remote one, so withholding it would make the gate refuse a rail
    over its NIM that ``RailsManager`` then compiles happily.

    The key sets match by construction, since ``EngineRegistry`` is built from these same
    models and registers each one.
    """
    return RailDependencies(
        llms={model.type: None for model in config.models},
        llm_task_manager=None,
        config=config,
    )


class IORails(BaseGuardrails):
    """Workflow engine for accelerated Input/Output rails inference."""

    # Rail sections and flows that this engine can handle. Configs using anything
    # outside these sets fall back to LLMRails.
    SUPPORTED_RAILS = frozenset({"input", "output", "config", "tool_input", "tool_output"})
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
        rail_checks = (
            (config.rails.input.flows, SurfaceDirection.INPUT),
            (config.rails.output.flows, SurfaceDirection.OUTPUT),
        )
        deps = _compile_only_deps(config)
        for flows, direction in rail_checks:
            reason = cls._unservable_rails_reason(flows, direction, deps)
            if reason is not None:
                return reason

        tool_checks = (
            ("tool output", config.rails.tool_output.flows, cls.SUPPORTED_TOOL_OUTPUT_FLOWS),
            ("tool input", config.rails.tool_input.flows, cls.SUPPORTED_TOOL_INPUT_FLOWS),
        )
        for label, flows, supported in tool_checks:
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
    def _unservable_rails_reason(
        cls, flows: list[str], direction: SurfaceDirection, deps: RailDependencies
    ) -> Optional[str]:
        """Return why a configured input/output flow cannot run here, or None when all can.

        Scope is decided by the manifest rather than by a list held here: a surface this engine
        cannot run is one whose declared contract it cannot satisfy, which is what
        ``unservable_reason`` reports. Nothing enumerates the runnable rails, so adding one to
        the catalog needs no change in this engine.
        """
        # Surface-level refusals come first because they need no action import, so a config
        # naming an optional integration is not made to pay for one just to be refused.
        for flow in flows:
            reason = unservable_reason(flow, direction)
            if reason is not None:
                return reason

        # Only now, for a flow this engine will actually run, is the action worth importing.
        for flow in flows:
            try:
                compile_rail(flow, direction, deps)
            except RailCompilationError as exc:
                return str(exc)
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
        self._speculative_generation = self._speculative_generation_allowed(config)

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

    def _speculative_generation_allowed(self, config: RailsConfig) -> bool:
        """Whether input rails may race the main call, which an input rail that rewrites rules out.

        Speculative generation starts the model before the input rails finish, so a rewrite lands
        after the model has already read the original text. Applying it would describe a call that
        never happened, and ignoring it would send unmasked text to the model, so the race goes.
        """
        if not config.rails.input.speculative_generation:
            return False
        rewriting = self.rails_manager.transform_flows[RailDirection.INPUT]
        if not rewriting:
            return True
        warnings.warn(
            f"rails.input.speculative_generation is not honored alongside an input rail that rewrites the "
            f"user message ({', '.join(rewriting)}); generation waits for the input rails so the model reads "
            f"the rewritten text.",
            stacklevel=3,
        )
        return False

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
                try:
                    await self.rails_manager.stop()
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

    @staticmethod
    def _convert_to_messages(
        prompt: Optional[Union[str, LLMMessages]] = None,
        messages: Optional[Union[LLMMessages, str]] = None,
    ) -> LLMMessages:
        """Normalize a prompt string or a message list into the standard messages format.

        Argument order mirrors the ``Guardrails`` facade and LLMRails (prompt first).
        ``messages`` takes priority when both are supplied; a prompt string becomes a
        single user turn. A wrong-typed value — a list passed as ``prompt`` or a string
        passed as ``messages`` (the common positional mix-ups) — raises ``TypeError``;
        neither provided raises ``ValueError``.
        """
        if messages is not None and not isinstance(messages, list):
            raise TypeError("messages must be a list of {'role', 'content'} dicts; pass a string via prompt=")
        if prompt is not None and not isinstance(prompt, str):
            raise TypeError("prompt must be a string; pass a message list via messages=")
        if messages:
            return messages
        if prompt:
            return [{"role": "user", "content": prompt}]
        raise ValueError("Neither prompt nor messages provided for generation")

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[LLMMessages] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        **kwargs,
    ) -> Union[LLMMessage, GenerationResponse]:
        """Synchronous version of generate_async.

        Telemetry is disabled for the ephemeral IORails object used for
        the ``generate()`` call. For production use, use the asynchronous
        `generate_async()` and `stream_async()` methods for non-streaming
        and streaming requests respectively.
        """
        messages = self._convert_to_messages(prompt, messages)

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
                return await iorails_engine.generate_async(messages=messages, options=options, **kwargs)

        return asyncio.run(_run_sync_iorails())

    async def generate_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[LLMMessages] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        **kwargs,
    ) -> Union[LLMMessage, GenerationResponse]:
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
        messages = self._convert_to_messages(prompt, messages)
        await self.start()
        metrics_ctx = request_metrics() if self._metrics_enabled else nullcontext()
        with metrics_ctx:
            try:
                return await self._generate_async_queue.submit(self._run_generate, messages, options=options, **kwargs)
            except asyncio.QueueFull:
                if self._metrics_enabled:
                    record_nonstream_rejected()
                raise

    async def _run_generate(
        self,
        messages: LLMMessages,
        options: Optional[Union[dict, GenerationOptions]] = None,
        **kwargs,
    ) -> Union[LLMMessage, GenerationResponse]:
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
                result = await self._do_generate(messages, req_id, request_span, options=options, **kwargs)
            except Exception:
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.error("[%s] generate_async failed time=%.1fms", req_id, elapsed_ms, exc_info=True)
                raise
            # Capture content once here at the traced_request boundary so any
            # future early-return added to _do_generate is covered automatically.
            if self._content_capture_enabled:
                set_request_content(request_span, messages, _response_content_for_capture(result))
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
        self,
        messages: LLMMessages,
        req_id: str,
        request_span: Optional["Span"] = None,
        *,
        options: Optional[Union[dict, GenerationOptions]] = None,
        **kwargs,
    ) -> Union[LLMMessage, GenerationResponse]:
        """Core pipeline: tool-result rails -> input rails -> LLM call -> tool-call + output rails."""
        log.info("[%s] generate_async called", req_id)
        log.debug("[%s] generate_async messages=%s", req_id, truncate(messages))

        options = _coerce_generation_options(options)
        _raise_on_unsupported_options(options, kwargs.get("state"))
        # When options are supplied we return a structured GenerationResponse
        # (mirroring LLMRails' `if gen_options:` branch); otherwise a bare LLMMessage.
        has_generation_options = options is not None
        # Pass llm_params (including tool definitions) unchanged to the LLM call.
        llm_kwargs = options.llm_params if (options and options.llm_params) else {}
        input_enabled = options.rails.input if options else True
        output_enabled = options.rails.output if options else True
        tool_input_enabled = options.rails.tool_input if options else True
        tool_output_enabled = options.rails.tool_output if options else True

        # Per-rail + generation records accumulated for the GenerationLog (built only when
        # options.log requests it). Each rail check and the main call append their record.
        t_start = time.monotonic()
        records: list[RailCallRecord] = []

        def _blocked_return() -> Union[LLMMessage, GenerationResponse]:
            """Refusal shaped for the return contract, carrying the log of rails run so far."""
            log_obj = (
                _build_generation_log(records, options, time.monotonic() - t_start) if has_generation_options else None
            )
            return _finalize_refusal(has_generation_options, log_obj)

        # Agent/client executes tool-calls and sends results to Main LLM with prior conversation history.
        # Symmetric with INPUT rails
        log.info("[%s] Running tool result rails", req_id)
        tool_result = await self.rails_manager.are_tool_results_safe(messages, enabled=tool_input_enabled)
        records.extend(tool_result.records)
        if not tool_result.is_safe:
            log.info("[%s] Tool result blocked: %s", req_id, display_reason(tool_result))
            if self._metrics_enabled:
                record_request_blocked(RailDirection.INPUT)
            return _blocked_return()

        if self._speculative_generation:
            turn = await self._do_generate_speculative(
                messages, req_id, llm_kwargs, request_span, input_enabled=input_enabled, records_out=records
            )
        else:
            turn = await self._do_generate_sequential(
                messages, req_id, llm_kwargs, input_enabled=input_enabled, records_out=records
            )

        if turn.response is None:
            return _blocked_return()

        # What the model read, which is what the output rails must check: an input rail may have
        # rewritten the user message, and checking the text it replaced would judge the wrong turn.
        response = turn.response
        messages = turn.messages

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
            records.extend(tool_call.records)
            if not tool_call.is_safe:
                log.info("[%s] Tool call blocked: %s", req_id, display_reason(tool_call))
                if self._metrics_enabled:
                    record_request_blocked(RailDirection.OUTPUT)
                return _blocked_return()

        # Output rails check the final answer, not reasoning traces.
        # Reasoning is re-attached as <think> tags only below so reasoning intentionally bypasses output
        # rails, matching LLMRails.
        # A tool-call-only response skips output rails (no text to check)
        # Tool calls have their own `ToolOutputRails` set of rails separate to `OutputRails`
        is_tool_call_only = bool(response.tool_calls) and not response_text
        if not is_tool_call_only:
            log.info("[%s] Running output rails", req_id)
            output_result = await self.rails_manager.is_output_safe(messages, response_text, enabled=output_enabled)
            records.extend(output_result.records)
            if not output_result.is_safe:
                log.info("[%s] Output blocked: %s", req_id, display_reason(output_result))
                if self._metrics_enabled:
                    record_request_blocked(RailDirection.OUTPUT)
                return _blocked_return()

            rewritten = _rewritten_bot_message(output_result)
            if rewritten is not None:
                log.info("[%s] Output rails rewrote the response", req_id)
                response_text = rewritten

        if has_generation_options:
            log_obj = _build_generation_log(records, options, time.monotonic() - t_start)
            return _build_generation_response(response_text, reasoning_content, response, log_obj)

        # Bare return path: reasoning is delivered inline as a <think> prefix
        # (LLMRails legacy shape). The structured path above instead puts it in the
        # reasoning_content field and keeps the message content clean.
        if reasoning_content:
            response_text = f"<think>{reasoning_content}</think>\n" + response_text

        return _build_assistant_message(response_text, response.tool_calls)

    async def _timed_main_call(self, messages: LLMMessages, llm_kwargs: dict) -> TimedLLMResponse:
        """Call the main model, returning the response with wall-clock start/finish and a monotonic duration.

        Shared by the sequential and speculative paths so the main call's ``RailCallRecord``
        always carries real timing (the speculative path previously left it None).
        """
        started_at = time.time()
        t0 = time.monotonic()
        response = await self.engine_registry.model_call("main", messages, **llm_kwargs)
        return TimedLLMResponse(
            response=response,
            started_at=started_at,
            finished_at=time.time(),
            duration=time.monotonic() - t0,
        )

    async def _do_generate_sequential(
        self,
        messages: LLMMessages,
        req_id: str,
        llm_kwargs: dict,
        *,
        input_enabled: Union[bool, list[str]] = True,
        records_out: Optional[list[RailCallRecord]] = None,
    ) -> _GeneratedTurn:
        """Sequential path: input rails block, or rewrite, before LLM generation starts."""
        log.info("[%s] Running input rails", req_id)
        input_result = await self.rails_manager.is_input_safe(messages, enabled=input_enabled)
        if records_out is not None:
            records_out.extend(input_result.records)
        if not input_result.is_safe:
            log.info("[%s] Input blocked: %s", req_id, display_reason(input_result))
            if self._metrics_enabled:
                record_request_blocked(RailDirection.INPUT)
            return _GeneratedTurn(response=None, messages=messages)

        rewritten = _rewritten_user_message(input_result)
        if rewritten is not None:
            log.info("[%s] Input rails rewrote the user message", req_id)
            messages = rewrite_user_message(messages, rewritten)

        log.info("[%s] Calling main LLM", req_id)
        timed_llm_response = await self._timed_main_call(messages, llm_kwargs)
        if records_out is not None:
            provider = self.engine_registry.provider_name("main")
            prompt = serialize_prompt(messages)
            records_out.append(_make_generation_record(timed_llm_response, provider, prompt))
        return _GeneratedTurn(response=timed_llm_response.response, messages=messages)

    async def _do_generate_speculative(
        self,
        messages: LLMMessages,
        req_id: str,
        llm_kwargs: dict,
        request_span: Optional["Span"] = None,
        *,
        input_enabled: Union[bool, list[str]] = True,
        records_out: Optional[list[RailCallRecord]] = None,
    ) -> _GeneratedTurn:
        """Speculative path: input rails and LLM generation race concurrently.

        The messages come back unchanged: the race is refused for a config whose input rails may
        rewrite, because the model reads the text before the rails have finished with it.
        """
        log.info("[%s] Speculative generation: launching input rails + LLM concurrently", req_id)

        rails_task = asyncio.create_task(self.rails_manager.is_input_safe(messages, enabled=input_enabled))
        gen_task = asyncio.create_task(self._timed_main_call(messages, llm_kwargs))

        try:
            response = await self._parallel_input_rail_and_response_generation(
                rails_task,
                gen_task,
                req_id,
                request_span,
                records_out=records_out,
                main_prompt=serialize_prompt(messages),
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

        return _GeneratedTurn(response=response, messages=messages)

    async def _parallel_input_rail_and_response_generation(
        self,
        rails_task: asyncio.Task,
        gen_task: asyncio.Task,
        req_id: str,
        request_span: Optional["Span"] = None,
        *,
        records_out: Optional[list[RailCallRecord]] = None,
        main_prompt: str = "",
    ) -> Optional[LLMResponse]:
        """Race input rails against LLM generation, return LLMResponse or None (rejected)."""

        def _record_generation(timed: TimedLLMResponse) -> None:
            if records_out is not None:
                provider = self.engine_registry.provider_name("main")
                records_out.append(_make_generation_record(timed, provider, main_prompt))

        done, _ = await asyncio.wait({rails_task, gen_task}, return_when=asyncio.FIRST_COMPLETED)

        first_completed = (
            GuardrailsAttributes.SPECULATIVE_FIRST_COMPLETED_INPUT_RAILS
            if rails_task in done
            else GuardrailsAttributes.SPECULATIVE_FIRST_COMPLETED_GENERATION
        )

        if rails_task in done:
            input_result = rails_task.result()
            if records_out is not None:
                records_out.extend(input_result.records)

            if not input_result.is_safe:
                log.info("[%s] Input blocked (speculative): %s", req_id, display_reason(input_result))
                gen_task.cancel()
                # Use gather(return_exceptions=True) instead of bare await: when both
                # tasks finish simultaneously, gen_task may hold a stored exception that
                # would leak through suppress(CancelledError). gather drains it safely.
                gen_result = (await asyncio.gather(gen_task, return_exceptions=True))[0]
                if isinstance(gen_result, TimedLLMResponse):
                    # Generation completed before cancellation took effect — record the call that was made.
                    _record_generation(gen_result)
                elif isinstance(gen_result, BaseException) and not isinstance(gen_result, asyncio.CancelledError):
                    log.warning("[%s] LLM generation error suppressed: %s", req_id, gen_result)
                if self._metrics_enabled:
                    record_request_blocked(RailDirection.INPUT)
                set_speculative_span_attrs(
                    request_span, first_completed, GuardrailsAttributes.SPECULATIVE_FIRST_COMPLETED_INPUT_RAILS
                )
                return None

            # Rails passed — wait for generation to finish
            timed = await gen_task
            set_speculative_span_attrs(request_span, first_completed, "none")
        else:
            # Generation finished first — wait for rails verdict
            timed = gen_task.result()

            input_result = await rails_task
            if records_out is not None:
                records_out.extend(input_result.records)

            if not input_result.is_safe:
                log.info("[%s] Input blocked (speculative, gen-first): %s", req_id, display_reason(input_result))
                _record_generation(timed)
                if self._metrics_enabled:
                    record_request_blocked(RailDirection.INPUT)
                set_speculative_span_attrs(
                    request_span, first_completed, GuardrailsAttributes.SPECULATIVE_FIRST_COMPLETED_INPUT_RAILS
                )
                return None

            set_speculative_span_attrs(request_span, first_completed, "none")

        log.debug("[%s] Main LLM response: %s", req_id, truncate(timed.response.content))
        _record_generation(timed)
        return timed.response

    def check(self, messages: LLMMessages, rail_types: Optional[list[RailType]] = None) -> RailsResult:
        """Synchronous version of ``check_async``.

        Mirrors ``generate``: spins up a short-lived IORails engine with tracing
        and metrics disabled and runs the check on it. For production use, prefer
        the asynchronous ``check_async``.
        """
        if check_sync_call_from_async_loop():
            raise RuntimeError(
                "You are using the sync `check` inside async code. You should replace with `await check_async(...)`."
            )

        sync_config = self.config.model_copy(deep=True)
        if sync_config.tracing is not None:
            sync_config.tracing.enabled = False
        if sync_config.metrics is not None:
            sync_config.metrics.enabled = False

        async def _run_sync_iorails():
            """Spin up a short-lived IORails engine for one synchronous check call."""
            async with IORails(sync_config, _report_usage=False) as iorails_engine:
                return await iorails_engine.check_async(messages, rail_types=rail_types)

        return asyncio.run(_run_sync_iorails())

    async def check_async(self, messages: LLMMessages, rail_types: Optional[list[RailType]] = None) -> RailsResult:
        """Run input and/or output rails on messages without main-LLM generation.

        When ``rail_types`` is None the rails to run are auto-detected from the
        message roles (user-only -> input, assistant-only -> output, both ->
        input and output). When provided, exactly the named rail types run; an
        empty list (``[]``) runs no rails and returns PASSED.

        Submitted through the same admission queue as ``generate_async`` so the
        check path shares non-streaming concurrency limits, request metrics, and
        the per-request trace span.
        """
        await self.start()
        metrics_ctx = request_metrics() if self._metrics_enabled else nullcontext()
        with metrics_ctx:
            try:
                return await self._generate_async_queue.submit(self._run_check, messages, rail_types)
            except asyncio.QueueFull:
                if self._metrics_enabled:
                    record_nonstream_rejected()
                raise

    async def _run_check(self, messages: LLMMessages, rail_types: Optional[list[RailType]]) -> RailsResult:
        """Queue-worker entry for ``check_async``: wrap the rails in a request span."""
        tracer = self._tracer if self._tracing_enabled else None
        with traced_request(tracer) as (request_span, req_id):
            t0 = time.monotonic()
            try:
                result = await self._do_check(messages, rail_types, req_id)
            except Exception:
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.error("[%s] check failed time=%.1fms", req_id, elapsed_ms, exc_info=True)
                raise
            if self._content_capture_enabled:
                set_request_content(request_span, messages, result.content)
            elapsed_ms = (time.monotonic() - t0) * 1000
            log.info(
                "[%s] check completed time=%.1fms status=%s",
                req_id,
                elapsed_ms,
                result.status.value,
            )
            return result

    async def _do_check(
        self,
        messages: LLMMessages,
        rail_types: Optional[list[RailType]],
        req_id: str,
    ) -> RailsResult:
        """Core check pipeline: run the requested input/output rails on messages."""
        log.info("[%s] check called", req_id)
        log.debug("[%s] check messages=%s", req_id, truncate(messages))

        if rail_types is not None:
            rails_to_run = [rail_type.value for rail_type in rail_types]
        else:
            determined = _determine_rails_from_messages(messages)
            if determined is None:
                last = messages[-1].get("content") if messages else ""
                return RailsResult(status=RailStatus.PASSED, content=last if isinstance(last, str) else "")
            rails_to_run = determined["rails"]

        # Which direction's text the caller gets back, and so which rewrites it can observe: with
        # output rails in play the answer is the response, and an input rewrite is internal to the
        # check. Matches how LLMRails decides what ``check`` reports.
        reports_output = "output" in rails_to_run
        if reports_output:
            pass_content = _get_last_content_by_role(messages, "assistant")
        else:
            pass_content = _get_last_content_by_role(messages, "user")
        original_content = pass_content

        if "input" in rails_to_run:
            user_content = _get_last_content_by_role(messages, "user")
            # Skip when there is no user content: the content-safety action requires
            # user_input and would otherwise raise, surfacing a false BLOCK.
            if user_content:
                log.info("[%s] Running input rails", req_id)
                input_result = await self.rails_manager.is_input_safe(messages)
                if not input_result.is_safe:
                    log.info("[%s] Input blocked: %s", req_id, display_reason(input_result))
                    if self._metrics_enabled:
                        record_request_blocked(RailDirection.INPUT)
                    return RailsResult(
                        status=RailStatus.BLOCKED, content=REFUSAL_MESSAGE, rail=input_result.triggered_rail
                    )
                rewritten = _rewritten_user_message(input_result)
                if rewritten is not None:
                    log.info("[%s] Input rails rewrote the user message", req_id)
                    messages = rewrite_user_message(messages, rewritten)
                    if not reports_output:
                        pass_content = rewritten
            else:
                log.info("[%s] Input rails requested but no user content to check; skipping", req_id)

        if reports_output:
            bot_response = _get_last_content_by_role(messages, "assistant")
            # Skip when there is no assistant content: the content-safety action requires
            # bot_response and would otherwise raise, surfacing a false BLOCK.
            if bot_response:
                log.info("[%s] Running output rails", req_id)
                output_result = await self.rails_manager.is_output_safe(messages, bot_response)
                if not output_result.is_safe:
                    log.info("[%s] Output blocked: %s", req_id, display_reason(output_result))
                    if self._metrics_enabled:
                        record_request_blocked(RailDirection.OUTPUT)
                    return RailsResult(
                        status=RailStatus.BLOCKED, content=REFUSAL_MESSAGE, rail=output_result.triggered_rail
                    )
                rewritten = _rewritten_bot_message(output_result)
                if rewritten is not None:
                    log.info("[%s] Output rails rewrote the response", req_id)
                    pass_content = rewritten
            else:
                log.info("[%s] Output rails requested but no assistant content to check; skipping", req_id)

        # MODIFIED is decided by comparing content, as LLMRails does, and names no rail: a rewrite
        # is not a rail "triggering", and several rails may have contributed to the final text.
        if pass_content != original_content:
            return RailsResult(status=RailStatus.MODIFIED, content=pass_content)
        return RailsResult(status=RailStatus.PASSED, content=pass_content)

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
        if self._speculative_generation:
            warnings.warn(
                "speculative_generation is not supported for streaming; falling back to sequential",
                stacklevel=2,
            )
        self._validate_streaming_with_output_rails()

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

        async def _generation_task(request_span):
            """Background task: input rails → stream LLM chunks → push to handler.

            ``request_span`` is the IORails request span (or ``None`` when
            tracing is disabled), captured by the caller from
            ``traced_request`` and passed in explicitly — never fetched via
            ``trace.get_current_span()`` which could return the host app's
            ambient span and pollute unrelated traces.

            Inherits the request ID from the caller context via create_task().
            """
            nonlocal accumulated_tool_calls, messages
            req_id = get_request_id()
            t0 = time.monotonic()
            try:
                # Step 0: Tool-result rails. Client/agent harness executes tool calls and sends
                # results of execution to Main LLM along with prior conversation history
                # Symmetric with INPUT rails for dialog use-case
                log.info("[%s] Running tool result rails", req_id)
                tool_result = await self.rails_manager.are_tool_results_safe(messages, enabled=tool_input_enabled)
                if not tool_result.is_safe:
                    log.info("[%s] Tool result blocked: %s", req_id, display_reason(tool_result))
                    if self._metrics_enabled:
                        record_request_blocked(RailDirection.INPUT)
                    await streaming_handler.push_chunk(
                        self._guardrails_violation_payload(
                            f"Blocked by tool input rails: {client_reason(tool_result)}", "tool_input_rails"
                        )
                    )
                    await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore[arg-type]
                    return

                # Step 1: Input rails (non-streaming)
                log.info("[%s] Running input rails", req_id)
                input_result = await self.rails_manager.is_input_safe(messages, enabled=input_enabled)
                if not input_result.is_safe:
                    log.info("[%s] Input blocked: %s", req_id, display_reason(input_result))
                    if self._metrics_enabled:
                        record_request_blocked(RailDirection.INPUT)
                    await streaming_handler.push_chunk(
                        self._guardrails_violation_payload(
                            f"Blocked by input rails: {client_reason(input_result)}", "input_rails"
                        )
                    )
                    await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore[arg-type]
                    return

                # Input rails finish before the first token is requested, so a rewrite reaches the
                # model here exactly as it does when not streaming. The output-rail wrapper was
                # handed the original messages when the stream was set up, and keeps them: LLMRails
                # builds its streaming output-rail context from the raw messages too.
                rewritten = _rewritten_user_message(input_result)
                if rewritten is not None:
                    log.info("[%s] Input rails rewrote the user message", req_id)
                    messages = rewrite_user_message(messages, rewritten)

                # Step 2: Stream main LLM content from structured response.
                # delta_content is forwarded as text chunks; delta_tool_calls are
                # accumulated and surfaced as a terminal JSON chunk after the text
                # stream ends. Reasoning is dropped for LLMRails compatibility.
                log.info("[%s] Streaming main LLM", req_id)
                content_parts: list[str] = []
                # Usage from the terminal usage-only chunk is folded into the END_OF_STREAM
                # frame below (not its own frame), matching LLMRails' single terminal frame.
                pending_usage_metadata: Optional[dict] = None
                async for chunk in self.engine_registry.stream_model_call("main", messages, **llm_kwargs):
                    chunk_metadata = _stream_chunk_metadata(chunk)
                    if chunk.delta_content:
                        content_parts.append(chunk.delta_content)
                        await streaming_handler.push_chunk(chunk.delta_content, chunk_metadata)
                    elif chunk.usage is not None:
                        pending_usage_metadata = chunk_metadata
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

                await streaming_handler.push_chunk(END_OF_STREAM, pending_usage_metadata)  # type: ignore[arg-type]
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
                error_payload = build_streaming_error_payload(e)
                await streaming_handler.push_chunk(error_payload)
                await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore[arg-type]
            finally:
                elapsed_ms = (time.monotonic() - t0) * 1000
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
                            try:
                                log.info("[%s] stream_async called", req_id)
                                log.debug("[%s] stream_async messages=%s", req_id, truncate(messages))

                                task = asyncio.create_task(_generation_task(request_span))
                                try:
                                    # Determine base iterator: with or without output rails
                                    if self._has_streaming_output_rails:
                                        base_iterator = self._run_output_rails_in_streaming(
                                            streaming_handler=streaming_handler,
                                            messages=messages,
                                            enabled=output_enabled,
                                            include_metadata=include_metadata,
                                        )
                                    else:
                                        base_iterator = streaming_handler

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
                                                display_reason(tool_call),
                                            )
                                            if self._metrics_enabled:
                                                record_request_blocked(RailDirection.OUTPUT)
                                            violation = self._guardrails_violation_payload(
                                                f"Blocked by tool output rails: {client_reason(tool_call)}",
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
                finally:
                    self._stream_semaphore.release()

        return _wrapped_iterator()

    async def _run_output_rails_in_streaming(
        self,
        streaming_handler: AsyncIterator[Union[str, dict]],
        messages: LLMMessages,
        *,
        enabled: Union[bool, list[str]] = True,
        include_metadata: Optional[bool] = False,
    ) -> AsyncGenerator[Union[str, dict], None]:
        """Buffer streamed chunks and run output rails on each batch.

        Uses the same ``RollingBuffer`` and ``stream_first`` semantics as
        LLMRails:
        - ``stream_first=True``: yield chunks immediately, then run output
          rails.  If unsafe, inject an error and stop.
        - ``stream_first=False``: run output rails first, only yield chunks
          if safe.
        """

        # Unpack streaming config and get the buffer strategy
        output_streaming_config = self.config.rails.output.streaming
        stream_first = output_streaming_config.stream_first
        buffer_strategy = get_buffer_strategy(output_streaming_config)
        # A rewrite arrives per buffered batch; saying so once is enough to explain a stream whose
        # masking rail appears to have done nothing.
        rewrite_reported = False

        async for chunk_batch in buffer_strategy(streaming_handler):
            user_output_chunks = chunk_batch.user_output_chunks
            bot_response_chunk = buffer_strategy.format_chunks(chunk_batch.processing_context)

            # If the batch contains an error chunk (generation or downstream HTTP),
            # yield it directly and stop — don't feed error JSON through output rails.
            for chunk in user_output_chunks:
                if _is_stream_error_chunk(chunk):
                    yield chunk
                    return

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
                log.info("[%s] Output blocked: %s", req_id, display_reason(output_result))
                if self._metrics_enabled:
                    record_request_blocked(RailDirection.OUTPUT)
                violation = self._guardrails_violation_payload(
                    f"Blocked by output rails: {client_reason(output_result)}", "output_rails"
                )
                yield _frame_for_stream(violation, include_metadata)
                return

            # A rewrite cannot be applied to a stream: with stream_first the chunk has already
            # reached the caller, and without it the rewrite covers one buffered batch rather than
            # the response. So the check counts only as non-blocking, which is what LLMRails does.
            if output_result.outcome.is_transform and not rewrite_reported:
                rewrite_reported = True
                log.info(
                    "[%s] Output rails rewrote the response, which streaming cannot apply; rails %s",
                    req_id,
                    list(self.rails_manager.transform_flows[RailDirection.OUTPUT]),
                )

            if not stream_first:
                for chunk in user_output_chunks:
                    yield chunk

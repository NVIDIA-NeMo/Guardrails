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

"""Inline OpenTelemetry instrumentation for the IORails engine.

All OpenTelemetry API imports are isolated in this module so the rest of the
guardrails package never imports ``opentelemetry`` directly.  When the
``opentelemetry-api`` package is not installed, the public entry points
``is_tracing_enabled``, ``get_tracer``, ``get_meter``, and ``traced_request``
degrade gracefully (returning ``False``, ``None``, or a no-span / no-metric
passthrough respectively).  Lower-level helpers like ``request_span`` and
``trace_id_to_request_id`` require OTEL to be available and are only
reachable through ``traced_request`` when a non-``None`` tracer is provided.
"""

import json
import logging
import os
import secrets
import time
import warnings
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    NamedTuple,
    Optional,
    Tuple,
)

from nemoguardrails.guardrails.guardrails_types import (
    REQUEST_ID_BYTES,
    REQUEST_ID_HEX_CHARS,
    LLMMessages,
    RailDirection,
    _set_request_id,
    get_request_id,
    reset_request_id,
    set_new_request_id,
)
from nemoguardrails.tracing.constants import (
    EventNames,
    GenAIAttributes,
    GuardrailsAttributes,
    MetricNames,
    OperationNames,
    OtelContentCapture,
    SpanNames,
    SystemConstants,
)

log = logging.getLogger(__name__)

_OTEL_AVAILABLE: bool
if TYPE_CHECKING:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry import trace
    from opentelemetry.metrics import (
        CallbackOptions,
        Counter,
        Histogram,
        Meter,
        Observation,
        UpDownCounter,
    )
    from opentelemetry.trace import Span, SpanKind, StatusCode, Tracer, format_trace_id

    from nemoguardrails.guardrails.async_work_queue import AsyncWorkQueue
    from nemoguardrails.rails.llm.config import MetricsConfig, TracingConfig

    _OTEL_AVAILABLE = True
else:
    try:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry import trace
        from opentelemetry.metrics import CallbackOptions, Observation
        from opentelemetry.trace import SpanKind, StatusCode, format_trace_id

        _OTEL_AVAILABLE = True
    except ImportError:  # pragma: no cover
        _OTEL_AVAILABLE = False

# Module-level tracer singleton.  Thread-safe: the OTEL spec requires that
# ``Tracer`` methods are safe for concurrent use, and ``get_tracer()`` is
# designed to be called once and cached (see "Get a Tracer" at
# https://opentelemetry.io/docs/specs/otel/trace/api/#get-a-tracer).
# A benign race on first access (two threads both see ``None``) is harmless
# because ``trace.get_tracer()`` returns equivalent instances for the same
# instrumentation scope.
_tracer = None


def get_tracer() -> Optional["Tracer"]:
    """Return a cached OpenTelemetry tracer for nemo-guardrails, or ``None``.

    The tracer is obtained via the OTEL API (not SDK), following the library
    instrumentation best practice.  The application is responsible for
    configuring a ``TracerProvider`` before any spans are created.
    """
    global _tracer
    if not _OTEL_AVAILABLE:
        return None
    if _tracer is None:
        from importlib.metadata import PackageNotFoundError, version

        try:
            lib_version = version("nemoguardrails")
        except PackageNotFoundError:  # pragma: no cover
            lib_version = "0.0.0-dev"

        _tracer = trace.get_tracer(
            SystemConstants.SYSTEM_NAME,
            instrumenting_library_version=lib_version,
            schema_url="https://opentelemetry.io/schemas/1.26.0",
        )
    return _tracer


# Module-level meter singleton.  Same caching rationale as ``_tracer`` above:
# ``metrics.get_meter()`` is designed to be called once per instrumentation
# scope and returns equivalent instances on subsequent calls, so a benign race
# on first access is harmless.
_meter = None
_request_instruments: Optional["RequestInstruments"] = None


@dataclass(frozen=True, slots=True)
class RequestInstruments:
    """Request-level OTEL instruments for the IORails engine.

    Field names mirror the emitted metric names (minus the ``guardrails.``
    prefix).  The saturation-metric group covers the full request lifecycle:

    * Aggregate: ``requests_active`` (``guardrails.requests.active``)
    * Non-streaming path: ``nonstream_rejections``
      (``guardrails.nonstream.rejections``); the two gauges
      ``nonstream.queued`` and ``nonstream.active`` are registered
      separately via ``register_nonstream_saturation_gauges`` because
      ObservableGauges need a live queue reference.
    * Streaming path: ``stream_active`` (``guardrails.stream.active``)
      and ``stream_rejections`` (``guardrails.stream.rejections``).
    """

    requests: "Counter"
    errors: "Counter"
    blocked: "Counter"
    duration: "Histogram"
    requests_active: "UpDownCounter"
    nonstream_rejections: "Counter"
    stream_active: "UpDownCounter"
    stream_rejections: "Counter"


def get_meter() -> Optional["Meter"]:
    """Return a cached OpenTelemetry meter for nemo-guardrails, or ``None``.

    The meter is obtained via the OTEL API (not SDK), following the library
    instrumentation best practice.  The application is responsible for
    configuring a ``MeterProvider`` before any metrics are recorded; without
    one, the API returns a no-op meter and all emissions are silently
    discarded.
    """
    global _meter
    if not _OTEL_AVAILABLE:
        return None
    if _meter is None:
        from importlib.metadata import PackageNotFoundError, version

        try:
            lib_version = version("nemoguardrails")
        except PackageNotFoundError:  # pragma: no cover
            lib_version = "0.0.0-dev"

        _meter = otel_metrics.get_meter(
            SystemConstants.SYSTEM_NAME,
            version=lib_version,
            schema_url="https://opentelemetry.io/schemas/1.26.0",
        )
    return _meter


def _ensure_request_instruments() -> Optional[RequestInstruments]:
    """Lazily create the request-level instruments and return them as a
    :class:`RequestInstruments`.  Returns ``None`` when the OTEL API is not
    installed.
    """
    global _request_instruments
    meter = get_meter()
    if meter is None:
        return None
    if _request_instruments is None:
        _request_instruments = RequestInstruments(
            requests=meter.create_counter(
                MetricNames.REQUESTS,
                description="Total guardrails requests handled",
                unit="1",
            ),
            errors=meter.create_counter(
                MetricNames.REQUESTS_ERRORS,
                description="Guardrails requests that ended in an unhandled error",
                unit="1",
            ),
            blocked=meter.create_counter(
                MetricNames.REQUESTS_BLOCKED,
                description="Guardrails requests blocked by an input or output rail",
                unit="1",
            ),
            duration=meter.create_histogram(
                MetricNames.REQUEST_DURATION,
                description="End-to-end guardrails request duration",
                unit="s",
                explicit_bucket_boundaries_advisory=[
                    0.005,
                    0.01,
                    0.025,
                    0.05,
                    0.075,
                    0.1,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                    2.5,
                    5.0,
                    7.5,
                    10.0,
                ],
            ),
            requests_active=meter.create_up_down_counter(
                MetricNames.REQUESTS_ACTIVE,
                description=("Guardrails requests currently in flight"),
                unit="1",
            ),
            nonstream_rejections=meter.create_counter(
                MetricNames.NONSTREAM_REJECTIONS,
                description="Rejected non-streaming requests",
                unit="1",
            ),
            stream_active=meter.create_up_down_counter(
                MetricNames.STREAM_ACTIVE,
                description="In-progress streaming requests",
                unit="1",
            ),
            stream_rejections=meter.create_counter(
                MetricNames.STREAM_REJECTIONS,
                description="Rejected streaming requests",
                unit="1",
            ),
        )
    return _request_instruments


def register_nonstream_saturation_gauges(
    queue: "AsyncWorkQueue",
    is_running: Callable[[], bool],
) -> None:
    """Register ``guardrails.nonstream.queued`` + ``guardrails.nonstream.active``
    ObservableGauges on the module-level Meter.

    ObservableGauges read live state at collection time, so both metrics
    reflect the *current* non-streaming queue + worker occupancy with no
    drift risk vs. an UpDownCounter lineage.

    ``is_running`` is a zero-arg callable returning ``bool``, deferred
    so each collection re-reads the current state (passing the bool
    directly would bake its start-time value into the closure).  The
    callbacks return an empty observation list when it returns ``False``
    — the state the flag holds after ``IORails.stop()`` flips
    ``self._running`` back to False.  OTEL Python has no public
    unregister API for observable instruments, so this "no data points"
    fallback is the only way to stop a dead IORails instance from
    polluting collection.

    No-op when the OTEL API is unavailable or no MeterProvider is
    configured.
    """
    meter = get_meter()
    if meter is None:
        return

    def _queued_callback(options: "CallbackOptions") -> Iterable["Observation"]:
        """Observe current backlog: items in the admission queue not yet
        picked up by a worker.  Returns ``[]`` after ``IORails.stop()``
        so a dead instance emits no data points."""
        if not is_running():
            return []
        return [Observation(queue.num_pending())]

    def _active_callback(options: "CallbackOptions") -> Iterable["Observation"]:
        """Observe current occupancy: workers currently executing a
        WorkItem.  Returns ``[]`` after ``IORails.stop()``, same
        rationale as :func:`_queued_callback`."""
        if not is_running():
            return []
        return [Observation(queue.num_busy_workers())]

    meter.create_observable_gauge(
        MetricNames.NONSTREAM_QUEUED,
        callbacks=[_queued_callback],
        description="Non-streaming requests buffered in the admission queue",
        unit="1",
    )
    meter.create_observable_gauge(
        MetricNames.NONSTREAM_ACTIVE,
        callbacks=[_active_callback],
        description="Non-streaming requests currently executing on a worker",
        unit="1",
    )


_INVALID_TRACE_ID = 0


def trace_id_to_request_id(span: "Span") -> str:
    """Derive a human-readable request ID from the span's OTEL trace ID.

    Returns the last ``REQUEST_ID_HEX_CHARS`` hex characters of the 128-bit
    trace ID (the low 64 bits, which carry the highest entropy).  When the
    trace ID is zero (e.g. a ``NoOpTracerProvider`` is active) a random
    fallback is used.
    """
    ctx = span.get_span_context()
    if ctx.trace_id == _INVALID_TRACE_ID:
        return secrets.token_hex(REQUEST_ID_BYTES)
    return format_trace_id(ctx.trace_id)[-REQUEST_ID_HEX_CHARS:]


def record_span_error(span: Optional["Span"], exc: BaseException) -> None:
    """Record an exception on an OTEL span and set its status to ERROR.

    Also sets the ``error.type`` attribute to the exception's class name
    (per OTEL GenAI conditional-required convention).  Safe to call with
    ``None`` (no-op).  Use from every span helper's ``except`` block and
    from callers that swallow exceptions before they can propagate.

    Best-effort: any failure while annotating the span (e.g. a broken
    exporter or SDK) is swallowed so it can never mask the original
    exception the caller is about to re-raise — notably ``CancelledError``
    / ``GeneratorExit`` on a cancelled stream.  Only ``Exception`` is
    suppressed, so a ``BaseException`` raised *inside* the SDK still
    propagates.
    """
    if span is None:
        return
    with suppress(Exception):
        span.set_attribute("error.type", type(exc).__name__)
        span.record_exception(exc)
        span.set_status(StatusCode.ERROR, str(exc))


def mark_rail_stop(span: Optional["Span"], is_safe: bool) -> None:
    """Set ``rail.stop=True`` on a rail span when the rail blocked the request.

    Safe to call with ``None`` (no-op) so callers don't have to branch on
    whether a real span was produced — matches the ``record_span_error``
    idiom.  Only marks stop when *is_safe* is ``False``; a passing rail
    leaves the attribute unset.
    """
    if span is None or is_safe:
        return
    span.set_attribute(GuardrailsAttributes.RAIL_STOP, True)


def set_speculative_span_attrs(
    span: Optional["Span"],
    first_completed: str,
    first_rejector: str,
) -> None:
    """Stamp speculative-generation outcome attributes on a request span.

    Records which branch of the speculative race finished first
    (input rails vs. main LLM generation) and which one ultimately
    rejected the request, on the IORails ``guardrails.request`` span.
    Safe to call with ``None`` (no-op) so callers don't have to branch
    on whether tracing is enabled — matches the ``record_span_error`` /
    ``mark_rail_stop`` idiom.
    """
    if span is None:
        return
    span.set_attribute(GuardrailsAttributes.SPECULATIVE_MODE_ACTIVE, True)
    span.set_attribute(GuardrailsAttributes.SPECULATIVE_FIRST_COMPLETED, first_completed)
    span.set_attribute(GuardrailsAttributes.SPECULATIVE_FIRST_REJECTOR, first_rejector)
    # TODO: Add it to metrics on next version
    # span.set_attribute(GuardrailsAttributes.SPECULATIVE_TIME_SAVED_MS, time_saved_ms)


# Maps an OpenAI-style ``role`` to the OTEL GenAI legacy event name used
# when content capture emits per-message span events (i.e. when the
# stability opt-in does NOT select the new structured attribute form).
# ``function`` role (OpenAI legacy function-call format) is deliberately
# excluded — it will need its own decision when function-call support lands.
_LEGACY_EVENT_BY_ROLE = {
    "system": EventNames.GEN_AI_SYSTEM_MESSAGE,
    "user": EventNames.GEN_AI_USER_MESSAGE,
    "assistant": EventNames.GEN_AI_ASSISTANT_MESSAGE,
    "tool": EventNames.GEN_AI_TOOL_MESSAGE,
}


def _use_json_span_format() -> bool:
    """Return True iff OTEL_SEMCONV_STABILITY_OPT_IN selects JSON span attrs.

    The env var holds a comma-separated list of opt-in tokens.  When
    ``gen_ai_latest_experimental`` is present, content is emitted as
    JSON-encoded span attributes, otherwise as legacy per-message span events.
    Read fresh each call so runtime changes to the env var take effect
    immediately.
    """
    raw_env_value = os.environ.get(OtelContentCapture.STABILITY_OPT_IN_ENV, "")
    tokens = {tok.strip() for tok in raw_env_value.split(",")}
    return OtelContentCapture.STABILITY_OPT_IN_LATEST in tokens


def _system_parts_from_messages(messages: LLMMessages) -> list[dict]:
    """Return the bare OTEL GenAI ``parts`` for system messages only.

    Feeds ``gen_ai.system_instructions``, which the spec defines as a flat
    list of parts with no role wrapper (every entry is implicitly system).
    Asymmetric with :func:`_non_system_input_messages`, which keeps the role
    wrapper — the two attributes have different shapes by spec.  Entries
    missing ``role`` or ``content`` are skipped silently.

    Example::

        >>> _system_parts_from_messages([
        ...     {"role": "system", "content": "be helpful"},
        ...     {"role": "user", "content": "hi"},
        ... ])
        [{"type": "text", "content": "be helpful"}]
    """
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role is None or content is None:
            continue
        if role == "system":
            out.append({"type": "text", "content": content})
    return out


def _non_system_input_messages(messages: LLMMessages) -> list[dict]:
    """Return the OTEL GenAI ``gen_ai.input.messages`` form for non-system messages.

    Each non-system message is role-wrapped as ``{"role": role, "parts":
    [{"type": "text", "content": content}]}``.  Named for the attribute it
    populates rather than "parts" because — unlike
    :func:`_system_parts_from_messages` — it keeps the role wrapper.

    Example::

        >>> _non_system_input_messages([
        ...     {"role": "system", "content": "be helpful"},
        ...     {"role": "user", "content": "hi"},
        ... ])
        [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}]
    """
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role is None or content is None:
            continue
        if role != "system":
            out.append({"role": role, "parts": [{"type": "text", "content": content}]})
    return out


def _set_llm_call_content_json(
    span: "Span",
    input_messages: LLMMessages,
    output_text: Optional[str],
) -> None:
    """JSON-attribute branch of :func:`set_llm_call_content`.

    Sets ``gen_ai.input.messages``, ``gen_ai.output.messages``, and
    ``gen_ai.system_instructions`` as JSON-encoded span attributes per
    the latest experimental OTEL GenAI semantic conventions.  Attributes
    are only set when non-empty so backends can distinguish "no system
    instructions" from "system instructions == ''".
    """
    system_parts = _system_parts_from_messages(input_messages)
    if system_parts:
        span.set_attribute(GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS, json.dumps(system_parts))

    non_system = _non_system_input_messages(input_messages)
    if non_system:
        span.set_attribute(GenAIAttributes.GEN_AI_INPUT_MESSAGES, json.dumps(non_system))

    if output_text is not None:
        output_messages = [{"role": "assistant", "parts": [{"type": "text", "content": output_text}]}]
        span.set_attribute(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages))


def _set_llm_call_content_events(
    span: "Span",
    input_messages: LLMMessages,
    output_text: Optional[str],
) -> None:
    """Legacy-event branch of :func:`set_llm_call_content`.

    Adds one span event per input message (``gen_ai.system.message`` /
    ``gen_ai.user.message`` / ``gen_ai.assistant.message`` /
    ``gen_ai.tool.message``) plus a ``gen_ai.choice`` event for the
    assistant output.  Roles not in :data:`_LEGACY_EVENT_BY_ROLE`
    (e.g. ``function``) are skipped silently.
    """
    for msg in input_messages:
        role = msg.get("role")
        content = msg.get("content")
        if role is None or content is None:
            continue
        event_name = _LEGACY_EVENT_BY_ROLE.get(role)
        if event_name is None:
            continue
        span.add_event(event_name, attributes={"role": role, "content": content})

    if output_text is not None:
        span.add_event(
            EventNames.GEN_AI_CHOICE,
            attributes={"index": 0, "message.role": "assistant", "message.content": output_text},
        )


def set_llm_call_content(
    span: Optional["Span"],
    input_messages: LLMMessages,
    output_text: Optional[str] = None,
) -> None:
    """Capture input/output messages on a span representing a model interaction.

    Used for both ``gen_ai.*`` CLIENT spans (LLM calls) and the
    ``guardrails.request`` SERVER span — the OTEL GenAI semconv
    attribute names apply to any span that represents a model
    interaction, so reusing the names lets backends correlate the outer
    guardrails request with the inner LLM call by attribute name alone.

    Dispatches on :func:`_use_json_span_format`:

    * **JSON attrs** (``OTEL_SEMCONV_STABILITY_OPT_IN`` includes
      ``gen_ai_latest_experimental``): :func:`_set_llm_call_content_json`
      sets the JSON-encoded ``gen_ai.input.messages``,
      ``gen_ai.output.messages``, and ``gen_ai.system_instructions``
      span attributes per the latest experimental OTEL GenAI semantic
      conventions.
    * **Legacy events** (default): :func:`_set_llm_call_content_events`
      adds one span event per input message plus a ``gen_ai.choice``
      event for the assistant output.

    Safe to call with ``span=None`` (no-op) so callers don't have to
    branch on whether tracing is enabled.  Caller is responsible for
    checking the content-capture flag — this helper does NOT re-check
    :func:`is_content_capture_enabled` so it stays cheap on hot paths.
    """
    if span is None:
        return
    if _use_json_span_format():
        _set_llm_call_content_json(span, input_messages, output_text)
    else:
        _set_llm_call_content_events(span, input_messages, output_text)


def set_request_content(
    span: Optional["Span"],
    input_messages: LLMMessages,
    output_text: Optional[str] = None,
) -> None:
    """Capture caller-facing input/output on the ``guardrails.request`` SERVER span.

    Uses ``guardrails.request.input`` (JSON-encoded input messages) and
    ``guardrails.request.output`` (the text actually returned to the caller)
    rather than the ``gen_ai.*`` attribute names used on LLM CLIENT spans.
    This distinction matters on block paths: the LLM CLIENT span records the
    raw model response, while the SERVER span records the refusal message —
    the same ``gen_ai.output.messages`` name on both spans would carry
    different values and confuse backends correlating the two.

    ``guardrails.request.input`` is always a JSON-encoded list of role/content
    message objects matching the caller's input.  ``guardrails.request.output``
    is the plain string that IORails returned (REFUSAL_MESSAGE on block paths,
    the model's response text on the success path).  ``output_text=None``
    suppresses the output attribute entirely — used by the streaming path when
    the stream produced no content, so an empty output is not falsely recorded.

    Safe to call with ``span=None`` (no-op).
    """
    if span is None:
        return
    span.set_attribute(GuardrailsAttributes.REQUEST_INPUT, json.dumps(input_messages))
    if output_text is not None:
        span.set_attribute(GuardrailsAttributes.REQUEST_OUTPUT, output_text)


def set_rail_content(
    span: Optional["Span"],
    rail_input: dict[str, Any],
    reason: Optional[str] = None,
) -> None:
    """Capture rail input + (optionally) block reason on a ``guardrails.rail`` span.

    Sets ``guardrails.rail.input`` to the JSON-encoded *rail_input* dict
    (typically ``{"messages": [...], "bot_response": ...}``).  When
    *reason* is non-None, also sets ``guardrails.rail.reason`` — caller
    passes the human-readable block reason from the failing rail (or
    ``None`` when the rail passed, in which case only the input
    attribute is recorded).

    Safe to call with ``span=None`` (no-op).  No GenAI semconv covers
    rail spans, so these attributes live under the guardrails.* namespace
    alongside ``rail.type`` / ``rail.name`` / ``rail.stop``.
    """
    if span is None:
        return
    span.set_attribute(GuardrailsAttributes.RAIL_INPUT, json.dumps(rail_input))
    if reason is not None:
        span.set_attribute(GuardrailsAttributes.RAIL_REASON, reason)


@contextmanager
def request_span(tracer: "Tracer") -> Generator[Tuple["Span", str], None, None]:
    """Create a live ``guardrails.request`` SERVER span.

    Yields ``(span, request_id)`` where *request_id* is derived from the
    OTEL trace ID.  The span is ended automatically when the block exits.
    If an exception propagates, the span records it and sets ERROR status
    before re-raising.
    """
    with tracer.start_as_current_span(
        SpanNames.GUARDRAILS_REQUEST,
        kind=SpanKind.SERVER,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        req_id = trace_id_to_request_id(span)
        span.set_attribute(GenAIAttributes.GEN_AI_OPERATION_NAME, OperationNames.GUARDRAILS)
        span.set_attribute("request.id", req_id)
        try:
            yield span, req_id
        except BaseException as exc:
            record_span_error(span, exc)
            raise


@contextmanager
def rail_span(
    tracer: Optional["Tracer"], flow: str, direction: RailDirection
) -> Generator[Optional["Span"], None, None]:
    """Create a ``guardrails.rail`` INTERNAL span for a single rail execution.

    Yields the span (or ``None`` when *tracer* is ``None``).
    The caller should set ``rail.stop`` on the span after execution if the
    rail blocked the request.
    """
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(
        SpanNames.GUARDRAILS_RAIL,
        kind=SpanKind.INTERNAL,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        span.set_attribute(GuardrailsAttributes.RAIL_TYPE, direction.value)
        span.set_attribute(GuardrailsAttributes.RAIL_NAME, flow)
        try:
            yield span
        except BaseException as exc:
            record_span_error(span, exc)
            raise


@contextmanager
def action_span(tracer: Optional["Tracer"], action_name: str) -> Generator[Optional["Span"], None, None]:
    """Create a ``guardrails.action`` INTERNAL span for a rail action execution.

    Yields the span (or ``None`` when *tracer* is ``None``).
    """
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(
        SpanNames.GUARDRAILS_ACTION,
        kind=SpanKind.INTERNAL,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        span.set_attribute(GuardrailsAttributes.ACTION_NAME, action_name)
        try:
            yield span
        except BaseException as exc:
            record_span_error(span, exc)
            raise


@contextmanager
def llm_call_span(
    tracer: Optional["Tracer"],
    model_name: str,
    provider_name: str,
    operation_name: str = "chat",
) -> Generator[Optional["Span"], None, None]:
    """Create a CLIENT span for an LLM call following GenAI semantic conventions.

    Span name follows the OTEL pattern: ``"{operation_name} {model_name}"``.

    ``operation_name`` defaults to ``"chat"`` because IORails only issues
    chat completions. In the future if any other non-chat LLM  operations are
    supported, callers should pass an explicit ``operation_name`` from the
    OTEL GenAI semantic conventions.

    Yields the span (or ``None`` when *tracer* is ``None``).
    """
    if tracer is None:
        yield None
        return
    span_name = f"{operation_name} {model_name}"
    with tracer.start_as_current_span(
        span_name,
        kind=SpanKind.CLIENT,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        span.set_attribute(GenAIAttributes.GEN_AI_OPERATION_NAME, operation_name)
        span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_MODEL, model_name)
        span.set_attribute(GenAIAttributes.GEN_AI_PROVIDER_NAME, provider_name)
        try:
            yield span
        except BaseException as exc:
            record_span_error(span, exc)
            raise


@contextmanager
def api_call_span(tracer: Optional["Tracer"], api_name: str) -> Generator[Optional["Span"], None, None]:
    """Create a CLIENT span for a non-LLM API call (e.g., jailbreak detection).

    Uses the ``api.name`` attribute rather than ``gen_ai.operation.name``
    because these APIs are plain HTTP endpoints, not GenAI operations.
    ``http.*`` transport attributes can be added additively later without
    conflict.  Yields the span (or ``None`` when *tracer* is ``None``).
    """
    if tracer is None:
        yield None
        return
    span_name = f"api {api_name}"
    with tracer.start_as_current_span(
        span_name,
        kind=SpanKind.CLIENT,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        span.set_attribute(GuardrailsAttributes.API_NAME, api_name)
        try:
            yield span
        except BaseException as exc:
            record_span_error(span, exc)
            raise


def is_tracing_enabled(config_tracing: Optional["TracingConfig"]) -> bool:
    """Return ``True`` when inline OTEL tracing should be active.

    Requires the ``opentelemetry-api`` package to be installed **and**
    ``config.tracing.enabled`` to be ``True``.  Other ``TracingConfig``
    fields (``adapters``, ``span_format``) are used by the LLMRails
    post-hoc tracing path and are ignored here.
    """
    if config_tracing is None or not config_tracing.enabled:
        return False
    if not _OTEL_AVAILABLE:
        warnings.warn(
            "Tracing is enabled in config but the opentelemetry-api package is "
            "not installed.  Install it with: pip install nemoguardrails[tracing]",
            UserWarning,
            stacklevel=2,
        )
        return False
    return True


def is_content_capture_enabled(config_tracing: Optional["TracingConfig"]) -> bool:
    """Return True when message content should be captured onto spans.

    ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` is the
    primary control — when set, it overrides any config-file value so
    operators have a single OTEL-standard env var that flips capture
    across all services regardless of what the deployed config says.
    Recognized values (case-insensitive, surrounding whitespace
    stripped): ``true`` / ``1`` enable; ``false`` / ``0`` disable; any
    other value falls through to the config field.

    When the env var is absent or unrecognized, capture is on iff
    ``config.tracing.enable_content_capture`` is True.

    Callers should ALSO require :func:`is_tracing_enabled` before
    treating capture as active — there is no point capturing content
    onto spans that won't be exported.  This helper deliberately does
    not perform that check itself so it stays orthogonal to the
    tracing-enabled signal (and so tests can exercise each independently).
    """
    env_value = os.environ.get(OtelContentCapture.CAPTURE_CONTENT_ENV, "").strip().lower()
    if env_value in ("true", "1"):
        return True
    if env_value in ("false", "0"):
        return False
    if config_tracing is None:
        return False
    return getattr(config_tracing, "enable_content_capture", False)


def are_metrics_enabled(config_metrics: Optional["MetricsConfig"]) -> bool:
    """Return ``True`` when inline OTEL metrics should be emitted.

    Requires the ``opentelemetry-api`` package to be installed **and**
    ``config.metrics.enabled`` to be ``True``.  Independent of
    :func:`is_tracing_enabled` — OTEL signals (traces, metrics, logs) are
    designed to be toggled independently so customers can, for example,
    run metrics-only for cost-optimized SLO dashboards without the
    overhead of full trace export.
    """
    if config_metrics is None or not config_metrics.enabled:
        return False
    if not _OTEL_AVAILABLE:
        warnings.warn(
            "Metrics are enabled in config but the opentelemetry-api package is "
            "not installed.  Install it with: pip install nemoguardrails[tracing]",
            UserWarning,
            stacklevel=2,
        )
        return False
    return True


class TracedRequest(NamedTuple):
    """Handle yielded by ``traced_request``.

    ``span`` is the IORails ``guardrails.request`` span when tracing is
    enabled, or ``None`` when it is not.  ``request_id`` is always a
    16-char hex string.  Unpacks as ``(span, request_id)`` for callers
    that prefer positional access.
    """

    span: Optional["Span"]
    request_id: str


def _cleanup_request_id(token) -> None:
    """Reset the request-ID ContextVar from a cleanup path, tolerating the
    one expected ``ValueError``.

    ``ContextVar.reset()`` raises ``ValueError("... was created in a
    different Context")`` when called from a different asyncio Context
    than where ``.set()`` was called.  That happens during async-generator
    cleanup (``aclose()`` running in an outer task's context) and is the
    only ``ValueError`` that ``reset_request_id`` raises today.  Any
    other ``ValueError`` indicates an unexpected bug in the helper and is
    re-raised so callers see it.
    """
    try:
        reset_request_id(token)
    except ValueError as exc:
        if "different Context" not in str(exc):
            raise


def record_request_blocked(direction: RailDirection) -> None:
    """Increment ``guardrails.requests.blocked`` with a ``rail.type`` label.

    Fires at the block sites in ``iorails.py`` (``_do_generate`` for the
    non-streaming path, ``_generation_task`` for streaming) whenever the
    request returns ``REFUSAL_MESSAGE`` because an input or output rail
    flagged it.  The counter is cumulative over the process lifetime; a
    per-rail grain (``rail.name``) will be added in split-2 alongside
    ``guardrails.rail.blocked``.

    No-op when the OTEL API is unavailable or instruments cannot be created.
    """
    instruments = _ensure_request_instruments()
    if instruments is None:
        return
    instruments.blocked.add(1, attributes={GuardrailsAttributes.RAIL_TYPE: direction.value})


def record_request_error(exc: BaseException) -> None:
    """Increment ``guardrails.requests.errors`` with an ``error.type`` label.

    ``request_metrics`` already bumps this counter when an exception
    propagates through its ``except`` branch (the non-streaming path).
    Streaming code paths catch-and-swallow exceptions inside
    ``_generation_task`` — converting them to error-payload chunks —
    so the counter never sees them via propagation.  Those paths should
    call this helper explicitly so the errors counter reflects ALL failed
    requests, not just those whose exceptions bubble up.

    No-op when the OTEL API is unavailable or instruments cannot be created.
    Best-effort: a failure inside the meter SDK is swallowed so it can never
    mask the original exception the caller is about to re-raise.
    """
    instruments = _ensure_request_instruments()
    if instruments is None:
        return
    with suppress(Exception):
        instruments.errors.add(1, attributes={"error.type": type(exc).__name__})


def record_stream_rejected() -> None:
    """Increment ``guardrails.stream.rejections`` by 1.

    Called from the streaming path when a request arrives while the stream
    concurrency semaphore is fully occupied (``_stream_semaphore.locked()``).
    """
    instruments = _ensure_request_instruments()
    if instruments is None:
        return
    instruments.stream_rejections.add(1)


def record_nonstream_rejected() -> None:
    """Increment ``guardrails.nonstream.rejections`` by 1.

    Called from the non-streaming path when the admission queue rejects a
    submission with ``asyncio.QueueFull`` (the queue's ``reject_on_full``
    behaviour, triggered when ``NONSTREAM_QUEUE_DEPTH`` is exceeded).
    """
    instruments = _ensure_request_instruments()
    if instruments is None:
        return
    instruments.nonstream_rejections.add(1)


@contextmanager
def stream_active_metric() -> Generator[None, None, None]:
    """Context manager that tracks a stream as active for its full lifetime.

    ``+1`` on enter / ``-1`` on exit (``finally``) on
    ``guardrails.stream.active`` (UpDownCounter).  No-op when metrics are
    unavailable.  Wrap the block where the stream holds a semaphore permit.
    """
    instruments = _ensure_request_instruments()
    if instruments is None:
        yield
        return
    instruments.stream_active.add(1)
    try:
        yield
    finally:
        instruments.stream_active.add(-1)


@contextmanager
def request_metrics() -> Generator[None, None, None]:
    """Emit request-level OTEL metrics around the wrapped block.

    Increments ``guardrails.requests`` on entry, bumps
    ``guardrails.requests.active`` (UpDownCounter) for the duration of
    the block, records ``guardrails.request.duration`` in seconds on
    exit, and increments ``guardrails.requests.errors`` with an
    ``error.type`` attribute when the block raises.

    ``requests.active`` covers both non-streaming (queue-wait + execution)
     and streaming (semaphore hold) requests.
     Summing the per-path saturation metrics
    (``nonstream.queued``, ``nonstream.active``, ``stream.active``)
    should approximate this value at any collection instant.

    Instruments are created lazily on first use.  No-op when the OTEL
    API is not installed or instruments cannot be created.
    """
    instruments = _ensure_request_instruments()
    if instruments is None:
        yield
        return
    t0 = time.monotonic()
    instruments.requests.add(1)
    instruments.requests_active.add(1)
    try:
        yield
    except BaseException as exc:
        record_request_error(exc)
        raise
    finally:
        # Best-effort emission: a broken meter SDK must never mask the
        # original exception propagating through ``finally``.  Guard each
        # emit independently so a failure in one still attempts the other —
        # the active-counter decrement must run to avoid leaking the gauge.
        with suppress(Exception):
            instruments.requests_active.add(-1)
        duration_s = time.monotonic() - t0
        with suppress(Exception):
            instruments.duration.record(duration_s)


@contextmanager
def traced_request(tracer: Optional["Tracer"], metrics_enabled: bool = False) -> Generator[TracedRequest, None, None]:
    """Unified request context: sets request ID, optionally creates a span
    and/or emits request-level metrics.

    The two signals are gated **independently**:

    * ``tracer is not None`` → a live ``guardrails.request`` SERVER span
      is created and the request ID is derived from its trace ID.
    * ``metrics_enabled=True`` → emit request-level OTEL metrics

    All four combinations are valid.  Metrics-only (``tracer=None,
    metrics_enabled=True``) is a supported setup for customers running
    cheap SLO dashboards without full trace export.

    Yields a :class:`TracedRequest` (``span``, ``request_id``).  Callers
    that want to mark the request span ERROR from a deeply-nested scope
    should capture the yielded span and pass it explicitly to
    ``record_span_error`` — never rely on ``trace.get_current_span()``
    which can return the host app's ambient span when IORails tracing is
    disabled.

    The request-ID ContextVar is always cleaned up on exit via
    :func:`_cleanup_request_id`, which tolerates the expected
    cross-context ``ValueError`` that async-generator cleanup can raise.
    """
    metrics_ctx = request_metrics() if metrics_enabled else nullcontext()
    if tracer is not None:
        with metrics_ctx, request_span(tracer) as (span, req_id):
            token = _set_request_id(req_id)
            try:
                yield TracedRequest(span, req_id)
            finally:
                _cleanup_request_id(token)
    else:
        with metrics_ctx:
            token = set_new_request_id()
            try:
                yield TracedRequest(None, get_request_id())
            finally:
                _cleanup_request_id(token)

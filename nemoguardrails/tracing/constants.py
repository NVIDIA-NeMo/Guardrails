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

"""OpenTelemetry constants, semantic conventions, and engine-agnostic
GenAI client-side metric instruments for NeMo Guardrails.

The OTEL GenAI client-side metric helpers (``LLMInstruments``,
``record_token_usage``, ``llm_operation_duration``,
``record_time_to_first_chunk``, ``record_time_per_output_chunk``) live
here next to the metric-name and attribute constants they emit.  They
are engine-agnostic — any caller that issues an LLM call can use them
to satisfy the OTEL GenAI semantic conventions.
"""

import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator, Optional

if TYPE_CHECKING:
    from opentelemetry.metrics import Histogram

    from nemoguardrails.types import UsageInfo


class SpanKind:
    """String constants for span kinds."""

    SERVER = "server"
    CLIENT = "client"
    INTERNAL = "internal"


class SpanTypes:
    """Internal span type identifiers used in span mapping.

    These are internal identifiers used to categorize spans before mapping
    to actual span names. They represent the type of operation being traced.

    Note: 'llm_call' maps to various GenAI semantic convention span types
    like inference (gen_ai.inference.client), embeddings, etc.
    """

    # NeMo Guardrails-specific internal types
    INTERACTION = "interaction"  # Entry point to guardrails
    RAIL = "rail"  # Rail execution
    ACTION = "action"  # Action execution

    # GenAI-related type (maps to official semantic conventions)
    LLM_CALL = "llm_call"  # maps to gen_ai.inference.client

    # NOTE: might use more specific types in the future
    # could add more specific types that align with semantic conventions:
    # INFERENCE = "inference"  # for gen_ai.inference.client spans
    # EMBEDDING = "embedding"  # for gen_ai.embeddings.client spans


class SpanNamePatterns:
    """Patterns used for identifying span types from span names."""

    # patterns that indicate SERVER spans
    INTERACTION = "interaction"
    GUARDRAILS_REQUEST_PATTERN = "guardrails.request"

    # patterns that indicate CLIENT spans
    GEN_AI_PREFIX = "gen_ai."
    LLM = "llm"
    COMPLETION = "completion"


class SystemConstants:
    """System-level constants for NeMo Guardrails."""

    SYSTEM_NAME = "nemo-guardrails"
    UNKNOWN = "unknown"


class OtelContentCapture:
    """OTEL environment-variable names and tokens for content-capture gating.

    Two independent OTEL-standard env vars control content capture:

    * ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` — fallback
      enable switch when ``config.tracing.enable_content_capture`` is
      unset/False.  Truthy values (``"true"``, ``"1"``) turn capture on.
    * ``OTEL_SEMCONV_STABILITY_OPT_IN`` — comma-separated stability
      opt-in list.  When ``"gen_ai_latest_experimental"`` is present,
      content is emitted as new-form span attributes
      (``gen_ai.input.messages`` etc.); otherwise as legacy span events
      (``gen_ai.user.message`` etc.).
    """

    CAPTURE_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
    STABILITY_OPT_IN_ENV = "OTEL_SEMCONV_STABILITY_OPT_IN"
    STABILITY_OPT_IN_LATEST = "gen_ai_latest_experimental"


class GenAIAttributes:
    """GenAI semantic convention attributes following the draft specification.

    Note: These are based on the experimental OpenTelemetry GenAI semantic conventions
    since they are not yet available in the stable semantic conventions package.

    See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
    """

    GEN_AI_SYSTEM = "gen_ai.system"  # @deprecated

    GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"

    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
    GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"

    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_RESPONSE_ID = "gen_ai.response.id"
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

    # Required label on the ``gen_ai.client.token.usage`` metric.
    # Allowed values (from spec): "input" or "output" only.  Reasoning
    # and cached tokens are span-only attributes, NOT valid token.type
    # metric label values.
    GEN_AI_TOKEN_TYPE = "gen_ai.token.type"

    # New-form content-capture span attributes (opt-in, gated by
    # OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental).  Values
    # are JSON-encoded strings.  Default emission uses the legacy event
    # form (EventNames.GEN_AI_*_MESSAGE / GEN_AI_CHOICE) instead.
    GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
    GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
    GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"


class CommonAttributes:
    """Common OpenTelemetry attributes used across spans."""

    SPAN_KIND = "span.kind"


class GuardrailsAttributes:
    """NeMo Guardrails-specific attributes for spans."""

    # rail attributes
    RAIL_TYPE = "rail.type"
    RAIL_NAME = "rail.name"
    RAIL_STOP = "rail.stop"
    RAIL_DECISIONS = "rail.decisions"

    # rail content-capture attributes (opt-in alongside the GenAI
    # content-capture knob).  No GenAI semconv exists for rail spans,
    # so these live under the guardrails.* namespace.  RAIL_INPUT is
    # a JSON-encoded snapshot of the rail's inputs; RAIL_REASON is set
    # only when the rail blocks.
    RAIL_INPUT = "guardrails.rail.input"
    RAIL_REASON = "guardrails.rail.reason"

    # request-level content-capture attributes on the guardrails.request
    # SERVER span.  These record the caller-facing input and output —
    # what the caller sent and what was returned — which differs
    # from gen_ai.input/output.messages on the LLM CLIENT span on block
    # paths (where the LLM CLIENT span records the raw model response
    # while the SERVER span records the refusal message).  Using a
    # distinct attribute namespace avoids conflating the two semantics.
    REQUEST_INPUT = "guardrails.request.input"
    REQUEST_OUTPUT = "guardrails.request.output"

    # action attributes
    ACTION_NAME = "action.name"
    ACTION_HAS_LLM_CALLS = "action.has_llm_calls"
    ACTION_LLM_CALLS_COUNT = "action.llm_calls_count"
    ACTION_PARAM_PREFIX = "action.param."

    # api call attributes (non-LLM HTTP APIs such as jailbreak detection)
    API_NAME = "api.name"

    # llm attributes (application-level, not provider-level)
    LLM_CACHE_HIT = "llm.cache.hit"

    # speculative generation attributes
    SPECULATIVE_MODE_ACTIVE = "speculative_generation.mode_active"
    SPECULATIVE_FIRST_COMPLETED = "speculative_generation.first_completed"
    SPECULATIVE_FIRST_REJECTOR = "speculative_generation.first_rejector"

    SPECULATIVE_FIRST_COMPLETED_INPUT_RAILS = "input_rails"
    SPECULATIVE_FIRST_COMPLETED_GENERATION = "generation"


class SpanNames:
    """Standard span names following OpenTelemetry GenAI semantic conventions.

    Based on: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/

    IMPORTANT: Span names must be low cardinality to avoid performance issues.
    Variable/high cardinality data (like specific rail types, model names, etc.)
    should go in attributes instead of the span name.
    """

    # server spans (entry points); NeMo Guardrails specific
    GUARDRAILS_REQUEST = "guardrails.request"  # Entry point for guardrails processing

    # internal spans; NeMo Guardrails specific
    GUARDRAILS_RAIL = "guardrails.rail"  # Use attributes for rail type/name
    GUARDRAILS_ACTION = "guardrails.action"  # Use attributes for action name

    # client spans (LLM calls), following official GenAI semantic conventions
    # "Span name SHOULD be `{gen_ai.operation.name} {gen_ai.request.model}`"
    # since model names are high cardinality, we'll build these dynamically
    # these are fallback operation names when model is unknown
    GEN_AI_COMPLETION = "completion"
    GEN_AI_CHAT = "chat"
    GEN_AI_EMBEDDING = "embedding"


class MetricNames:
    """OTEL metric names emitted by the IORails engine.

    These names are part of the library's public API — customers point
    dashboards and alerts at them.  Tests deliberately assert on the raw
    strings rather than these constants so the assertions verify the wire
    contract instead of re-referencing the same symbol the production code
    uses.
    """

    REQUESTS = "guardrails.requests"
    REQUESTS_ERRORS = "guardrails.requests.errors"
    REQUESTS_BLOCKED = "guardrails.requests.blocked"
    REQUESTS_ACTIVE = "guardrails.requests.active"
    REQUEST_DURATION = "guardrails.request.duration"

    # Non-streaming (AsyncWorkQueue) saturation signals
    NONSTREAM_QUEUED = "guardrails.nonstream.queued"
    NONSTREAM_ACTIVE = "guardrails.nonstream.active"
    NONSTREAM_REJECTIONS = "guardrails.nonstream.rejections"

    # Streaming (semaphore) saturation signals
    STREAM_ACTIVE = "guardrails.stream.active"
    STREAM_REJECTIONS = "guardrails.stream.rejections"

    # OTEL GenAI semantic-convention metric names emitted by IORails for
    # downstream LLM calls. These names are mandated by OTEL hence ``gen_ai``
    # prefix separate to ``guardrails`` metrics above.
    GEN_AI_CLIENT_TOKEN_USAGE = "gen_ai.client.token.usage"
    GEN_AI_CLIENT_OPERATION_DURATION = "gen_ai.client.operation.duration"
    GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK = "gen_ai.client.operation.time_to_first_chunk"
    GEN_AI_CLIENT_OPERATION_TIME_PER_OUTPUT_CHUNK = "gen_ai.client.operation.time_per_output_chunk"


class TokenType:
    """Allowed values for the ``gen_ai.token.type`` metric label.

    Per OTEL GenAI semconv, only ``input`` and ``output`` are valid.
    Reasoning and cached tokens are exposed as span attributes
    (``gen_ai.usage.reasoning.output_tokens`` etc.), not as additional
    ``token.type`` values on the ``gen_ai.client.token.usage`` metric.
    """

    INPUT = "input"
    OUTPUT = "output"


class OperationNames:
    """Standard operation names for GenAI semantic conventions.

    Note: This only defines standard LLM operations. Custom actions and tasks
    should be passed through as-is since they are dynamic and user-defined.
    """

    # standard LLM operations (from GenAI semantic conventions)
    COMPLETION = "completion"
    CHAT = "chat"
    EMBEDDING = "embedding"

    # default operation for guardrails interactions
    GUARDRAILS = "guardrails"


class EventNames:
    """Standard event names for OpenTelemetry GenAI semantic conventions.

    Based on official spec at:
    https://github.com/open-telemetry/semantic-conventions/blob/main/model/gen-ai/events.yaml
    """

    GEN_AI_SYSTEM_MESSAGE = "gen_ai.system.message"
    GEN_AI_USER_MESSAGE = "gen_ai.user.message"
    GEN_AI_ASSISTANT_MESSAGE = "gen_ai.assistant.message"
    GEN_AI_TOOL_MESSAGE = "gen_ai.tool.message"

    GEN_AI_CHOICE = "gen_ai.choice"

    GEN_AI_CONTENT_PROMPT = "gen_ai.content.prompt"  # @deprecated ; use GEN_AI_USER_MESSAGE  instead, as we are still using text completions we should use it!
    GEN_AI_CONTENT_COMPLETION = "gen_ai.content.completion"  # @deprecated ; use GEN_AI_ASSISTANT_MESSAGE, but as we are still using text completions we should use it!


class GuardrailsEventNames:
    """NeMo Guardrails-specific event names (not OTel GenAI conventions).

    These events represent internal guardrails state changes, not LLM API calls.
    They use a guardrails-specific namespace to avoid confusion with OTel GenAI semantic conventions.
    """

    UTTERANCE_USER_FINISHED = "guardrails.utterance.user.finished"
    UTTERANCE_BOT_STARTED = "guardrails.utterance.bot.started"
    UTTERANCE_BOT_FINISHED = "guardrails.utterance.bot.finished"

    USER_MESSAGE = "guardrails.user_message"


class GuardrailsEventTypes:
    """NeMo Guardrails internal event type constants.

    These are the type values from internal guardrails events.
    """

    UTTERANCE_USER_ACTION_FINISHED = "UtteranceUserActionFinished"
    USER_MESSAGE = "UserMessage"

    START_UTTERANCE_BOT_ACTION = "StartUtteranceBotAction"
    UTTERANCE_BOT_ACTION_FINISHED = "UtteranceBotActionFinished"

    SYSTEM_MESSAGE = "SystemMessage"


# Module-level singleton.  The lazy-init pattern mirrors
# ``_request_instruments`` in ``guardrails.telemetry``: a benign race on
# first access is harmless because OTEL guarantees ``meter.create_*``
# returns equivalent instances for the same instrumentation scope.
_llm_instruments: Optional["LLMInstruments"] = None


@dataclass(frozen=True, slots=True)
class LLMInstruments:
    """LLM-call-scope OTEL instruments for downstream model calls.

    These metrics fire once per LLM call (not once per IORails request)
    and follow the OTEL GenAI semantic conventions exactly — the field
    names mirror the metric names with the ``gen_ai.client.`` prefix
    stripped, and both are Histograms (per spec).

    * ``token_usage`` — ``gen_ai.client.token.usage`` Histogram, unit
      ``{token}``.  Records input and output tokens as separate
      observations distinguished by the required ``gen_ai.token.type``
      label (``input`` or ``output``).
    * ``operation_duration`` — ``gen_ai.client.operation.duration``
      Histogram, unit ``s``.  Records the wall-clock time of each
      LLM call from request issue to response completion.
    * ``time_to_first_chunk`` — ``gen_ai.client.operation.time_to_first_chunk``
      Histogram, unit ``s``.  Streaming-only.  Time from request
      issue to the first content-bearing chunk yielded.
    * ``time_per_output_chunk`` — ``gen_ai.client.operation.time_per_output_chunk``
      Histogram, unit ``s``.  Streaming-only.  Inter-chunk gap; one
      observation per content-bearing chunk after the first.
    """

    token_usage: "Histogram"
    operation_duration: "Histogram"
    time_to_first_chunk: "Histogram"
    time_per_output_chunk: "Histogram"


# Bucket boundaries recommended in the OTEL GenAI semantic-conventions
# spec page:
# https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/#generative-ai-client-metrics
_LLM_DURATION_BUCKETS = [
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
]
_LLM_TOKEN_BUCKETS = [
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
]


def _ensure_llm_instruments() -> Optional[LLMInstruments]:
    """Lazily create the LLM-call-scope instruments and return them as
    an :class:`LLMInstruments`.  Returns ``None`` when the OTEL API is
    not installed.

    Bucket boundaries on every histogram are exact matches to the OTEL
    GenAI semantic-conventions spec recommendations:
      https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/
    See :data:`_LLM_DURATION_BUCKETS` and :data:`_LLM_TOKEN_BUCKETS`
    above.
    """
    # Imported lazily to avoid an import cycle: ``guardrails.telemetry``
    # imports from this module, and the meter helper happens to live there.
    from nemoguardrails.guardrails.telemetry import get_meter

    global _llm_instruments
    meter = get_meter()
    if meter is None:
        return None
    if _llm_instruments is None:
        _llm_instruments = LLMInstruments(
            token_usage=meter.create_histogram(
                MetricNames.GEN_AI_CLIENT_TOKEN_USAGE,
                description="Number of input or output tokens used by an LLM call",
                unit="{token}",
                explicit_bucket_boundaries_advisory=_LLM_TOKEN_BUCKETS,
            ),
            operation_duration=meter.create_histogram(
                MetricNames.GEN_AI_CLIENT_OPERATION_DURATION,
                description="End-to-end duration of an LLM call",
                unit="s",
                explicit_bucket_boundaries_advisory=_LLM_DURATION_BUCKETS,
            ),
            time_to_first_chunk=meter.create_histogram(
                MetricNames.GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK,
                description="Time from a streaming LLM request to its first content chunk",
                unit="s",
                explicit_bucket_boundaries_advisory=_LLM_DURATION_BUCKETS,
            ),
            time_per_output_chunk=meter.create_histogram(
                MetricNames.GEN_AI_CLIENT_OPERATION_TIME_PER_OUTPUT_CHUNK,
                description="Inter-chunk interval during a streaming LLM response",
                unit="s",
                explicit_bucket_boundaries_advisory=_LLM_DURATION_BUCKETS,
            ),
        )
    return _llm_instruments


def _llm_call_attributes(
    model_name: str,
    provider_name: str,
    operation_name: str,
) -> dict:
    """Return the standard OTEL GenAI label set shared by every
    ``gen_ai.client.*`` Histogram emission.

    These three are the lowest-cardinality labels the spec mandates as
    Required (``operation.name``, ``provider.name``) or Conditionally
    Required (``request.model``).  Per-metric labels (``token.type``,
    ``error.type``) are added by individual emission helpers.
    """
    return {
        GenAIAttributes.GEN_AI_OPERATION_NAME: operation_name,
        GenAIAttributes.GEN_AI_PROVIDER_NAME: provider_name,
        GenAIAttributes.GEN_AI_REQUEST_MODEL: model_name,
    }


def record_token_usage(
    model_name: str,
    provider_name: str,
    operation_name: str,
    usage: Optional["UsageInfo"],
) -> None:
    """Emit two ``gen_ai.client.token.usage`` observations (one input,
    one output) for a completed LLM call.

    Per spec only ``input`` and ``output`` are valid
    ``gen_ai.token.type`` values — reasoning and cached tokens are
    span-only attributes, not metric labels.

    No-op when ``usage`` is ``None`` (the upstream provider didn't
    return a ``usage`` field — common for streaming when
    ``stream_options.include_usage`` is suppressed) or the OTEL API is
    unavailable.  Skipping emission rather than recording zeros keeps
    the histogram honest: "no observation" is distinct from "0 tokens".
    """
    if usage is None:
        return
    instruments = _ensure_llm_instruments()
    if instruments is None:
        return
    base = _llm_call_attributes(model_name, provider_name, operation_name)
    instruments.token_usage.record(
        usage.input_tokens,
        attributes={**base, GenAIAttributes.GEN_AI_TOKEN_TYPE: TokenType.INPUT},
    )
    instruments.token_usage.record(
        usage.output_tokens,
        attributes={**base, GenAIAttributes.GEN_AI_TOKEN_TYPE: TokenType.OUTPUT},
    )


@contextmanager
def llm_operation_duration(
    model_name: str,
    provider_name: str,
    operation_name: str,
) -> Generator[None, None, None]:
    """Context manager that records the wrapped block's wall-clock
    duration into ``gen_ai.client.operation.duration``.

    On exception, adds the ``error.type`` label (per spec, conditionally
    required on the duration metric only — token usage carries no
    error.type even on failed calls) and re-raises.  No-op when the
    OTEL API is unavailable.
    """
    instruments = _ensure_llm_instruments()
    if instruments is None:
        yield
        return
    base = _llm_call_attributes(model_name, provider_name, operation_name)
    t0 = time.monotonic()
    exc_type: Optional[str] = None
    try:
        yield
    except BaseException as exc:
        exc_type = type(exc).__name__
        raise
    finally:
        elapsed = time.monotonic() - t0
        attrs = base if exc_type is None else {**base, "error.type": exc_type}
        # Best-effort emission: a broken meter SDK must never mask the
        # original exception propagating through ``finally``.
        with suppress(Exception):
            instruments.operation_duration.record(elapsed, attributes=attrs)


def record_time_to_first_chunk(
    model_name: str,
    provider_name: str,
    operation_name: str,
    duration_s: float,
) -> None:
    """Emit a ``gen_ai.client.operation.time_to_first_chunk`` observation.

    Records the elapsed seconds from request issue to the first
    content-bearing chunk yielded by the streaming response.  Caller
    is responsible for the timing — this helper just records the value
    onto the histogram with the standard label set.

    Per OTEL semconv, "first chunk" is the first chunk carrying actual
    output (content or reasoning delta) — not the role-only or other
    cosmetic SSE frames that don't carry data.

    No-op when the OTEL API is unavailable.
    """
    instruments = _ensure_llm_instruments()
    if instruments is None:
        return
    instruments.time_to_first_chunk.record(
        duration_s,
        attributes=_llm_call_attributes(model_name, provider_name, operation_name),
    )


def record_time_per_output_chunk(
    model_name: str,
    provider_name: str,
    operation_name: str,
    duration_s: float,
) -> None:
    """Emit a ``gen_ai.client.operation.time_per_output_chunk`` observation.

    Records the inter-chunk interval for one content-bearing chunk
    after the first.  Each chunk produces one observation; aggregates
    show p50/p95/p99 for chunk-arrival pacing across the stream.

    Caller is responsible for skipping the first chunk (covered by
    ``record_time_to_first_chunk`` instead) and for skipping
    non-content frames (terminal usage chunk, role-only frames) that
    would skew the distribution.

    No-op when the OTEL API is unavailable.
    """
    instruments = _ensure_llm_instruments()
    if instruments is None:
        return
    instruments.time_per_output_chunk.record(
        duration_s,
        attributes=_llm_call_attributes(model_name, provider_name, operation_name),
    )

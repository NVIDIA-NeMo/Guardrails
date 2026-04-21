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
``is_tracing_enabled``, ``get_tracer``, and ``traced_request`` degrade
gracefully (returning ``False``, ``None``, or a no-span passthrough
respectively).  Lower-level helpers like ``request_span`` and
``trace_id_to_request_id`` require OTEL to be available and are only
reachable through ``traced_request`` when a non-``None`` tracer is provided.
"""

import logging
import secrets
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, NamedTuple, Optional, Tuple

from nemoguardrails.guardrails.guardrails_types import (
    REQUEST_ID_BYTES,
    REQUEST_ID_HEX_CHARS,
    RailDirection,
    _set_request_id,
    get_request_id,
    reset_request_id,
    set_new_request_id,
)
from nemoguardrails.tracing.constants import (
    GenAIAttributes,
    GuardrailsAttributes,
    OperationNames,
    SpanNames,
    SystemConstants,
)

log = logging.getLogger(__name__)

_OTEL_AVAILABLE: bool
if TYPE_CHECKING:
    from opentelemetry import trace
    from opentelemetry.trace import Span, SpanKind, StatusCode, Tracer, format_trace_id

    from nemoguardrails.rails.llm.config import TracingConfig

    _OTEL_AVAILABLE = True
else:
    try:
        from opentelemetry import trace
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
    """
    if span is None:
        return
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
        except Exception as exc:
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
        except Exception as exc:
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
        except Exception as exc:
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
        except Exception as exc:
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
        except Exception as exc:
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


@contextmanager
def traced_request(tracer: Optional["Tracer"]) -> Generator[TracedRequest, None, None]:
    """Unified request context: sets request ID, optionally creates a span.

    When *tracer* is not ``None``, a live ``guardrails.request`` SERVER span
    is created and the request ID is derived from its trace ID.  When
    *tracer* is ``None``, a random request ID is generated and the yielded
    span is ``None``.

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
    if tracer is not None:
        with request_span(tracer) as (span, req_id):
            token = _set_request_id(req_id)
            try:
                yield TracedRequest(span, req_id)
            finally:
                _cleanup_request_id(token)
    else:
        token = set_new_request_id()
        try:
            yield TracedRequest(None, get_request_id())
        finally:
            _cleanup_request_id(token)

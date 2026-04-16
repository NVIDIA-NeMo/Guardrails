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
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, Optional, Tuple

from nemoguardrails.guardrails.guardrails_types import (
    get_request_id as _get_request_id,
)
from nemoguardrails.guardrails.guardrails_types import (
    reset_request_id as _reset_request_id,
)
from nemoguardrails.guardrails.guardrails_types import (
    set_new_request_id as _set_new_request_id,
)
from nemoguardrails.guardrails.guardrails_types import (
    set_request_id as _set_request_id,
)
from nemoguardrails.tracing.constants import (
    GenAIAttributes,
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

    Returns the last 16 hex characters of the 128-bit trace ID (the low
    64 bits, which carry the highest entropy).  When the trace ID is zero
    (e.g. a ``NoOpTracerProvider`` is active) a random fallback is used.
    """
    ctx = span.get_span_context()
    if ctx.trace_id == _INVALID_TRACE_ID:
        return secrets.token_hex(8)  # 16 hex chars
    return format_trace_id(ctx.trace_id)[-16:]


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
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
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
        log.warning(
            "Tracing is enabled in config but the opentelemetry-api package is "
            "not installed.  Install it with: pip install nemoguardrails[tracing]"
        )
        return False
    return True


@contextmanager
def traced_request(tracer: Optional["Tracer"]) -> Generator[str, None, None]:
    """Unified request context: sets request ID, optionally creates a span.

    When *tracer* is not ``None``, a live ``guardrails.request`` SERVER span
    is created and the request ID is derived from its trace ID.  When
    *tracer* is ``None``, a random request ID is generated instead.

    Yields ``request_id``.  The request-ID ContextVar is always cleaned up
    on exit.
    """
    if tracer is not None:
        with request_span(tracer) as (_span, req_id):
            token = _set_request_id(req_id)
            try:
                yield req_id
            finally:
                _reset_request_id(token)
    else:
        token = _set_new_request_id()
        try:
            yield _get_request_id()
        finally:
            _reset_request_id(token)

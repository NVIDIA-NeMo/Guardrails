# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator

from nemoguardrails.http._url import sanitize_url, split_url
from nemoguardrails.http.types import HTTPResponse
from nemoguardrails.tracing.constants import HTTPAttributes, _ensure_http_instruments

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

log = logging.getLogger(__name__)


@dataclass(slots=True)
class _HTTPRequestDurationState:
    """Carry response status from request execution to metric recording."""

    response_status_code: int | None = None


def _http_metric_attributes(method: str, url: str) -> dict[str, str | int] | None:
    parts = split_url(url)
    if parts.hostname is None:
        return None
    port = parts.port
    if port is None:
        port = {"http": 80, "https": 443}.get(parts.scheme)
    if port is None:
        return None
    return {
        HTTPAttributes.REQUEST_METHOD: method,
        HTTPAttributes.SERVER_ADDRESS: parts.hostname,
        HTTPAttributes.SERVER_PORT: port,
    }


@contextmanager
def http_request_duration(
    method: str,
    url: str,
) -> Generator[_HTTPRequestDurationState, None, None]:
    """Record one HTTP client request-duration observation.

    The metric contains only low-cardinality endpoint attributes plus response
    status or exception type. Telemetry failures never affect the request.
    """
    state = _HTTPRequestDurationState()
    try:
        instruments = _ensure_http_instruments()
    except Exception:
        instruments = None
    if instruments is None:
        yield state
        return

    started_at = time.monotonic()
    error_type: str | None = None
    try:
        yield state
    except BaseException as error:
        error_type = type(error).__name__
        raise
    finally:
        with suppress(Exception):
            attributes = _http_metric_attributes(method, url)
            if attributes is not None:
                if state.response_status_code is not None:
                    attributes[HTTPAttributes.RESPONSE_STATUS_CODE] = state.response_status_code
                    if state.response_status_code >= 400:
                        error_type = str(state.response_status_code)
                if error_type is not None:
                    attributes[HTTPAttributes.ERROR_TYPE] = error_type
                instruments.request_duration.record(
                    time.monotonic() - started_at,
                    attributes=attributes,
                )


def set_http_request_attributes(
    span: "Span | None",
    method: str,
    url: str,
    content: bytes | str | None,
) -> None:
    """Record privacy-safe HTTP request attributes on a span.

    The URL is stripped of credentials, query parameters, and fragments. Raw
    headers and body content are never recorded.
    """
    if span is None:
        return
    with suppress(Exception):
        parts = split_url(url)
        span.set_attribute(HTTPAttributes.REQUEST_METHOD, method)
        span.set_attribute(HTTPAttributes.URL_FULL, sanitize_url(url))
        if parts.scheme:
            span.set_attribute(HTTPAttributes.URL_SCHEME, parts.scheme)
        if parts.hostname:
            span.set_attribute(HTTPAttributes.SERVER_ADDRESS, parts.hostname)
        with suppress(ValueError):
            if parts.port is not None:
                span.set_attribute(HTTPAttributes.SERVER_PORT, parts.port)
        if content is not None:
            size = len(content) if isinstance(content, bytes) else len(content.encode())
            span.set_attribute(HTTPAttributes.REQUEST_BODY_SIZE, size)


def set_http_response_attributes(span: "Span | None", response: HTTPResponse) -> None:
    """Record response status, body size, and retry count on a span."""
    if span is None:
        return
    with suppress(Exception):
        from opentelemetry.trace import StatusCode

        span.set_attribute(HTTPAttributes.RESPONSE_STATUS_CODE, response.status_code)
        span.set_attribute(HTTPAttributes.RESPONSE_BODY_SIZE, len(response.content))
        retry_count = response.extensions.get("retry_count")
        if isinstance(retry_count, int) and retry_count > 0:
            span.set_attribute(HTTPAttributes.REQUEST_RESEND_COUNT, retry_count)
        if response.status_code >= 400:
            span.set_attribute(HTTPAttributes.ERROR_TYPE, str(response.status_code))
            span.set_status(StatusCode.ERROR)


def record_http_error(span: "Span | None", error: BaseException) -> None:
    """Record an exception type and retry count without exposing its message."""
    if span is None:
        return
    from opentelemetry.trace import StatusCode

    try:
        span.set_attribute(HTTPAttributes.ERROR_TYPE, type(error).__name__)
        retry_count = getattr(error, "retry_count", 0)
        if isinstance(retry_count, int) and retry_count > 0:
            span.set_attribute(HTTPAttributes.REQUEST_RESEND_COUNT, retry_count)
        span.add_event("exception", {HTTPAttributes.EXCEPTION_TYPE: type(error).__name__})
        span.set_status(StatusCode.ERROR)
    except Exception as telemetry_error:
        log.warning(
            "Failed to record HTTP error telemetry: %s",
            type(telemetry_error).__name__,
        )


@contextmanager
def http_call_span(
    tracer: "Tracer | None",
    method: str,
    url: str,
    content: bytes | str | None,
) -> Generator["Span | None", None, None]:
    """Create a privacy-safe client span for one HTTP call.

    Yields ``None`` when no tracer is configured. Request exceptions are
    recorded and re-raised unchanged.
    """
    if tracer is None:
        yield None
        return

    from opentelemetry.trace import SpanKind

    with tracer.start_as_current_span(
        f"HTTP {method}",
        kind=SpanKind.CLIENT,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        set_http_request_attributes(span, method, url, content)
        try:
            yield span
        except BaseException as error:
            record_http_error(span, error)
            raise


__all__ = [
    "http_call_span",
    "http_request_duration",
    "record_http_error",
    "set_http_request_attributes",
    "set_http_response_attributes",
]

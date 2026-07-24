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
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Generator

from nemoguardrails.http._url import sanitize_url, split_url
from nemoguardrails.http.types import HTTPResponse
from nemoguardrails.tracing.constants import HTTPAttributes

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

log = logging.getLogger(__name__)


def set_http_request_attributes(
    span: "Span | None",
    method: str,
    url: str,
    content: bytes | str | None,
) -> None:
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
    "record_http_error",
    "set_http_request_attributes",
    "set_http_response_attributes",
]

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

"""Bind guarded provider streams to an injected HTTP exchange."""

import inspect
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace

from starlette.responses import Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from nemoguardrails.server.experimental._buffered_kernel import InspectionStage, OperationProjectionFailed
from nemoguardrails.server.experimental._content_checker import ContentChecker, StreamBufferingPolicy
from nemoguardrails.server.experimental._guarded_operation import UnsupportedGuardedPayload
from nemoguardrails.server.experimental._guarded_stream import (
    StreamOutcomeRenderer,
    StreamProcessingFailed,
    StreamUpstreamFailed,
    guard_provider_stream,
    validate_streaming_policy,
)
from nemoguardrails.server.experimental._http_kernel import (
    BufferedHttpRequest,
    BufferedHttpResponse,
    HttpDispatchFailed,
    HttpHeaders,
    ResponseBodyTooLarge,
    _end_to_end_headers,
    _has_identity_encoding,
    _has_media_type,
    _render_response,
    _request_identity_encoding,
)
from nemoguardrails.server.experimental._stream_cleanup import StreamClose, close_stream
from nemoguardrails.server.experimental.provider.stream import ProviderStreamAdapter
from nemoguardrails.server.experimental.provider.types import GuardedMessage


@dataclass(frozen=True, slots=True)
class StreamingHttpResponse:
    """Carry one successful response without choosing an HTTP client.

    ``close`` releases the upstream response/connection independently of body
    iteration. An ordinary self-owning iterator may instead expose ``aclose``;
    a bare async generator must supply an independent resource closer. Never
    rely solely on its finally block: cancellation can interrupt it permanently.

    Dispatch owns resources until it returns this value and must clean up if
    acquisition or construction fails. The executor then owns cleanup through
    rejection or response completion, including disconnect before iteration.
    Closing must be idempotent: iterator and resource cleanup may be retried.
    Each attempt is shielded and bounded; failures are logged, not substituted
    for the primary outcome. Cancellation-suppressing closers cannot be forced.
    """

    status_code: int
    headers: HttpHeaders
    body: AsyncIterator[bytes]
    close: StreamClose | None = None

    def __post_init__(self) -> None:
        if type(self.status_code) is not int or not 200 <= self.status_code <= 299:
            raise ValueError("A streaming HTTP response must have a successful status code.")
        if not hasattr(self.body, "__aiter__"):
            raise TypeError("A streaming HTTP response body must be an async iterator.")
        if any(not isinstance(name, bytes) or not isinstance(value, bytes) for name, value in self.headers):
            raise TypeError("Streaming HTTP response headers must contain byte pairs.")
        if self.close is None:
            if inspect.isasyncgen(self.body) or not callable(getattr(self.body, "aclose", None)):
                raise TypeError("A streaming body needs an independent resource close callback")
        elif not callable(self.close):
            raise TypeError("The streaming resource close callback must be callable")


StreamingHttpDispatch = Callable[[BufferedHttpRequest], Awaitable[StreamingHttpResponse | BufferedHttpResponse]]


class _ClosingStreamingResponse(StreamingResponse):
    """Close the provider body however the response ends, even before iteration."""

    def __init__(
        self,
        body: AsyncIterator[bytes],
        source: AsyncIterator[bytes],
        *,
        status_code: int,
        close: StreamClose | None = None,
    ):
        super().__init__(body, status_code=status_code)
        self.owned_body = body
        self.source = source
        self.close = close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await close_stream(self.owned_body)
            finally:
                await close_stream(self.source, close=self.close)


_BODY_DEPENDENT_HEADERS = frozenset(
    {b"content-length", b"content-md5", b"content-digest", b"digest", b"etag", b"repr-digest"}
)


def _streaming_response(
    value: StreamingHttpResponse,
    body: AsyncIterator[bytes],
    *,
    may_modify: bool,
) -> StreamingResponse:
    response = _ClosingStreamingResponse(body, value.body, status_code=value.status_code, close=value.close)
    response.raw_headers = [
        (name, header_value)
        for name, header_value in _end_to_end_headers(value.headers)
        if not may_modify or name.lower() not in _BODY_DEPENDENT_HEADERS
    ]
    return response


def _header_values(headers: HttpHeaders, name: bytes) -> tuple[bytes, ...]:
    return tuple(value for header_name, value in headers if header_name.lower() == name)


async def _relay(
    source: AsyncIterator[bytes], expected_length: bytes | None, *, close: StreamClose | None
) -> AsyncIterator[bytes]:
    observed = 0
    try:
        async for chunk in source:
            if not isinstance(chunk, bytes):
                raise TypeError("A provider stream must yield bytes.")
            observed += len(chunk)
            if expected_length is not None:
                digits = str(observed).encode()
                if len(digits) > len(expected_length) or (
                    len(digits) == len(expected_length) and digits > expected_length
                ):
                    raise ValueError("The provider stream exceeds its declared Content-Length.")
            yield chunk
        if expected_length is not None and str(observed).encode() != expected_length:
            raise ValueError("The provider stream does not match its declared Content-Length.")
    finally:
        await close_stream(source, close=close)


async def execute_streaming_http(
    request: BufferedHttpRequest,
    *,
    dispatch: StreamingHttpDispatch,
    checker: ContentChecker,
    streaming_policy: StreamBufferingPolicy | None,
    input_message: GuardedMessage,
    adapter: ProviderStreamAdapter,
    render_outcome: StreamOutcomeRenderer,
    max_event_bytes: int,
    max_pending_bytes: int,
) -> Response:
    """Dispatch one prepared stream and preserve its HTTP lifecycle."""

    if any(type(limit) is not int or limit <= 0 for limit in (max_event_bytes, max_pending_bytes)):
        raise ValueError("Streaming HTTP byte limits must be positive.")
    if input_message.role != "user":
        raise ValueError("A guarded stream requires a user input message.")
    if streaming_policy is not None:
        validate_streaming_policy(streaming_policy)
        # An encoded stream cannot be inspected; uninspected relays keep the client's preference.
        request = replace(request, headers=_request_identity_encoding(request.headers))

    try:
        upstream = await dispatch(request)
    except HttpDispatchFailed as failure:
        # Match the buffered path: only a typed dispatch failure is an upstream
        # outage. Other exceptions are bugs and propagate.
        return _render_response(render_outcome(StreamUpstreamFailed(failure)), request.method)

    if isinstance(upstream, BufferedHttpResponse):
        if 200 <= upstream.status_code <= 299:
            return _render_response(
                render_outcome(
                    OperationProjectionFailed(
                        InspectionStage.OUTPUT,
                        UnsupportedGuardedPayload("A successful streaming response must have an open body."),
                    )
                ),
                request.method,
            )
        if len(upstream.body) > max_pending_bytes:
            return _render_response(render_outcome(StreamUpstreamFailed(ResponseBodyTooLarge())), request.method)
        return _render_response(upstream, request.method)
    if not isinstance(upstream, StreamingHttpResponse):
        return _render_response(
            render_outcome(StreamProcessingFailed(TypeError("Unsupported streaming dispatch result."))), request.method
        )
    headers = tuple(_end_to_end_headers(upstream.headers))
    content_lengths = _header_values(headers, b"content-length")
    if (
        request.method == "HEAD"
        or upstream.status_code in {204, 205}
        or not _has_media_type(headers, b"text/event-stream")
        or len(content_lengths) > 1
        or any(re.fullmatch(rb"[0-9]+", length) is None for length in content_lengths)
    ):
        await close_stream(upstream.body, close=upstream.close)
        return _render_response(
            render_outcome(
                OperationProjectionFailed(
                    InspectionStage.OUTPUT,
                    UnsupportedGuardedPayload("A successful provider stream requires valid SSE body framing."),
                )
            ),
            request.method,
        )
    if streaming_policy is not None and not _has_identity_encoding(upstream.headers):
        await close_stream(upstream.body, close=upstream.close)
        return _render_response(
            render_outcome(
                OperationProjectionFailed(
                    InspectionStage.OUTPUT,
                    UnsupportedGuardedPayload("An inspected provider stream must use identity content encoding."),
                )
            ),
            request.method,
        )
    if streaming_policy is None:
        expected_length = (content_lengths[0].lstrip(b"0") or b"0") if content_lengths else None
        return _streaming_response(
            upstream, _relay(upstream.body, expected_length, close=upstream.close), may_modify=False
        )

    body = guard_provider_stream(
        upstream.body,
        checker=checker,
        streaming_policy=streaming_policy,
        input_message=input_message,
        adapter=adapter,
        render_outcome=render_outcome,
        max_event_bytes=max_event_bytes,
        max_pending_bytes=max_pending_bytes,
        close=upstream.close,
    )
    return _streaming_response(upstream, body, may_modify=True)

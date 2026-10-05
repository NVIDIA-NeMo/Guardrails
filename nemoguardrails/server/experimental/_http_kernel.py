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

"""Create transparent HTTP proxy routes for buffered provider operations."""

import re
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, cast
from urllib.parse import quote

from fastapi import APIRouter, Request
from starlette.convertors import FloatConvertor, IntegerConvertor, StringConvertor, UUIDConvertor
from starlette.responses import Response
from starlette.routing import compile_path

from nemoguardrails.server.experimental._buffered_kernel import (
    InspectionStage,
    OperationBlocked,
    OperationCheckFailed,
    OperationCompleted,
    OperationModificationUnsupported,
    OperationProjectionFailed,
    execute_buffered_operation,
)
from nemoguardrails.server.experimental._content_checker import (
    ContentChecker,
    _ResolvedContentChecker,
    validate_content_checker,
)
from nemoguardrails.server.experimental._guarded_operation import BufferedGuardedOperation, UnsupportedGuardedPayload
from nemoguardrails.server.experimental._http_paths import HTTP_METHODS, GuardedOperationPath

DEFAULT_MAX_REQUEST_BODY_BYTES = 1024 * 1024
DEFAULT_MAX_RESPONSE_BODY_BYTES = 10 * 1024 * 1024
HttpHeaders = tuple[tuple[bytes, bytes], ...]
_HOP_BY_HOP_HEADERS = frozenset(
    {
        b"connection",
        b"keep-alive",
        b"proxy-authenticate",
        b"proxy-authorization",
        b"proxy-connection",
        b"te",
        b"trailer",
        b"transfer-encoding",
        b"upgrade",
    }
)


class RequestBodyTooLarge(Exception):
    """Report a buffered downstream body beyond the configured limit."""


class ResponseBodyTooLarge(Exception):
    """Report a buffered upstream body beyond the configured limit."""


class InvalidContentLength(ValueError):
    """Report a malformed or negative request Content-Length header."""


class HttpDispatchFailed(Exception):
    """Report a typed outbound dispatch failure."""


class InvalidRequestPath(ValueError):
    """Reject paths that a sender could reinterpret as a different destination."""


class HttpRouteRejectionKind(str, Enum):
    """Classify proxy-owned route rejections before provider dispatch."""

    METHOD_NOT_ALLOWED = "method_not_allowed"
    NON_CANONICAL_PATH = "non_canonical_path"
    INVALID_REQUEST_PATH = "invalid_request_path"


@dataclass(frozen=True, slots=True)
class HttpRouteRejected:
    """Carry a route rejection and its allowed methods to the response mapping."""

    kind: HttpRouteRejectionKind
    allowed_methods: frozenset[str] = frozenset()


class HttpFailureKind(str, Enum):
    """Classify failures while buffering or dispatching HTTP requests."""

    INVALID_CONTENT_LENGTH = "invalid_content_length"
    REQUEST_BODY_TOO_LARGE = "request_body_too_large"
    UPSTREAM_REQUEST_FAILED = "upstream_request_failed"
    RESPONSE_BODY_TOO_LARGE = "response_body_too_large"


@dataclass(frozen=True, slots=True)
class HttpOperationFailed:
    """Carry one HTTP failure to the configured response renderer."""

    kind: HttpFailureKind
    failure: BaseException


@dataclass(frozen=True, slots=True)
class BufferedHttpRequest:
    """Preserve one fully buffered downstream HTTP request."""

    method: str
    path: str
    raw_path: bytes
    query: bytes
    headers: HttpHeaders
    body: bytes


@dataclass(frozen=True, slots=True)
class BufferedHttpResponse:
    """Preserve one fully buffered upstream HTTP response."""

    status_code: int
    headers: HttpHeaders
    body: bytes

    def __post_init__(self) -> None:
        """Validate status, headers, and body before rendering a response."""

        if not isinstance(self.status_code, int) or not 100 <= self.status_code <= 599:
            raise ValueError("An HTTP response status code must be between 100 and 599.")
        if not isinstance(self.body, bytes):
            raise TypeError("An HTTP response body must be bytes.")
        if any(not isinstance(name, bytes) or not isinstance(value, bytes) for name, value in self.headers):
            raise TypeError("HTTP response headers must contain byte pairs.")


@dataclass(frozen=True, slots=True)
class GuardedHttpOperation:
    """Bind one buffered guarded operation to an owned HTTP path."""

    operation_path: GuardedOperationPath
    operation: BufferedGuardedOperation[Any, BufferedHttpResponse]
    prepare_request: Callable[[BufferedHttpRequest], Any] | None = None
    forward_request: Callable[[Any], BufferedHttpRequest] | None = None
    documented_responses: dict[int | str, dict[str, Any]] | None = None
    guarded_operation_paths: tuple[GuardedOperationPath, ...] = ()
    openapi_extra: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate adapters and derive the default guarded route set."""

        if (self.prepare_request is None) != (self.forward_request is None):
            raise ValueError("Guarded HTTP request preparation and forwarding adapters must be declared together.")
        if not self.guarded_operation_paths:
            object.__setattr__(self, "guarded_operation_paths", (self.operation_path,))


HttpDispatch = Callable[[BufferedHttpRequest], Awaitable[BufferedHttpResponse]]
OutcomeRenderer = Callable[
    [
        OperationBlocked
        | OperationCheckFailed
        | OperationModificationUnsupported
        | OperationProjectionFailed
        | HttpOperationFailed
        | HttpRouteRejected
    ],
    BufferedHttpResponse,
]


def _route_shape(path: str) -> str:
    """Normalize route parameter names for duplicate detection."""

    _, path_format, convertors = compile_path(path)
    for name, convertor in convertors.items():
        path_format = path_format.replace(f"{{{name}}}", f"{{{type(convertor).__name__}}}")
    return path_format


def _route_boundaries_overlap(left: str, right: str) -> bool:
    """Compare the fixed prefix and suffix of two compiled route formats."""

    left_prefix, right_prefix = left.split("{", 1)[0], right.split("{", 1)[0]
    if not (left_prefix.startswith(right_prefix) or right_prefix.startswith(left_prefix)):
        return False
    left_suffix, right_suffix = left.rsplit("}", 1)[-1], right.rsplit("}", 1)[-1]
    return left_suffix.endswith(right_suffix) or right_suffix.endswith(left_suffix)


def _route_segments_overlap(left: str, right: str) -> bool:
    """Return whether two segments may overlap."""

    left_regex, left_format, left_convertors = compile_path(f"/{left}")
    right_regex, right_format, right_convertors = compile_path(f"/{right}")
    if not left_convertors:
        return right_regex.fullmatch(f"/{left}") is not None
    if not right_convertors:
        return left_regex.fullmatch(f"/{right}") is not None
    return _route_boundaries_overlap(left_format, right_format)


def _routes_overlap(left: str, right: str) -> bool:
    """Conservatively reject routes unless their segments prove disjoint.

    Custom converters may span segments, so only whole-route literal boundaries
    can prove two dynamic routes using them disjoint.
    """

    left_regex, left_format, left_convertors = compile_path(left)
    right_regex, right_format, right_convertors = compile_path(right)
    if not left_convertors:
        return right_regex.fullmatch(left) is not None
    if not right_convertors:
        return left_regex.fullmatch(right) is not None
    segment_convertors = (StringConvertor, IntegerConvertor, FloatConvertor, UUIDConvertor)
    if any(
        type(convertor) not in segment_convertors
        for convertor in (*left_convertors.values(), *right_convertors.values())
    ):
        return _route_boundaries_overlap(left_format, right_format)
    left_segments, right_segments = left.split("/"), right.split("/")
    return len(left_segments) == len(right_segments) and all(
        _route_segments_overlap(left_segment, right_segment)
        for left_segment, right_segment in zip(left_segments, right_segments)
    )


def _validate_operations(
    operations: Collection[GuardedHttpOperation],
) -> tuple[GuardedHttpOperation, ...]:
    """Require at least one operation with unique names and routes."""

    resolved = tuple(operations)
    if not resolved:
        raise ValueError("At least one guarded HTTP operation is required.")
    names = [operation.operation.name for operation in resolved]
    if len(names) != len(set(names)):
        raise ValueError("Guarded operation names must be unique.")
    routes = [
        (method, _route_shape(operation.operation_path.route_path))
        for operation in resolved
        for method in operation.operation_path.methods
    ]
    if len(routes) != len(set(routes)):
        raise ValueError("Guarded operation routes must be unique.")
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if left.operation_path.methods & right.operation_path.methods and _routes_overlap(
                left.operation_path.route_path,
                right.operation_path.route_path,
            ):
                raise ValueError("Guarded operation routes must not overlap for the same method.")
    return resolved


def _request_path(request: Request) -> bytes:
    """Return the raw request path or reconstruct its ASCII encoding."""

    raw_path = request.scope.get("raw_path")
    if isinstance(raw_path, bytes):
        return raw_path
    return quote(request.scope["path"], safe="/").encode("ascii")


def _normalized_route_path(path: str) -> str:
    """Normalize ownership checks without changing the forwarded path."""

    segments = []
    for segment in path.split("/"):
        if not segment or segment == ".":
            continue
        if segment == "..":
            if segments:
                segments.pop()
        else:
            segments.append(segment)
    return "/" + "/".join(segments)


def _route_path_candidates(path: str) -> frozenset[str]:
    """Account for decoded delimiters being reparsed by an injected sender."""

    return frozenset(_normalized_route_path(candidate) for candidate in (path, re.split(r"[?#]", path, maxsplit=1)[0]))


def _validate_request_path(request: Request) -> None:
    """Reject network references, backslashes, and path control characters."""

    path = request.scope["path"]
    raw_path = _request_path(request)
    if (
        not path.startswith("/")
        or path.startswith("//")
        or "\\" in path
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
        or not raw_path.startswith(b"/")
        or raw_path.startswith(b"//")
        or any(character < 32 or character == 127 for character in raw_path)
        or any(character in raw_path for character in (b"\\", b"?", b"#"))
    ):
        raise InvalidRequestPath("The request must contain an unambiguous absolute path.")


def _validate_reserved_routes(
    routes: Mapping[str, Collection[str]],
) -> tuple[tuple[re.Pattern[str], frozenset[str]], ...]:
    """Validate application route reservations, including root and trailing slash routes."""

    resolved = []
    for path, methods in routes.items():
        if (
            not isinstance(path, str)
            or not path.startswith("/")
            or "//" in path
            or any(character in path for character in ("\\", "?", "#"))
            or any(segment in {".", ".."} for segment in path.split("/"))
        ):
            raise ValueError("A reserved HTTP path must be an unambiguous absolute route template.")
        try:
            path_regex, _, _ = compile_path(path.rstrip("/") or "/")
        except (AssertionError, KeyError, ValueError) as error:
            raise ValueError("A reserved HTTP path must be a valid route template.") from error
        if isinstance(methods, (str, bytes)) or not methods or any(method not in HTTP_METHODS for method in methods):
            raise ValueError("Reserved HTTP methods must be supported uppercase methods.")
        resolved.append((path_regex, frozenset(methods)))
    return tuple(resolved)


async def _buffer_request(request: Request, max_body_bytes: int) -> BufferedHttpRequest:
    """Read and preserve one request within the configured body limit."""

    _validate_request_path(request)
    content_lengths = request.headers.getlist("content-length")
    parsed_content_length = None
    if len(content_lengths) > 1:
        raise InvalidContentLength("Content-Length must not be repeated.")
    if content_lengths:
        if request.headers.getlist("transfer-encoding"):
            raise InvalidContentLength("Content-Length must not accompany Transfer-Encoding.")
        if re.fullmatch(r"[0-9]+", content_lengths[0]) is None:
            raise InvalidContentLength("Content-Length must contain only decimal digits.")
        digits = content_lengths[0].lstrip("0") or "0"
        limit = str(max_body_bytes)
        if len(digits) > len(limit) or (len(digits) == len(limit) and digits > limit):
            raise RequestBodyTooLarge
        parsed_content_length = int(digits)
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_body_bytes:
            raise RequestBodyTooLarge
        body.extend(chunk)
    if parsed_content_length is not None and parsed_content_length != len(body):
        raise InvalidContentLength("Content-Length must match the buffered request body.")
    return BufferedHttpRequest(
        method=request.method,
        path=request.scope["path"],
        raw_path=_request_path(request),
        query=request.scope.get("query_string", b""),
        headers=tuple(request.scope.get("headers", ())),
        body=bytes(body),
    )


def _end_to_end_headers(headers: HttpHeaders) -> list[tuple[bytes, bytes]]:
    """Remove connection-specific metadata from buffered and streamed responses."""

    connection_tokens = {
        token.strip().lower()
        for name, content in headers
        if name.lower() == b"connection"
        for token in content.split(b",")
    }
    excluded = _HOP_BY_HOP_HEADERS | connection_tokens
    return [(name, content) for name, content in headers if name.lower() not in excluded]


def _render_response(value: BufferedHttpResponse, request_method: str | None = None) -> Response:
    """Preserve end-to-end values while generating safe downstream framing."""

    headers = [
        (name, content) for name, content in _end_to_end_headers(value.headers) if name.lower() != b"content-length"
    ]
    bodyless = request_method == "HEAD" or value.status_code < 200 or value.status_code in {204, 205, 304}
    response = Response(content=b"" if bodyless else value.body, status_code=value.status_code)
    if value.status_code == 205:
        headers.append((b"content-length", b"0"))
    elif value.status_code == 304 or (
        request_method == "HEAD" and value.status_code >= 200 and value.status_code != 204
    ):
        lengths = [content for name, content in value.headers if name.lower() == b"content-length"]
        if len(lengths) == 1 and re.fullmatch(rb"[0-9]+", lengths[0]) is not None:
            headers.append((b"content-length", lengths[0]))
    elif request_method != "HEAD":
        headers.extend(response.raw_headers)
    response.raw_headers = headers
    return response


def _request_identity_encoding(headers: HttpHeaders) -> HttpHeaders:
    """Ask the provider for an unencoded response that output checks can read.

    Clients such as the OpenAI SDK accept gzip, and an encoded successful
    response cannot be inspected. Without output checks the response is relayed
    unchanged, so the client's own preference is kept.
    """
    return (
        *((name, value) for name, value in headers if name.lower() != b"accept-encoding"),
        (b"accept-encoding", b"identity"),
    )


def _render_failure(
    failure: OperationBlocked
    | OperationCheckFailed
    | OperationModificationUnsupported
    | OperationProjectionFailed
    | HttpOperationFailed
    | HttpRouteRejected,
    render_outcome: OutcomeRenderer,
    request_method: str | None = None,
) -> Response:
    """Render one operation failure through the configured response mapping."""

    response = _render_response(render_outcome(failure), request_method)
    if isinstance(failure, HttpRouteRejected) and failure.allowed_methods:
        response.headers["allow"] = ", ".join(sorted(failure.allowed_methods))
    return response


def _guarded_handler(
    declaration: GuardedHttpOperation,
    checker: _ResolvedContentChecker,
    dispatch: HttpDispatch,
    render_outcome: OutcomeRenderer,
    max_request_body_bytes: int,
    max_response_body_bytes: int,
) -> Callable[[Request], Awaitable[Response]]:
    """Create the HTTP handler for one guarded operation."""

    async def bounded_dispatch(request: Any) -> BufferedHttpResponse:
        """Dispatch one request and enforce the response body limit."""

        forwarded = (
            declaration.forward_request(request)
            if declaration.forward_request is not None
            else cast(BufferedHttpRequest, request)
        )
        if checker.policy.inspect_output:
            forwarded = replace(forwarded, headers=_request_identity_encoding(forwarded.headers))
        response = await dispatch(forwarded)
        if len(response.body) > max_response_body_bytes:
            raise ResponseBodyTooLarge
        return response

    async def handle(request: Request) -> Response:
        """Buffer, check, dispatch, and render one guarded request."""

        if _normalized_route_path(request.scope["path"]) != request.scope["path"]:
            return _render_failure(
                HttpRouteRejected(HttpRouteRejectionKind.NON_CANONICAL_PATH), render_outcome, request.method
            )
        try:
            buffered_request = await _buffer_request(request, max_request_body_bytes)
            try:
                operation_request = (
                    declaration.prepare_request(buffered_request)
                    if declaration.prepare_request is not None
                    else buffered_request
                )
            except UnsupportedGuardedPayload as failure:
                return _render_failure(
                    OperationProjectionFailed(InspectionStage.INPUT, failure),
                    render_outcome,
                )
            outcome = await execute_buffered_operation(
                declaration.operation,
                checker,
                operation_request,
                bounded_dispatch,
            )
        except RequestBodyTooLarge as failure:
            return _render_failure(
                HttpOperationFailed(HttpFailureKind.REQUEST_BODY_TOO_LARGE, failure),
                render_outcome,
                request.method,
            )
        except InvalidContentLength as failure:
            return _render_failure(
                HttpOperationFailed(HttpFailureKind.INVALID_CONTENT_LENGTH, failure),
                render_outcome,
                request.method,
            )
        except InvalidRequestPath:
            return _render_failure(
                HttpRouteRejected(HttpRouteRejectionKind.INVALID_REQUEST_PATH), render_outcome, request.method
            )
        except HttpDispatchFailed as failure:
            return _render_failure(
                HttpOperationFailed(HttpFailureKind.UPSTREAM_REQUEST_FAILED, failure),
                render_outcome,
                request.method,
            )
        except ResponseBodyTooLarge as failure:
            return _render_failure(
                HttpOperationFailed(HttpFailureKind.RESPONSE_BODY_TOO_LARGE, failure),
                render_outcome,
                request.method,
            )
        if isinstance(outcome, OperationCompleted):
            return _render_response(outcome.response, request.method)
        return _render_failure(outcome, render_outcome, request.method)

    return handle


def create_http_proxy_router(
    *,
    operations: Collection[GuardedHttpOperation],
    checker: ContentChecker,
    dispatch: HttpDispatch,
    render_outcome: OutcomeRenderer,
    reserved_routes: Mapping[str, Collection[str]] | None = None,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    max_response_body_bytes: int = DEFAULT_MAX_RESPONSE_BODY_BYTES,
) -> APIRouter:
    """Create guarded routes followed by a transparent provider catch-all.

    Application-owned routes must be registered before this router because its
    catch-all owns every path not handled by an earlier route.
    """

    resolved_operations = _validate_operations(operations)
    if any(type(limit) is not int or limit <= 0 for limit in (max_request_body_bytes, max_response_body_bytes)):
        raise ValueError("Buffered HTTP body limits must be positive integers.")
    validated_checker = validate_content_checker(checker)
    guarded_matchers = []
    reserved_matchers = _validate_reserved_routes(reserved_routes or {})
    router = APIRouter()

    for declaration in resolved_operations:
        guarded_matchers.extend(declaration.guarded_operation_paths)
        router.add_api_route(
            declaration.operation_path.route_path,
            _guarded_handler(
                declaration,
                validated_checker,
                dispatch,
                render_outcome,
                max_request_body_bytes,
                max_response_body_bytes,
            ),
            methods=sorted(declaration.operation_path.methods),
            name=declaration.operation.name,
            operation_id=declaration.operation.name,
            response_class=Response,
            responses=declaration.documented_responses,
            openapi_extra=declaration.openapi_extra,
        )

    @router.api_route("/{path:path}", methods=list(HTTP_METHODS), include_in_schema=False)
    async def passthrough(request: Request, path: str) -> Response:
        """Forward provider-owned routes without content checking."""

        try:
            _validate_request_path(request)
        except InvalidRequestPath:
            return _render_failure(
                HttpRouteRejected(HttpRouteRejectionKind.INVALID_REQUEST_PATH), render_outcome, request.method
            )
        candidates = _route_path_candidates(f"/{path}")
        reserved_methods = next(
            (
                methods
                for path_regex, methods in reserved_matchers
                if any(path_regex.fullmatch(candidate) is not None for candidate in candidates)
            ),
            None,
        )
        if reserved_methods is not None:
            return _render_failure(
                HttpRouteRejected(HttpRouteRejectionKind.METHOD_NOT_ALLOWED, reserved_methods),
                render_outcome,
                request.method,
            )
        matched_methods = {
            method
            for operation_path in guarded_matchers
            if any(operation_path.matches(candidate) for candidate in candidates)
            for method in operation_path.methods
        }
        if matched_methods:
            if request.method not in matched_methods:
                return _render_failure(
                    HttpRouteRejected(HttpRouteRejectionKind.METHOD_NOT_ALLOWED, frozenset(matched_methods)),
                    render_outcome,
                    request.method,
                )
            return _render_failure(
                HttpRouteRejected(HttpRouteRejectionKind.NON_CANONICAL_PATH), render_outcome, request.method
            )
        try:
            buffered_request = await _buffer_request(request, max_request_body_bytes)
            response = await dispatch(buffered_request)
        except RequestBodyTooLarge as failure:
            return _render_failure(
                HttpOperationFailed(HttpFailureKind.REQUEST_BODY_TOO_LARGE, failure),
                render_outcome,
                request.method,
            )
        except InvalidContentLength as failure:
            return _render_failure(
                HttpOperationFailed(HttpFailureKind.INVALID_CONTENT_LENGTH, failure),
                render_outcome,
                request.method,
            )
        except HttpDispatchFailed as failure:
            return _render_failure(
                HttpOperationFailed(HttpFailureKind.UPSTREAM_REQUEST_FAILED, failure),
                render_outcome,
                request.method,
            )
        if len(response.body) > max_response_body_bytes:
            return _render_failure(
                HttpOperationFailed(HttpFailureKind.RESPONSE_BODY_TOO_LARGE, ResponseBodyTooLarge()),
                render_outcome,
                request.method,
            )
        return _render_response(response, request.method)

    return router

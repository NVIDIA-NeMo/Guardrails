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

"""Bind the private buffered kernel to a transparent HTTP boundary."""

import re
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass
from enum import Enum
from urllib.parse import quote

from fastapi import APIRouter, Request, status
from starlette.convertors import PathConvertor
from starlette.responses import Response
from starlette.routing import compile_path

from nemoguardrails.server.experimental._buffered_kernel import (
    OperationBlocked,
    OperationCheckFailed,
    OperationCompleted,
    OperationModificationUnsupported,
    execute_buffered_operation,
)
from nemoguardrails.server.experimental._content_checker import (
    ContentChecker,
    _ResolvedContentChecker,
    validate_content_checker,
)
from nemoguardrails.server.experimental._guarded_operation import BufferedGuardedOperation

HTTP_METHODS = ("DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT")
DEFAULT_MAX_REQUEST_BODY_BYTES = 1024 * 1024
DEFAULT_MAX_RESPONSE_BODY_BYTES = 10 * 1024 * 1024
HttpHeaders = tuple[tuple[bytes, bytes], ...]


class RequestBodyTooLarge(Exception):
    """Report a buffered downstream body beyond the configured limit."""


class ResponseBodyTooLarge(Exception):
    """Report a buffered upstream body beyond the configured limit."""


class InvalidContentLength(ValueError):
    """Report a malformed or negative request Content-Length header."""


class HttpDispatchFailed(Exception):
    """Report a typed outbound dispatch failure."""


class HttpFailureKind(str, Enum):
    """Classify provider-neutral HTTP boundary failures."""

    INVALID_CONTENT_LENGTH = "invalid_content_length"
    REQUEST_BODY_TOO_LARGE = "request_body_too_large"
    UPSTREAM_REQUEST_FAILED = "upstream_request_failed"
    RESPONSE_BODY_TOO_LARGE = "response_body_too_large"


@dataclass(frozen=True, slots=True)
class HttpOperationFailed:
    """Carry one HTTP boundary failure to the configured renderer."""

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
        if not isinstance(self.status_code, int) or not 100 <= self.status_code <= 599:
            raise ValueError("An HTTP response status code must be between 100 and 599.")
        if not isinstance(self.body, bytes):
            raise TypeError("An HTTP response body must be bytes.")
        if any(not isinstance(name, bytes) or not isinstance(value, bytes) for name, value in self.headers):
            raise TypeError("HTTP response headers must contain byte pairs.")


@dataclass(frozen=True, slots=True)
class GuardedOperationPath:
    """Describe a path shape owned by one guarded provider operation."""

    route_path: str
    methods: frozenset[str] = frozenset({"POST"})

    def __post_init__(self) -> None:
        if not self.route_path.startswith("/") or self.route_path == "/" or self.route_path.endswith("/"):
            raise ValueError("A guarded HTTP path must be absolute, non-root, and have no trailing slash.")
        try:
            _, _, convertors = compile_path(self.route_path)
        except (AssertionError, KeyError, ValueError) as error:
            raise ValueError("A guarded HTTP path must be a valid route template.") from error
        if any(isinstance(convertor, PathConvertor) for convertor in convertors.values()):
            raise ValueError("A guarded HTTP path must not contain a path-spanning parameter.")
        if not self.methods or any(method not in HTTP_METHODS for method in self.methods):
            raise ValueError("Guarded HTTP methods must be supported uppercase methods.")

    def matches(self, path: str) -> bool:
        """Return whether a concrete path belongs to this operation."""

        path_regex, _, _ = compile_path(self.route_path)
        return path_regex.fullmatch(path) is not None


@dataclass(frozen=True, slots=True)
class GuardedHttpOperation:
    """Bind one buffered guarded operation to an owned HTTP path."""

    operation_path: GuardedOperationPath
    operation: BufferedGuardedOperation[BufferedHttpRequest, BufferedHttpResponse]


HttpDispatch = Callable[[BufferedHttpRequest], Awaitable[BufferedHttpResponse]]
OutcomeRenderer = Callable[
    [OperationBlocked | OperationCheckFailed | OperationModificationUnsupported | HttpOperationFailed],
    BufferedHttpResponse,
]


def _route_shape(path: str) -> str:
    _, path_format, convertors = compile_path(path)
    for name, convertor in convertors.items():
        path_format = path_format.replace(f"{{{name}}}", f"{{{type(convertor).__name__}}}")
    return path_format


def _validate_operations(operations: Collection[GuardedHttpOperation]) -> tuple[GuardedHttpOperation, ...]:
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
    return resolved


def _request_path(request: Request) -> bytes:
    raw_path = request.scope.get("raw_path")
    if isinstance(raw_path, bytes):
        return raw_path
    return quote(request.url.path, safe="/").encode("ascii")


async def _buffer_request(request: Request, max_body_bytes: int) -> BufferedHttpRequest:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            parsed_content_length = int(content_length)
        except ValueError as error:
            raise InvalidContentLength("Content-Length must be an integer.") from error
        if parsed_content_length < 0:
            raise InvalidContentLength("Content-Length must not be negative.")
        if parsed_content_length > max_body_bytes:
            raise RequestBodyTooLarge
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_body_bytes:
            raise RequestBodyTooLarge
        body.extend(chunk)
    return BufferedHttpRequest(
        method=request.method,
        path=request.url.path,
        raw_path=_request_path(request),
        query=request.scope.get("query_string", b""),
        headers=tuple(request.scope.get("headers", ())),
        body=bytes(body),
    )


def _render_response(value: BufferedHttpResponse) -> Response:
    response = Response(content=value.body, status_code=value.status_code)
    response.raw_headers = list(value.headers)
    return response


def _render_failure(
    failure: OperationBlocked | OperationCheckFailed | OperationModificationUnsupported | HttpOperationFailed,
    render_outcome: OutcomeRenderer,
) -> Response:
    return _render_response(render_outcome(failure))


def _guarded_handler(
    declaration: GuardedHttpOperation,
    checker: _ResolvedContentChecker,
    dispatch: HttpDispatch,
    render_outcome: OutcomeRenderer,
    max_request_body_bytes: int,
    max_response_body_bytes: int,
) -> Callable[[Request], Awaitable[Response]]:
    async def bounded_dispatch(request: BufferedHttpRequest) -> BufferedHttpResponse:
        response = await dispatch(request)
        if len(response.body) > max_response_body_bytes:
            raise ResponseBodyTooLarge
        return response

    async def handle(request: Request) -> Response:
        try:
            buffered_request = await _buffer_request(request, max_request_body_bytes)
            outcome = await execute_buffered_operation(
                declaration.operation,
                checker,
                buffered_request,
                bounded_dispatch,
            )
        except RequestBodyTooLarge as failure:
            return _render_failure(
                HttpOperationFailed(HttpFailureKind.REQUEST_BODY_TOO_LARGE, failure),
                render_outcome,
            )
        except InvalidContentLength as failure:
            return _render_failure(
                HttpOperationFailed(HttpFailureKind.INVALID_CONTENT_LENGTH, failure),
                render_outcome,
            )
        except HttpDispatchFailed as failure:
            return _render_failure(
                HttpOperationFailed(HttpFailureKind.UPSTREAM_REQUEST_FAILED, failure),
                render_outcome,
            )
        except ResponseBodyTooLarge as failure:
            return _render_failure(
                HttpOperationFailed(HttpFailureKind.RESPONSE_BODY_TOO_LARGE, failure),
                render_outcome,
            )
        if isinstance(outcome, OperationCompleted):
            return _render_response(outcome.response)
        return _render_failure(outcome, render_outcome)

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
    """Create guarded routes followed by a transparent provider catch-all."""

    resolved_operations = _validate_operations(operations)
    if max_request_body_bytes <= 0 or max_response_body_bytes <= 0:
        raise ValueError("Buffered HTTP body limits must be positive.")
    validated_checker = validate_content_checker(checker)
    guarded_matchers = []
    reserved_matchers = tuple(
        (compile_path(path)[0], frozenset(method.upper() for method in methods))
        for path, methods in (reserved_routes or {}).items()
    )
    router = APIRouter()

    for declaration in resolved_operations:
        guarded_matchers.append(declaration.operation_path)
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
        )

    @router.api_route("/{path:path}", methods=list(HTTP_METHODS), include_in_schema=False)
    async def passthrough(request: Request) -> Response:
        normalized_path = re.sub(r"/+", "/", request.url.path).rstrip("/") or "/"
        reserved_methods = next(
            (methods for path_regex, methods in reserved_matchers if path_regex.fullmatch(normalized_path) is not None),
            None,
        )
        if reserved_methods is not None:
            return Response(
                status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                headers={"allow": ", ".join(sorted(reserved_methods))},
            )
        matched_methods = {
            method
            for operation_path in guarded_matchers
            if operation_path.matches(normalized_path)
            for method in operation_path.methods
        }
        if matched_methods:
            if request.method not in matched_methods:
                return Response(
                    status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                    headers={"allow": ", ".join(sorted(matched_methods))},
                )
            return Response(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)
        try:
            buffered_request = await _buffer_request(request, max_request_body_bytes)
            response = await dispatch(buffered_request)
        except RequestBodyTooLarge as failure:
            return _render_failure(
                HttpOperationFailed(HttpFailureKind.REQUEST_BODY_TOO_LARGE, failure),
                render_outcome,
            )
        except InvalidContentLength as failure:
            return _render_failure(
                HttpOperationFailed(HttpFailureKind.INVALID_CONTENT_LENGTH, failure),
                render_outcome,
            )
        except HttpDispatchFailed as failure:
            return _render_failure(
                HttpOperationFailed(HttpFailureKind.UPSTREAM_REQUEST_FAILED, failure),
                render_outcome,
            )
        if len(response.body) > max_response_body_bytes:
            return _render_failure(
                HttpOperationFailed(HttpFailureKind.RESPONSE_BODY_TOO_LARGE, ResponseBodyTooLarge()),
                render_outcome,
            )
        return _render_response(response)

    return router

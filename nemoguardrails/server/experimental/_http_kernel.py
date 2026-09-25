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
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass

from fastapi import APIRouter, Request, status
from starlette.convertors import PathConvertor
from starlette.responses import Response
from starlette.routing import compile_path

from nemoguardrails.server.experimental._buffered_kernel import (
    OperationBlocked,
    OperationCheckFailed,
    OperationCompleted,
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
class GuardedHttpOperation:
    """Bind one buffered guarded operation to an owned HTTP route."""

    path: str
    method: str
    operation: BufferedGuardedOperation[BufferedHttpRequest, BufferedHttpResponse]

    def __post_init__(self) -> None:
        if not self.path.startswith("/") or self.path == "/" or self.path.endswith("/"):
            raise ValueError("A guarded HTTP path must be absolute, non-root, and have no trailing slash.")
        try:
            _, _, convertors = compile_path(self.path)
        except (AssertionError, KeyError, ValueError) as error:
            raise ValueError("A guarded HTTP path must be a valid route template.") from error
        if any(isinstance(convertor, PathConvertor) for convertor in convertors.values()):
            raise ValueError("A guarded HTTP path must not contain a path-spanning parameter.")
        if self.method not in HTTP_METHODS:
            raise ValueError("A guarded HTTP method must be a supported uppercase method.")


HttpDispatch = Callable[[BufferedHttpRequest], Awaitable[BufferedHttpResponse]]
OutcomeRenderer = Callable[[OperationBlocked | OperationCheckFailed], BufferedHttpResponse]


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
    routes = [(operation.method, _route_shape(operation.path)) for operation in resolved]
    if len(routes) != len(set(routes)):
        raise ValueError("Guarded operation routes must be unique.")
    return resolved


def _request_path(request: Request) -> bytes:
    raw_path = request.scope.get("raw_path")
    if isinstance(raw_path, bytes):
        return raw_path
    return request.url.path.encode("ascii")


async def _buffer_request(request: Request, max_body_bytes: int) -> BufferedHttpRequest:
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
        except RequestBodyTooLarge:
            return Response(status_code=status.HTTP_413_CONTENT_TOO_LARGE)
        except ResponseBodyTooLarge:
            return Response(status_code=status.HTTP_502_BAD_GATEWAY)
        if isinstance(outcome, OperationCompleted):
            return _render_response(outcome.response)
        rendered = render_outcome(outcome)
        if len(rendered.body) > max_response_body_bytes:
            return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return _render_response(rendered)

    return handle


def create_http_proxy_router(
    *,
    operations: Collection[GuardedHttpOperation],
    checker: ContentChecker,
    dispatch: HttpDispatch,
    render_outcome: OutcomeRenderer,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    max_response_body_bytes: int = DEFAULT_MAX_RESPONSE_BODY_BYTES,
) -> APIRouter:
    """Create guarded routes followed by a transparent provider catch-all."""

    resolved_operations = _validate_operations(operations)
    if max_request_body_bytes <= 0 or max_response_body_bytes <= 0:
        raise ValueError("Buffered HTTP body limits must be positive.")
    validated_checker = validate_content_checker(checker)
    guarded_matchers = []
    router = APIRouter()

    for declaration in resolved_operations:
        guarded_matchers.append((compile_path(declaration.path)[0], declaration.method))
        router.add_api_route(
            declaration.path,
            _guarded_handler(
                declaration,
                validated_checker,
                dispatch,
                render_outcome,
                max_request_body_bytes,
                max_response_body_bytes,
            ),
            methods=[declaration.method],
            name=declaration.operation.name,
            operation_id=declaration.operation.name,
            response_class=Response,
        )

    @router.api_route("/{path:path}", methods=list(HTTP_METHODS), include_in_schema=False)
    async def passthrough(request: Request) -> Response:
        normalized_path = re.sub(r"/+", "/", request.url.path).rstrip("/") or "/"
        matched_methods = {
            method for path_regex, method in guarded_matchers if path_regex.fullmatch(normalized_path) is not None
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
        except RequestBodyTooLarge:
            return Response(status_code=status.HTTP_413_CONTENT_TOO_LARGE)
        if len(response.body) > max_response_body_bytes:
            return Response(status_code=status.HTTP_502_BAD_GATEWAY)
        return _render_response(response)

    return router

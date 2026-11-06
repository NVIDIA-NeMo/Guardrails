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

"""Create catch-all routes for provider endpoints that are not guarded."""

from collections.abc import Collection, Mapping

from fastapi import APIRouter, Request
from starlette._utils import get_route_path
from starlette.responses import Response

from nemoguardrails.server.experimental._http_kernel import (
    BufferedHttpResponse,
    HttpDispatchFailed,
    HttpFailureKind,
    HttpOperationFailed,
    InvalidContentLength,
    InvalidRequestPath,
    _render_response,
    _request_path,
    _validate_request_path,
    _validate_reserved_routes,
)
from nemoguardrails.server.experimental._http_kernel import (
    _render_failure as _render_http_failure,
)
from nemoguardrails.server.experimental._http_paths import (
    HTTP_METHODS,
    HttpRouteOwnership,
    HttpRouteRejected,
    HttpRouteRejectionKind,
    _route_path_candidates,
)
from nemoguardrails.server.experimental._httpx_response import _HttpxResponseBody
from nemoguardrails.server.experimental._provider_transport import ProviderTransport, _response_headers
from nemoguardrails.server.experimental._streaming_http import _ClosingStreamingResponse
from nemoguardrails.server.experimental.provider.errors import ProviderErrorMapping


async def _passthrough_response(request: Request, transport: ProviderTransport) -> Response:
    """Adapt an unbuffered provider exchange to ASGI with response-owned cleanup."""

    query = request.scope.get("query_string", b"")
    if not isinstance(query, bytes):
        raise TypeError("The downstream query string must be bytes.")
    headers = tuple(request.scope.get("headers", ()))
    bodyless = request.scope.get("http_version") in {"1.0", "1.1"} and not any(
        name.lower() in {b"content-length", b"transfer-encoding"} for name, _ in headers
    )
    response = await transport.open(
        method=request.method,
        path=get_route_path(request.scope),
        raw_path=_request_path(request),
        query=query,
        headers=headers,
        body=b"" if bodyless else request.stream(),
        require_identity_response=False,
    )
    body = _HttpxResponseBody(response)
    try:
        if request.method == "HEAD" or response.status_code < 200 or response.status_code in {204, 205, 304}:
            await body.aclose()
            return _render_response(
                BufferedHttpResponse(response.status_code, _response_headers(response.headers), b""), request.method
            )
        downstream = _ClosingStreamingResponse(body, body, status_code=response.status_code)
        downstream.raw_headers = [
            (name, value) for name, value in _response_headers(response.headers) if name.lower() != b"content-length"
        ]
        return downstream
    except BaseException:
        await body.aclose()
        raise


def create_passthrough_router(
    *,
    transport: ProviderTransport,
    errors: ProviderErrorMapping,
    route_ownership: HttpRouteOwnership,
    reserved_routes: Mapping[str, Collection[str]] | None = None,
) -> APIRouter:
    """Create a provider catch-all that transparently streams requests and responses."""

    reserved_matchers = _validate_reserved_routes(reserved_routes or {})
    router = APIRouter(include_in_schema=False)

    @router.api_route("/{path:path}", methods=list(HTTP_METHODS))
    async def passthrough(request: Request):
        try:
            _validate_request_path(request)
        except InvalidRequestPath:
            return _render_http_failure(
                HttpRouteRejected(HttpRouteRejectionKind.INVALID_REQUEST_PATH), errors.renderer, request.method
            )
        path = get_route_path(request.scope)
        candidates = _route_path_candidates(path)
        if not any(
            operation_path.matches(candidate)
            for operation_path in route_ownership.local_paths
            for candidate in candidates
        ):
            reserved_methods = next(
                (
                    methods
                    for path_regex, methods in reserved_matchers
                    if any(path_regex.fullmatch(candidate) for candidate in candidates)
                ),
                None,
            )
            if reserved_methods is not None:
                return _render_http_failure(
                    HttpRouteRejected(HttpRouteRejectionKind.METHOD_NOT_ALLOWED, reserved_methods),
                    errors.renderer,
                    request.method,
                )
        rejection = route_ownership.reject_passthrough(request.method, path, _request_path(request))
        if rejection is not None:
            return _render_http_failure(rejection, errors.renderer, request.method)
        try:
            return await _passthrough_response(request, transport)
        except InvalidContentLength as failure:
            return _render_http_failure(
                HttpOperationFailed(HttpFailureKind.INVALID_CONTENT_LENGTH, failure), errors.renderer, request.method
            )
        except HttpDispatchFailed as failure:
            return _render_http_failure(
                HttpOperationFailed(HttpFailureKind.UPSTREAM_REQUEST_FAILED, failure), errors.renderer, request.method
            )

    return router

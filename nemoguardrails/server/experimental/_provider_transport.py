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

"""Send guarded proxy exchanges through an application-owned HTTP client."""

import math
from urllib.parse import unquote_to_bytes

import httpx

from nemoguardrails.server.experimental._http_kernel import (
    DEFAULT_MAX_RESPONSE_BODY_BYTES,
    BufferedHttpRequest,
    BufferedHttpResponse,
    HttpDispatchFailed,
    HttpHeaders,
    ResponseBodyTooLarge,
)
from nemoguardrails.server.experimental._httpx_response import _HttpxResponseBody
from nemoguardrails.server.experimental._streaming_http import StreamingHttpResponse

DEFAULT_OUTBOUND_TIMEOUT = 30.0

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


def _connection_tokens(headers: HttpHeaders) -> set[bytes]:
    """Collect case-insensitive header names nominated by Connection fields."""

    tokens = set()
    for name, value in headers:
        if name.lower() == b"connection":
            tokens.update(token.strip().lower() for token in value.split(b",") if token.strip())
    return tokens


def _request_headers(headers: HttpHeaders) -> list[tuple[bytes, bytes]]:
    """Remove connection-specific headers and let HTTPX rebuild host and framing."""

    excluded = (
        _HOP_BY_HOP_HEADERS
        | _connection_tokens(headers)
        | {
            b"host",
            b"content-length",
        }
    )
    return [(name, value) for name, value in headers if name.lower() not in excluded]


def _response_headers(headers: httpx.Headers) -> HttpHeaders:
    """Lowercase ASGI response names, preserving end-to-end values and order."""

    raw = tuple(headers.raw)
    excluded = _HOP_BY_HOP_HEADERS | _connection_tokens(raw)
    return tuple((name.lower(), value) for name, value in raw if name.lower() not in excluded)


def _strip_mount_prefix(path: str, raw_path: bytes, mount_path: str) -> bytes:
    """Match the mount using ASGI's decoded path and preserve its raw suffix.

    The mount and its following decoded slash are replaced by one literal
    upstream separator. Percent escapes in the remaining suffix are untouched.
    """

    if path == mount_path:
        raw_mount = mount_path.encode("ascii")
    elif path.startswith(mount_path + "/"):
        raw_mount = (mount_path + "/").encode("ascii")
    else:
        raise ValueError("The outbound request path is outside the configured mount path.")
    offset = 0
    for character in raw_mount:
        token = raw_path[offset : offset + 1]
        width = 1
        if token == b"%":
            decoded = unquote_to_bytes(raw_path[offset : offset + 3])
            if len(decoded) == 1:
                token, width = decoded, 3
        if token != bytes((character,)):
            raise ValueError("The encoded outbound path does not match the configured mount path.")
        offset += width
    if path == mount_path and offset != len(raw_path):
        raise ValueError("The encoded outbound path does not match the configured mount path.")
    return b"/" + raw_path[offset:]


class ProviderTransport:
    """Dispatch one resolved target through a caller-owned HTTPX client.

    Provider-instance selection happens before this private boundary. This
    object owns responses, not the injected client. Request headers and cookies
    come only from the current caller; client defaults and authentication are
    not inherited. Expected HTTPX network failures become HttpDispatchFailed.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        *,
        base_url: str,
        mount_path: str = "",
        max_response_body_bytes: int = DEFAULT_MAX_RESPONSE_BODY_BYTES,
        timeout: float = DEFAULT_OUTBOUND_TIMEOUT,
    ):
        """Validate the fixed upstream mapping, response bound, and timeout."""

        if not isinstance(http_client, httpx.AsyncClient):
            raise TypeError("The outbound HTTP client must be an httpx.AsyncClient.")
        parsed_base_url = httpx.URL(base_url)
        if not parsed_base_url.is_absolute_url or parsed_base_url.scheme not in {"http", "https"}:
            raise ValueError("The outbound base URL must be an absolute HTTP or HTTPS URL.")
        if parsed_base_url.userinfo:
            raise ValueError("The outbound base URL must not contain credentials.")
        if parsed_base_url.query or parsed_base_url.fragment:
            raise ValueError("The outbound base URL must not contain a query or fragment.")
        normalized_mount = mount_path.rstrip("/")
        if normalized_mount and (not normalized_mount.startswith("/") or not normalized_mount.isascii()):
            raise ValueError("The outbound mount path must be an absolute ASCII path.")
        if (
            type(max_response_body_bytes) is not int
            or max_response_body_bytes <= 0
            or not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or timeout <= 0
            or not math.isfinite(timeout)
        ):
            raise ValueError("The outbound response limit and timeout must be positive.")
        self._http_client = http_client
        self._base_url = parsed_base_url
        self._mount_path = normalized_mount
        self._max_response_body_bytes = max_response_body_bytes
        self._timeout = float(timeout)

    def _upstream_url(self, request: BufferedHttpRequest) -> httpx.URL:
        """Apply the base and mount mapping without reencoding suffix or query bytes."""

        path = request.path
        raw_path = request.raw_path
        if not path.startswith("/") or not raw_path.startswith(b"/") or b"?" in raw_path:
            raise ValueError("An outbound request must contain an absolute path without a query.")
        if self._mount_path:
            raw_path = _strip_mount_prefix(path, raw_path, self._mount_path)
        base_path = self._base_url.raw_path.rstrip(b"/")
        upstream_path = base_path + raw_path
        if request.query:
            upstream_path += b"?" + request.query
        return self._base_url.copy_with(raw_path=upstream_path)

    async def _open(self, request: BufferedHttpRequest) -> httpx.Response:
        """Send caller bytes and headers without inherited headers, cookies, or auth.

        Redirects are disabled. Expected HTTPX request failures become
        HttpDispatchFailed. The returned response remains open and its cleanup
        belongs to the receiving caller.
        """

        upstream_request = httpx.Request(
            request.method,
            self._upstream_url(request),
            headers=_request_headers(request.headers),
            content=request.body,
            extensions={"timeout": httpx.Timeout(self._timeout).as_dict()},
        )
        try:
            return await self._http_client.send(upstream_request, stream=True, follow_redirects=False, auth=None)
        except httpx.RequestError as failure:
            raise HttpDispatchFailed("The upstream request failed.") from failure

    async def _buffer(self, response: httpx.Response) -> BufferedHttpResponse:
        """Bound raw response reads and attempt closure on every exit path.

        Raise ResponseBodyTooLarge above the configured limit and translate
        HTTPX read failures to HttpDispatchFailed.
        """

        body = _HttpxResponseBody(response)
        chunks = []
        size = 0
        try:
            async for chunk in body:
                size += len(chunk)
                if size > self._max_response_body_bytes:
                    raise ResponseBodyTooLarge
                chunks.append(chunk)
        except httpx.RequestError as failure:
            raise HttpDispatchFailed("The upstream response could not be read.") from failure
        finally:
            await body.aclose()
        return BufferedHttpResponse(response.status_code, _response_headers(response.headers), b"".join(chunks))

    async def dispatch(self, request: BufferedHttpRequest) -> BufferedHttpResponse:
        """Send and buffer one response within the limit, attempting closure before return."""

        return await self._buffer(await self._open(request))

    async def stream_dispatch(self, request: BufferedHttpRequest) -> StreamingHttpResponse | BufferedHttpResponse:
        """Transfer a successful response's byte source or buffer a non-success reply.

        The caller must close a returned streaming body even if never iterated.
        Non-success responses use the configured buffering limit and cleanup.
        """

        response = await self._open(request)
        if 200 <= response.status_code <= 299:
            body = _HttpxResponseBody(response)
            try:
                return StreamingHttpResponse(response.status_code, _response_headers(response.headers), body)
            except BaseException:
                await body.aclose()
                raise
        return await self._buffer(response)

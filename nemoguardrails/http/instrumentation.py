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

from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from nemoguardrails.http._url import sanitize_url, split_url
from nemoguardrails.http.client import HTTPClient, ManagedHTTPClient
from nemoguardrails.http.types import HTTPResponse

if TYPE_CHECKING:
    from collections.abc import Callable

    from opentelemetry.trace import Span, Tracer


@dataclass(frozen=True)
class HTTPBodyCapturePolicy:
    capture_request: bool = False
    capture_response: bool = False
    max_bytes: int = 4096
    allowed_content_types: frozenset[str] = field(default_factory=lambda: frozenset({"application/json", "text/plain"}))
    allowed_hosts: frozenset[str] = field(default_factory=frozenset)
    redactor: "Callable[[str], str] | None" = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.max_bytes < 1:
            raise ValueError("max_bytes must be at least 1")
        if self.capture_request or self.capture_response:
            if not self.allowed_hosts:
                raise ValueError("body capture requires at least one allowed host")
            if self.redactor is None:
                raise ValueError("body capture requires a redactor")


class InstrumentedHTTPClient:
    def __init__(
        self,
        client: HTTPClient,
        tracer: "Tracer | None",
        *,
        body_capture: HTTPBodyCapturePolicy | None = None,
    ):
        self._client = client
        self._tracer = tracer
        self._body_capture = body_capture or HTTPBodyCapturePolicy()
        self._closed = False

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        content: bytes | str | None = None,
        timeout: float | None = None,
    ) -> HTTPResponse:
        if self._tracer is None:
            return await self._client.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json,
                content=content,
                timeout=timeout,
            )

        from opentelemetry.trace import SpanKind, StatusCode

        normalized_method = method.upper()
        with self._tracer.start_as_current_span(
            f"HTTP {normalized_method}",
            kind=SpanKind.CLIENT,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            self._set_request_attributes(span, normalized_method, url, content)
            self._capture_request_body(span, url, headers, json, content)
            try:
                response = await self._client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json,
                    content=content,
                    timeout=timeout,
                )
            except BaseException as error:
                self._record_error(span, error, StatusCode)
                raise
            self._set_response_attributes(span, response, StatusCode)
            self._capture_response_body(span, url, response)
            return response

    def _set_request_attributes(self, span: "Span", method: str, url: str, content: bytes | str | None) -> None:
        parts = split_url(url)
        span.set_attribute("http.request.method", method)
        span.set_attribute("url.full", sanitize_url(url))
        if parts.scheme:
            span.set_attribute("url.scheme", parts.scheme)
        if parts.hostname:
            span.set_attribute("server.address", parts.hostname)
        with suppress(ValueError):
            if parts.port is not None:
                span.set_attribute("server.port", parts.port)
        if content is not None:
            size = len(content) if isinstance(content, bytes) else len(content.encode())
            span.set_attribute("http.request.body.size", size)

    def _set_response_attributes(self, span: "Span", response: HTTPResponse, status_code_type: Any) -> None:
        span.set_attribute("http.response.status_code", response.status_code)
        span.set_attribute("http.response.body.size", len(response.content))
        retry_count = response.extensions.get("retry_count")
        if isinstance(retry_count, int) and retry_count > 0:
            span.set_attribute("http.request.resend_count", retry_count)
        if response.status_code >= 400:
            span.set_attribute("error.type", str(response.status_code))
            span.set_status(status_code_type.ERROR)

    def _record_error(self, span: "Span", error: BaseException, status_code_type: Any) -> None:
        with suppress(Exception):
            span.set_attribute("error.type", type(error).__name__)
            retry_count = getattr(error, "retry_count", 0)
            if isinstance(retry_count, int) and retry_count > 0:
                span.set_attribute("http.request.resend_count", retry_count)
            span.record_exception(error)
            span.set_status(status_code_type.ERROR)

    def _capture_request_body(
        self,
        span: "Span",
        url: str,
        headers: Mapping[str, str] | None,
        json_body: Any,
        content: bytes | str | None,
    ) -> None:
        if not self._body_capture.capture_request or not self._host_is_allowed(url):
            return
        if json_body is not None:
            import json

            body = json.dumps(json_body, separators=(",", ":"))
            content_type = "application/json"
        elif content is not None:
            body = content.decode(errors="replace") if isinstance(content, bytes) else content
            content_type = self._content_type(headers or {})
        else:
            return
        self._add_body_event(span, "http.request.body", body, content_type)

    def _capture_response_body(self, span: "Span", url: str, response: HTTPResponse) -> None:
        if not self._body_capture.capture_response or not self._host_is_allowed(url):
            return
        self._add_body_event(
            span,
            "http.response.body",
            response.text,
            self._content_type(response.headers),
        )

    def _host_is_allowed(self, url: str) -> bool:
        hostname = split_url(url).hostname
        allowed_hosts = {host.lower() for host in self._body_capture.allowed_hosts}
        return hostname is not None and hostname.lower() in allowed_hosts

    @staticmethod
    def _content_type(headers: Mapping[str, str]) -> str:
        value = next((value for key, value in headers.items() if key.lower() == "content-type"), "")
        return value.partition(";")[0].strip().lower()

    def _add_body_event(self, span: "Span", name: str, body: str, content_type: str) -> None:
        allowed_types = {value.lower() for value in self._body_capture.allowed_content_types}
        if content_type not in allowed_types or self._body_capture.redactor is None:
            return
        with suppress(Exception):
            redacted = self._body_capture.redactor(body)
            encoded = redacted.encode()
            truncated = len(encoded) > self._body_capture.max_bytes
            captured = encoded[: self._body_capture.max_bytes].decode(errors="ignore")
            span.add_event(name, {"content": captured, "truncated": truncated})

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if isinstance(self._client, ManagedHTTPClient):
            await self._client.close()

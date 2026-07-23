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
from typing import TYPE_CHECKING, Any, Mapping

from nemoguardrails.http._url import sanitize_url, split_url
from nemoguardrails.http.client import ClosableHTTPClient, HTTPClient
from nemoguardrails.http.types import HTTPResponse

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer


class InstrumentedHTTPClient:
    def __init__(
        self,
        client: HTTPClient,
        tracer: "Tracer | None",
    ):
        self._client = client
        self._tracer = tracer
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
            span.add_event("exception", {"exception.type": type(error).__name__})
            span.set_status(status_code_type.ERROR)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if isinstance(self._client, ClosableHTTPClient):
            await self._client.close()

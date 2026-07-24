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

import warnings
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Mapping, overload

from nemoguardrails.http.client import ClosableHTTPClient, HTTPClient
from nemoguardrails.http.telemetry import (
    http_call_span,
    http_request_duration,
    set_http_response_attributes,
)
from nemoguardrails.http.types import HTTPResponse

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer


class InstrumentedHTTPClient:
    def __new__(cls, client: HTTPClient, *args: Any, **kwargs: Any):
        if isinstance(client, cls):
            return client
        return super().__new__(cls)

    def __init__(
        self,
        client: HTTPClient,
        tracer: "Tracer | None",
        *,
        metrics_enabled: bool = False,
    ):
        if client is self:
            if tracer is not self._tracer or metrics_enabled != self._metrics_enabled:
                warnings.warn(
                    "InstrumentedHTTPClient is already instrumented; new instrumentation "
                    "settings are ignored. Re-instrument the underlying wrapped_client instead.",
                    stacklevel=2,
                )
            return
        self._client = client
        self._tracer = tracer
        self._metrics_enabled = metrics_enabled
        self._closed = False

    @property
    def wrapped_client(self) -> HTTPClient:
        return self._client

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
        if self._tracer is None and not self._metrics_enabled:
            return await self._client.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json,
                content=content,
                timeout=timeout,
            )

        normalized_method = method.upper()
        with http_call_span(self._tracer, normalized_method, url, content) as span:
            duration = http_request_duration(normalized_method, url) if self._metrics_enabled else nullcontext(None)
            with duration as metric_state:
                response = await self._client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json,
                    content=content,
                    timeout=timeout,
                )
                if metric_state is not None:
                    metric_state.response_status_code = response.status_code
            set_http_response_attributes(span, response)
            return response

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if isinstance(self._client, ClosableHTTPClient):
            await self._client.close()


@overload
def instrument_http_client(
    client: ClosableHTTPClient,
    *,
    tracer: "Tracer | None" = None,
    metrics_enabled: bool = False,
) -> ClosableHTTPClient: ...


@overload
def instrument_http_client(
    client: HTTPClient,
    *,
    tracer: "Tracer | None" = None,
    metrics_enabled: bool = False,
) -> HTTPClient: ...


def instrument_http_client(
    client: HTTPClient,
    *,
    tracer: "Tracer | None" = None,
    metrics_enabled: bool = False,
) -> HTTPClient:
    if isinstance(client, InstrumentedHTTPClient):
        return client
    if tracer is None and not metrics_enabled:
        return client
    return InstrumentedHTTPClient(client, tracer, metrics_enabled=metrics_enabled)


__all__ = ["InstrumentedHTTPClient", "instrument_http_client"]

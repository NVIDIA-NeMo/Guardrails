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

import asyncio

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from nemoguardrails.http import (
    HTTPConnectionError,
    HTTPResponse,
    InstrumentedHTTPClient,
    RetryingHTTPClient,
    RetryPolicy,
)
from nemoguardrails.http.testing import RecordingHTTPClient


@pytest.fixture
def otel():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


@pytest.mark.asyncio
async def test_instrumented_client_records_safe_http_attributes(otel):
    tracer, exporter = otel
    transport = RecordingHTTPClient(
        [
            HTTPResponse(
                status_code=200,
                headers={"Content-Type": "application/json", "Set-Cookie": "session=secret"},
                content=b'{"token":"response-secret"}',
                extensions={"retry_count": 2},
            )
        ]
    )
    client = InstrumentedHTTPClient(transport, tracer)

    response = await client.request(
        "post",
        "https://user:password@example.com:8443/check?api_key=query-secret#fragment",
        headers={"Authorization": "Bearer header-secret", "Content-Type": "application/json"},
        params={"token": "param-secret"},
        json={"token": "request-secret"},
    )

    assert response.status_code == 200
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "HTTP POST"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET
    assert span.attributes == {
        "http.request.method": "POST",
        "url.full": "https://example.com:8443/check",
        "url.scheme": "https",
        "server.address": "example.com",
        "server.port": 8443,
        "http.response.status_code": 200,
        "http.response.body.size": 27,
        "http.request.resend_count": 2,
    }
    assert span.events == ()
    serialized = repr(span.attributes)
    assert "password" not in serialized
    assert "secret" not in serialized


@pytest.mark.asyncio
async def test_instrumented_client_creates_one_span_for_all_retry_attempts(otel):
    tracer, exporter = otel
    transport = RecordingHTTPClient([HTTPResponse(status_code=503), HTTPResponse(status_code=200)])

    async def sleep(delay: float) -> None:
        return None

    retrying = RetryingHTTPClient(
        transport,
        RetryPolicy(retryable_methods=frozenset({"POST"})),
        sleep=sleep,
    )
    client = InstrumentedHTTPClient(retrying, tracer)

    await client.request("POST", "https://example.com/check", json={"text": "hello"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["http.request.resend_count"] == 1
    assert len(transport.requests) == 2


@pytest.mark.asyncio
async def test_instrumented_client_records_status_errors(otel):
    tracer, exporter = otel
    client = InstrumentedHTTPClient(RecordingHTTPClient([HTTPResponse(status_code=503)]), tracer)

    response = await client.request("GET", "https://example.com/check")

    assert response.status_code == 503
    span = exporter.get_finished_spans()[0]
    assert span.attributes["error.type"] == "503"
    assert span.status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_instrumented_client_preserves_exceptions(otel):
    tracer, exporter = otel
    error = HTTPConnectionError("request failed with token=secret")
    error.retry_count = 2
    client = InstrumentedHTTPClient(RecordingHTTPClient([error]), tracer)

    with pytest.raises(HTTPConnectionError) as exc_info:
        await client.request("GET", "https://example.com/check")

    assert exc_info.value is error
    span = exporter.get_finished_spans()[0]
    assert span.attributes["error.type"] == "HTTPConnectionError"
    assert span.attributes["http.request.resend_count"] == 2
    assert span.status.status_code == StatusCode.ERROR
    assert span.events[0].name == "exception"
    assert span.events[0].attributes == {"exception.type": "HTTPConnectionError"}
    assert "secret" not in repr(span.events)


@pytest.mark.asyncio
async def test_instrumented_client_uses_active_parent(otel):
    tracer, exporter = otel
    client = InstrumentedHTTPClient(RecordingHTTPClient([HTTPResponse(status_code=200)]), tracer)

    with tracer.start_as_current_span("parent") as parent:
        parent_span_id = parent.get_span_context().span_id
        await client.request("GET", "https://example.com/check")

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["HTTP GET"].parent.span_id == parent_span_id


@pytest.mark.asyncio
async def test_disabled_instrumentation_is_a_passthrough(otel):
    _, exporter = otel
    response = HTTPResponse(status_code=200)
    transport = RecordingHTTPClient([response])
    client = InstrumentedHTTPClient(transport, None)

    result = await client.request("GET", "https://example.com/check")

    assert result is response
    assert exporter.get_finished_spans() == ()


@pytest.mark.asyncio
async def test_instrumented_client_closes_wrapped_client_once():
    transport = RecordingHTTPClient()
    client = InstrumentedHTTPClient(transport, None)

    await asyncio.gather(client.close(), client.close())

    assert transport.close_calls == 1

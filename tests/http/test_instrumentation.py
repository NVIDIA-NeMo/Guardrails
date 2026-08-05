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
import logging
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from nemoguardrails.guardrails import telemetry as guardrails_telemetry
from nemoguardrails.http import (
    HTTPConnectionError,
    HTTPResponse,
    InstrumentedHTTPClient,
    RetryingHTTPClient,
    RetryPolicy,
    instrument_http_client,
)
from nemoguardrails.http.telemetry import record_http_error
from nemoguardrails.testing.http import RecordingHTTPClient
from nemoguardrails.tracing import constants as tracing_constants
from nemoguardrails.tracing.constants import SystemConstants
from tests.guardrails.metric_helpers import collect_metric_points


@pytest.fixture
def otel():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


@pytest.fixture
def metric_reader():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    previous_meter = guardrails_telemetry._meter
    previous_instruments = tracing_constants._http_instruments
    guardrails_telemetry._meter = provider.get_meter(SystemConstants.SYSTEM_NAME)
    tracing_constants._http_instruments = None
    yield reader
    guardrails_telemetry._meter = previous_meter
    tracing_constants._http_instruments = previous_instruments


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
async def test_instrumented_client_creates_one_observation_for_all_retry_attempts(otel, metric_reader):
    tracer, exporter = otel
    transport = RecordingHTTPClient([HTTPResponse(status_code=503), HTTPResponse(status_code=200)])

    async def sleep(delay: float) -> None:
        return None

    retrying = RetryingHTTPClient(
        transport,
        RetryPolicy(retryable_methods=frozenset({"POST"})),
        sleep=sleep,
    )
    client = InstrumentedHTTPClient(retrying, tracer, metrics_enabled=True)

    await client.request("POST", "https://example.com/check", json={"text": "hello"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["http.request.resend_count"] == 1
    assert len(collect_metric_points(metric_reader)["http.client.request.duration"]) == 1
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


def test_instrumented_client_reports_error_telemetry_failures(caplog):
    span = MagicMock()
    span.set_attribute.side_effect = TypeError("invalid attribute")

    with caplog.at_level(logging.WARNING):
        record_http_error(span, HTTPConnectionError("token=secret"))

    assert "Failed to record HTTP error telemetry: TypeError" in caplog.text
    assert "secret" not in caplog.text


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


@pytest.mark.asyncio
async def test_concurrent_close_waits_and_retries_after_failure():
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    error = RuntimeError("close failed")

    async def close() -> None:
        close_started.set()
        await allow_close.wait()
        raise error

    transport = RecordingHTTPClient()
    transport.close = AsyncMock(side_effect=close)
    client = InstrumentedHTTPClient(transport, None)

    first = asyncio.create_task(client.close())
    await close_started.wait()
    second = asyncio.create_task(client.close())
    await asyncio.sleep(0)

    assert not second.done()

    allow_close.set()
    results = await asyncio.gather(first, second, return_exceptions=True)

    assert results == [error, error]
    assert transport.close.await_count == 2


@pytest.mark.asyncio
async def test_instrumentation_is_idempotent(otel, metric_reader):
    tracer, exporter = otel
    transport = RecordingHTTPClient([HTTPResponse(status_code=200)])

    first = instrument_http_client(transport, tracer=tracer, metrics_enabled=True)
    second = instrument_http_client(first, tracer=tracer, metrics_enabled=True)
    third = InstrumentedHTTPClient(second, tracer, metrics_enabled=True)

    assert second is first
    assert third is first
    assert isinstance(first, InstrumentedHTTPClient)
    assert first.wrapped_client is transport

    await third.request("GET", "https://example.com/check")

    assert len(exporter.get_finished_spans()) == 1
    assert len(collect_metric_points(metric_reader)["http.client.request.duration"]) == 1


def test_reinstrument_with_changed_tracer_warns_and_keeps_original(otel):
    tracer, _ = otel
    original = InstrumentedHTTPClient(RecordingHTTPClient(), tracer)

    with pytest.warns(UserWarning, match="already instrumented"):
        result = InstrumentedHTTPClient(original, MagicMock())

    assert result is original


def test_reinstrument_with_identical_tracer_does_not_warn(otel):
    tracer, _ = otel
    original = InstrumentedHTTPClient(RecordingHTTPClient(), tracer)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = InstrumentedHTTPClient(original, tracer)

    assert result is original


def test_instrument_http_client_disabled_returns_original():
    transport = RecordingHTTPClient()

    assert instrument_http_client(transport) is transport


@pytest.mark.asyncio
async def test_instrumented_client_preserves_wrapped_response_and_ownership(otel):
    tracer, exporter = otel
    response = HTTPResponse(status_code=200, content=b"ok")
    transport = RecordingHTTPClient([response])
    client = instrument_http_client(transport, tracer=tracer)

    result = await client.request("GET", "https://example.com/check")

    assert result is response
    assert len(exporter.get_finished_spans()) == 1
    assert transport.close_calls == 0


@pytest.mark.asyncio
async def test_attribute_telemetry_failures_do_not_change_http_result():
    span = MagicMock()
    span.set_attribute.side_effect = TypeError("invalid attribute")
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__.return_value = span
    response = HTTPResponse(status_code=200, content=b"ok")
    client = InstrumentedHTTPClient(RecordingHTTPClient([response]), tracer)

    result = await client.request("GET", "https://example.com/check")

    assert result is response


@pytest.mark.asyncio
async def test_metrics_only_records_http_request_duration(metric_reader):
    response = HTTPResponse(status_code=200)
    transport = RecordingHTTPClient([response])
    client = instrument_http_client(transport, metrics_enabled=True)

    result = await client.request("post", "https://example.com/check")

    assert result is response
    points = collect_metric_points(metric_reader)["http.client.request.duration"]
    assert len(points) == 1
    assert points[0].attributes == {
        "http.request.method": "POST",
        "server.address": "example.com",
        "server.port": 443,
        "http.response.status_code": 200,
    }


@pytest.mark.asyncio
async def test_http_error_metrics_include_error_type(metric_reader):
    response = HTTPResponse(status_code=503)
    client = instrument_http_client(
        RecordingHTTPClient([response]),
        metrics_enabled=True,
    )

    result = await client.request("GET", "http://example.com/check")

    assert result is response
    point = collect_metric_points(metric_reader)["http.client.request.duration"][0]
    assert point.attributes["server.port"] == 80
    assert point.attributes["http.response.status_code"] == 503
    assert point.attributes["error.type"] == "503"


@pytest.mark.asyncio
async def test_http_exception_metrics_include_error_type(metric_reader):
    error = HTTPConnectionError("unavailable")
    client = instrument_http_client(
        RecordingHTTPClient([error]),
        metrics_enabled=True,
    )

    with pytest.raises(HTTPConnectionError) as exc_info:
        await client.request("GET", "https://example.com/check")

    assert exc_info.value is error
    point = collect_metric_points(metric_reader)["http.client.request.duration"][0]
    assert point.attributes["error.type"] == "HTTPConnectionError"
    assert "http.response.status_code" not in point.attributes


@pytest.mark.asyncio
async def test_metric_failures_do_not_change_http_result():
    instruments = MagicMock()
    instruments.request_duration.record.side_effect = TypeError("invalid metric")
    response = HTTPResponse(status_code=200)
    client = instrument_http_client(
        RecordingHTTPClient([response]),
        metrics_enabled=True,
    )

    with patch(
        "nemoguardrails.http.telemetry._ensure_http_instruments",
        return_value=instruments,
    ):
        result = await client.request("GET", "https://example.com/check")

    assert result is response

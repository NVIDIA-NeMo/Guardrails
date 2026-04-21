# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for nemoguardrails.guardrails.telemetry module."""

import re
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode, format_trace_id

from nemoguardrails.guardrails import telemetry
from nemoguardrails.guardrails.guardrails_types import REQUEST_ID_HEX_CHARS
from nemoguardrails.guardrails.telemetry import (
    get_tracer,
    is_tracing_enabled,
    mark_rail_stop,
    record_span_error,
    request_span,
    trace_id_to_request_id,
    traced_request,
)

_HEX_PATTERN = re.compile(r"^[0-9a-f]+$")


def _is_valid_hex_string(value: str, expected_length: int) -> bool:
    """Return True if *value* is a lowercase hex string of exactly *expected_length* chars."""
    return len(value) == expected_length and _HEX_PATTERN.match(value) is not None


@pytest.fixture(autouse=True)
def reset_tracer_singleton():
    """Reset the module-level tracer singleton between tests."""
    telemetry._tracer = None
    yield
    telemetry._tracer = None


@pytest.fixture
def otel_provider():
    """Set up a real TracerProvider with an InMemorySpanExporter for assertions."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


class TestIsValidHexString:
    def test_happy_path(self):
        assert _is_valid_hex_string("abcdef0123456789", 16) is True

    def test_length_mismatch(self):
        assert _is_valid_hex_string("abcd", 16) is False

    def test_invalid_hex_chars(self):
        assert _is_valid_hex_string("zzzzzzzzzzzzzzzz", 16) is False

    def test_invalid_chars_and_length_mismatch(self):
        assert _is_valid_hex_string("xyz", 16) is False


class TestGetTracer:
    def test_returns_tracer(self):
        tracer = get_tracer()
        assert tracer is not None

    def test_returns_same_instance(self):
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2

    def test_returns_none_without_otel(self):
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._tracer = None
            assert get_tracer() is None


class TestTraceIdToRequestId:
    def test_format_is_16_hex_chars(self, otel_provider):
        provider, _ = otel_provider
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("test") as span:
            req_id = trace_id_to_request_id(span)
            assert _is_valid_hex_string(req_id, REQUEST_ID_HEX_CHARS)

    def test_matches_trace_id_suffix(self, otel_provider):
        provider, _ = otel_provider
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("test") as span:
            full_trace_id = format_trace_id(span.get_span_context().trace_id)
            req_id = trace_id_to_request_id(span)
            assert full_trace_id.endswith(req_id)

    def test_zero_trace_id_falls_back_to_random(self):
        span = MagicMock()
        ctx = MagicMock()
        ctx.trace_id = 0
        span.get_span_context.return_value = ctx

        req_id = trace_id_to_request_id(span)
        assert _is_valid_hex_string(req_id, REQUEST_ID_HEX_CHARS)


class TestRequestSpan:
    def test_creates_server_span(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with request_span(tracer) as _:
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "guardrails.request"
        assert spans[0].kind == SpanKind.SERVER

    def test_sets_required_attributes(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with request_span(tracer) as (_, req_id):
            pass

        attrs = dict(spans[0].attributes) if (spans := exporter.get_finished_spans()) else {}
        assert attrs["gen_ai.operation.name"] == "guardrails"
        assert attrs["request.id"] == req_id

    def test_service_name_not_on_span(self, otel_provider):
        """service.name is a Resource attribute, not a span attribute."""
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with request_span(tracer) as _:
            pass

        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert "service.name" not in attrs

    def test_service_name_on_resource(self):
        """service.name should come from the TracerProvider's Resource."""
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": "nemo-guardrails"})
        provider = TracerProvider(resource=resource)
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        with request_span(tracer) as _:
            pass

        finished = exporter.get_finished_spans()
        assert finished[0].resource.attributes["service.name"] == "nemo-guardrails"

    def test_request_id_is_16_hex(self, otel_provider):
        provider, _ = otel_provider
        tracer = provider.get_tracer("test")

        with request_span(tracer) as (_, req_id):
            assert _is_valid_hex_string(req_id, REQUEST_ID_HEX_CHARS)

    def test_records_exception_on_error(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with pytest.raises(ValueError, match="boom"):
            with request_span(tracer) as _:
                raise ValueError("boom")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.ERROR
        events = spans[0].events
        exception_events = [e for e in events if e.name == "exception"]
        assert len(exception_events) == 1
        assert "boom" in exception_events[0].attributes["exception.message"]

    def test_span_ended_on_success(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with request_span(tracer) as _:
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].end_time is not None


class TestRecordSpanError:
    def test_noop_when_span_is_none(self):
        # Should not raise
        record_span_error(None, RuntimeError("boom"))

    def test_records_on_live_span(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("test") as span:
            record_span_error(span, ValueError("bad"))

        finished = exporter.get_finished_spans()[0]
        assert finished.status.status_code == StatusCode.ERROR
        exc_events = [e for e in finished.events if e.name == "exception"]
        assert len(exc_events) == 1
        assert exc_events[0].attributes["exception.type"] == "ValueError"


class TestMarkRailStop:
    """``mark_rail_stop`` encapsulates the None-span + rail-safe conditional."""

    def test_noop_when_span_is_none(self):
        # Tracer disabled → rail_span yields None; helper must not crash.
        mark_rail_stop(None, is_safe=False)

    def test_sets_attribute_when_rail_blocks(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("test") as span:
            mark_rail_stop(span, is_safe=False)

        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["rail.stop"] is True

    def test_does_not_set_attribute_when_rail_passes(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("test") as span:
            mark_rail_stop(span, is_safe=True)

        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert "rail.stop" not in attrs


class TestIsTracingEnabled:
    def test_enabled_with_otel(self):
        config = MagicMock()
        config.enabled = True
        assert is_tracing_enabled(config) is True

    def test_disabled_when_config_disabled(self):
        config = MagicMock()
        config.enabled = False
        assert is_tracing_enabled(config) is False

    def test_disabled_when_config_none(self):
        assert is_tracing_enabled(None) is False

    def test_disabled_without_otel(self):
        config = MagicMock()
        config.enabled = True
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            assert is_tracing_enabled(config) is False


class TestTracedRequestValueErrorTolerance:
    def test_value_error_on_reset_swallowed_in_tracer_branch(self, otel_provider):
        """traced_request must tolerate ValueError from reset_request_id — this
        models async-generator cleanup running in a different task context.

        The ValueError happens in the token-reset after ``yield``, so the span
        should still close cleanly with UNSET status and a valid req_id
        derived from the trace ID.
        """
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with patch.object(
            telemetry, "reset_request_id", side_effect=ValueError("<Token> was created in a different Context")
        ):
            with traced_request(tracer) as traced:
                # With a tracer, traced_request must yield a real span.
                assert traced.span is not None
                captured_req_id = traced.request_id

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "guardrails.request"
        assert span.kind == SpanKind.SERVER
        assert span.status.status_code == StatusCode.UNSET

        # request_id is a valid hex suffix of the trace ID
        assert _is_valid_hex_string(captured_req_id, REQUEST_ID_HEX_CHARS)
        assert format_trace_id(span.context.trace_id).endswith(captured_req_id)

        # The span carries the request_id attribute set by request_span
        attrs = dict(span.attributes)
        assert attrs["request.id"] == captured_req_id
        assert attrs["gen_ai.operation.name"] == "guardrails"

        # The swallowed ValueError must NOT leak into the span: no exception
        # events, no error.type attribute, no trace of the error anywhere.
        assert [e for e in span.events if e.name == "exception"] == []
        assert "error.type" not in attrs
        assert not span.status.description
        for value in attrs.values():
            assert "different Context" not in str(value)

    def test_value_error_on_reset_swallowed_in_no_tracer_branch(self):
        """Same tolerance in the tracer=None branch (random req-id path).

        No span is created; we only verify the yielded request_id is a valid
        hex string and the ValueError is not propagated.
        """
        with patch.object(
            telemetry, "reset_request_id", side_effect=ValueError("<Token> was created in a different Context")
        ):
            with traced_request(None) as traced:
                # Without a tracer, traced_request must yield None for the span.
                assert traced.span is None
                captured_req_id = traced.request_id

        assert _is_valid_hex_string(captured_req_id, REQUEST_ID_HEX_CHARS)

    def test_unexpected_value_error_is_reraised(self):
        """Only the cross-context ValueError is swallowed.  Any other
        ValueError from reset_request_id (e.g. a future refactor bug) must
        propagate loudly — catching all ValueErrors would hide regressions.
        """
        with patch.object(telemetry, "reset_request_id", side_effect=ValueError("unrelated failure")):
            with pytest.raises(ValueError, match="unrelated failure"):
                with traced_request(None):
                    pass

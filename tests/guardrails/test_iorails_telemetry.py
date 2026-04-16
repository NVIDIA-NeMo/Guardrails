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

"""Integration tests: IORails generate_async with inline OTEL instrumentation."""

import asyncio
import copy
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode, format_trace_id

from nemoguardrails.guardrails import telemetry
from nemoguardrails.guardrails.guardrails_types import REQUEST_ID_HEX_CHARS, RailResult, get_request_id
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.rails.llm.config import RailsConfig
from tests.guardrails.test_data import NEMOGUARDS_CONFIG
from tests.guardrails.test_telemetry import _is_valid_hex_string


def _make_tracing_config():
    """Return a NEMOGUARDS_CONFIG copy with tracing enabled."""
    cfg = copy.deepcopy(NEMOGUARDS_CONFIG)
    cfg["tracing"] = {"enabled": True}
    return cfg


def _stub_safe_pipeline(iorails, llm_response="Hello"):
    """Mock input/output rails as safe and the LLM to return *llm_response*."""
    iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
    iorails.engine_registry.model_call = AsyncMock(return_value=llm_response)
    iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def tracer_from_provider(exporter):
    """Create a TracerProvider with InMemorySpanExporter and return its tracer."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test")


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def iorails_tracing(tracer_from_provider):
    """IORails instance with tracing enabled, using a test tracer."""
    config = RailsConfig.from_content(config=_make_tracing_config())
    iorails = IORails(config)
    # Inject the test tracer directly so spans go to our InMemorySpanExporter
    iorails._tracer = tracer_from_provider
    iorails._tracing_enabled = True
    return iorails


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def iorails_no_tracing():
    """IORails instance with default config (tracing disabled)."""
    return IORails(RailsConfig.from_content(config=NEMOGUARDS_CONFIG))


class TestGenerateAsyncWithTracing:
    @pytest.mark.asyncio
    async def test_creates_span(self, iorails_tracing, exporter):
        _stub_safe_pipeline(iorails_tracing)

        result = await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        assert result == {"role": "assistant", "content": "Hello"}
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "guardrails.request"
        assert spans[0].kind == SpanKind.SERVER

    @pytest.mark.asyncio
    async def test_span_has_required_attributes(self, iorails_tracing, exporter):
        _stub_safe_pipeline(iorails_tracing)

        await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["gen_ai.operation.name"] == "guardrails"
        assert "request.id" in attrs
        assert len(attrs["request.id"]) == REQUEST_ID_HEX_CHARS

    @pytest.mark.asyncio
    async def test_request_id_derived_from_trace_id(self, iorails_tracing, exporter):
        """The request ID visible to downstream code matches the span's trace ID suffix."""
        captured_req_id = None

        async def capture_req_id(messages):
            nonlocal captured_req_id
            captured_req_id = get_request_id()
            return RailResult(is_safe=True)

        _stub_safe_pipeline(iorails_tracing)
        iorails_tracing.rails_manager.is_input_safe = capture_req_id

        await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        spans = exporter.get_finished_spans()
        span_req_id = spans[0].attributes["request.id"]
        assert captured_req_id == span_req_id

    @pytest.mark.asyncio
    async def test_span_records_exception(self, iorails_tracing, exporter):
        _stub_safe_pipeline(iorails_tracing)
        iorails_tracing.engine_registry.model_call = AsyncMock(side_effect=RuntimeError("LLM failed"))

        with pytest.raises(RuntimeError, match="LLM failed"):
            await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.ERROR
        exception_events = [e for e in spans[0].events if e.name == "exception"]
        assert len(exception_events) == 1

    @pytest.mark.asyncio
    async def test_span_created_on_input_block(self, iorails_tracing, exporter):
        """A span is still created and completed even when input rails block."""
        iorails_tracing.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=False, reason="unsafe"))

        result = await iorails_tracing.generate_async([{"role": "user", "content": "bad"}])

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "guardrails.request"


class TestGenerateAsyncWithoutTracing:
    @pytest.mark.asyncio
    async def test_no_spans_exported(self, iorails_no_tracing, exporter):
        _stub_safe_pipeline(iorails_no_tracing)

        result = await iorails_no_tracing.generate_async([{"role": "user", "content": "hi"}])

        assert result == {"role": "assistant", "content": "Hello"}
        assert len(exporter.get_finished_spans()) == 0

    @pytest.mark.asyncio
    async def test_request_id_length_consistent_without_tracing(self, iorails_no_tracing):
        """Without tracing, request IDs are random but the same length as trace-derived IDs."""
        captured_req_id = None

        async def capture_req_id(messages):
            nonlocal captured_req_id
            captured_req_id = get_request_id()
            return RailResult(is_safe=True)

        _stub_safe_pipeline(iorails_no_tracing)
        iorails_no_tracing.rails_manager.is_input_safe = capture_req_id

        await iorails_no_tracing.generate_async([{"role": "user", "content": "hi"}])

        assert captured_req_id is not None
        assert len(captured_req_id) == REQUEST_ID_HEX_CHARS


class TestEndToEndTracing:
    """End-to-end tests that verify the full tracing lifecycle."""

    @pytest.mark.asyncio
    async def test_full_trace_lifecycle(self, iorails_tracing, exporter):
        """Verify span timing, attributes, status, and request-ID propagation
        through the entire generate_async pipeline."""
        captured = {}

        async def capturing_input_check(messages):
            captured["input_req_id"] = get_request_id()
            captured["input_messages"] = messages
            return RailResult(is_safe=True)

        async def capturing_model_call(model_name, messages, **kwargs):
            captured["llm_req_id"] = get_request_id()
            captured["llm_model"] = model_name
            return "Generated response"

        async def capturing_output_check(messages, response):
            captured["output_req_id"] = get_request_id()
            captured["output_response"] = response
            return RailResult(is_safe=True)

        iorails_tracing.rails_manager.is_input_safe = capturing_input_check
        iorails_tracing.engine_registry.model_call = capturing_model_call
        iorails_tracing.rails_manager.is_output_safe = capturing_output_check

        messages = [{"role": "user", "content": "hello"}]
        result = await iorails_tracing.generate_async(messages)

        # Response correctness
        assert result == {"role": "assistant", "content": "Generated response"}

        # Span structure
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "guardrails.request"
        assert span.kind == SpanKind.SERVER
        assert span.status.status_code == StatusCode.UNSET

        # Timing sanity
        assert span.start_time > 0
        assert span.end_time >= span.start_time

        # Attributes
        attrs = dict(span.attributes)
        assert attrs["gen_ai.operation.name"] == "guardrails"
        req_id = attrs["request.id"]
        assert _is_valid_hex_string(req_id, REQUEST_ID_HEX_CHARS)

        # Request ID derived from trace ID
        full_trace_id = format_trace_id(span.context.trace_id)
        assert full_trace_id.endswith(req_id)

        # Same request ID visible at every pipeline stage
        assert captured["input_req_id"] == req_id
        assert captured["llm_req_id"] == req_id
        assert captured["output_req_id"] == req_id

        # Pipeline received correct data
        assert captured["input_messages"] == messages
        assert captured["llm_model"] == "main"
        assert captured["output_response"] == "Generated response"

    @pytest.mark.asyncio
    async def test_concurrent_requests_get_distinct_traces(self, iorails_tracing, exporter):
        """Two concurrent generate_async calls produce separate spans
        with distinct trace IDs and request IDs."""
        req_ids_seen = []

        async def record_req_id(messages):
            req_ids_seen.append(get_request_id())
            return RailResult(is_safe=True)

        _stub_safe_pipeline(iorails_tracing)
        iorails_tracing.rails_manager.is_input_safe = record_req_id

        messages = [{"role": "user", "content": "hi"}]
        await asyncio.gather(
            iorails_tracing.generate_async(messages),
            iorails_tracing.generate_async(messages),
        )

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        trace_ids = {s.context.trace_id for s in spans}
        assert len(trace_ids) == 2

        span_req_ids = {s.attributes["request.id"] for s in spans}
        assert len(span_req_ids) == 2

        assert set(req_ids_seen) == span_req_ids

    @pytest.mark.asyncio
    async def test_error_span_preserves_full_context(self, iorails_tracing, exporter):
        """On LLM failure: span records the exception type, message,
        and traceback, and the request ID is still valid."""
        captured_req_id = None

        async def capture_then_pass(messages):
            nonlocal captured_req_id
            captured_req_id = get_request_id()
            return RailResult(is_safe=True)

        iorails_tracing.rails_manager.is_input_safe = capture_then_pass
        iorails_tracing.engine_registry.model_call = AsyncMock(side_effect=RuntimeError("connection refused"))

        with pytest.raises(RuntimeError, match="connection refused"):
            await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        # Status
        assert span.status.status_code == StatusCode.ERROR
        assert "connection refused" in span.status.description

        # Exception event
        exc_events = [e for e in span.events if e.name == "exception"]
        assert len(exc_events) == 1
        assert exc_events[0].attributes["exception.type"] == "RuntimeError"
        assert "connection refused" in exc_events[0].attributes["exception.message"]
        assert "exception.stacktrace" in exc_events[0].attributes

        # Request ID was still set correctly before the error
        assert captured_req_id == span.attributes["request.id"]

        # Span still has valid timing
        assert span.end_time >= span.start_time


class TestOtelNotInstalled:
    @pytest.mark.asyncio
    async def test_falls_back_gracefully(self, exporter):
        """When OTEL is not available, IORails works without tracing."""
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._tracer = None
            with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
                config = RailsConfig.from_content(config=_make_tracing_config())
                iorails = IORails(config)

            _stub_safe_pipeline(iorails)

            result = await iorails.generate_async([{"role": "user", "content": "hi"}])

            assert result == {"role": "assistant", "content": "Hello"}
            assert iorails._tracing_enabled is False
            assert len(exporter.get_finished_spans()) == 0

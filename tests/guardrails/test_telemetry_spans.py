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

"""Unit tests for telemetry span helpers: rail_span, action_span, llm_call_span, api_call_span."""

import asyncio

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from nemoguardrails.guardrails.guardrails_types import RailDirection
from nemoguardrails.guardrails.telemetry import (
    action_span,
    api_call_span,
    llm_call_span,
    rail_span,
)


@pytest.fixture
def otel_provider():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


class TestRailSpan:
    def test_creates_internal_span(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with rail_span(tracer, "content safety check input $model=content_safety", RailDirection.INPUT) as _:
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "guardrails.rail"
        assert spans[0].kind == SpanKind.INTERNAL

    def test_sets_attributes(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with rail_span(tracer, "content safety check output $model=content_safety", RailDirection.OUTPUT):
            pass

        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["rail.type"] == "Output"
        assert attrs["rail.name"] == "content safety check output $model=content_safety"

    def test_noop_when_tracer_none(self):
        with rail_span(None, "some flow", RailDirection.INPUT) as span:
            assert span is None

    def test_records_exception(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with pytest.raises(RuntimeError, match="rail failed"):
            with rail_span(tracer, "some flow", RailDirection.INPUT):
                raise RuntimeError("rail failed")

        spans = exporter.get_finished_spans()
        assert spans[0].status.status_code == StatusCode.ERROR

    def test_records_error_type_on_cancelled_error(self, otel_provider):
        """A consumer cancel that propagates through a rail span must
        mark it ERROR with ``error.type=CancelledError`` — otherwise
        the rail leg of a cancelled-request trace is silently untagged.
        """
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with pytest.raises(asyncio.CancelledError):
            with rail_span(tracer, "some flow", RailDirection.INPUT):
                raise asyncio.CancelledError()

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "CancelledError"

    def test_records_error_type_on_generator_exit(self, otel_provider):
        """``GeneratorExit`` propagating through a rail span must mark
        it ERROR with ``error.type=GeneratorExit``."""
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with pytest.raises(GeneratorExit):
            with rail_span(tracer, "some flow", RailDirection.INPUT):
                raise GeneratorExit()

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "GeneratorExit"


class TestActionSpan:
    def test_creates_internal_span(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with action_span(tracer, "content safety check input"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "guardrails.action"
        assert spans[0].kind == SpanKind.INTERNAL

    def test_sets_action_name(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with action_span(tracer, "jailbreak detection"):
            pass

        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["action.name"] == "jailbreak detection"

    def test_noop_when_tracer_none(self):
        with action_span(None, "some action") as span:
            assert span is None

    def test_records_exception(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with pytest.raises(RuntimeError, match="action failed"):
            with action_span(tracer, "some action"):
                raise RuntimeError("action failed")

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.ERROR
        exc_events = [e for e in span.events if e.name == "exception"]
        assert len(exc_events) == 1
        assert exc_events[0].attributes["exception.type"] == "RuntimeError"

    def test_records_error_type_on_cancelled_error(self, otel_provider):
        """A consumer cancel propagating through an action span must
        mark it ERROR with ``error.type=CancelledError``."""
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with pytest.raises(asyncio.CancelledError):
            with action_span(tracer, "some action"):
                raise asyncio.CancelledError()

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "CancelledError"

    def test_records_error_type_on_generator_exit(self, otel_provider):
        """``GeneratorExit`` propagating through an action span must
        mark it ERROR with ``error.type=GeneratorExit``."""
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with pytest.raises(GeneratorExit):
            with action_span(tracer, "some action"):
                raise GeneratorExit()

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "GeneratorExit"


class TestLlmCallSpan:
    def test_creates_client_span(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with llm_call_span(tracer, "meta/llama-3.3-70b-instruct", "nim"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].kind == SpanKind.CLIENT

    def test_span_name_follows_convention(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with llm_call_span(tracer, "meta/llama-3.3-70b-instruct", "nim"):
            pass

        assert exporter.get_finished_spans()[0].name == "chat meta/llama-3.3-70b-instruct"

    def test_sets_genai_attributes(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with llm_call_span(tracer, "meta/llama-3.3-70b-instruct", "nim", "chat"):
            pass

        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["gen_ai.operation.name"] == "chat"
        assert attrs["gen_ai.request.model"] == "meta/llama-3.3-70b-instruct"
        assert attrs["gen_ai.provider.name"] == "nim"

    def test_records_error_type(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with pytest.raises(ConnectionError):
            with llm_call_span(tracer, "model", "nim"):
                raise ConnectionError("timeout")

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "ConnectionError"

    def test_records_error_type_on_cancelled_error(self, otel_provider):
        """Consumer-cancelled streams raise ``asyncio.CancelledError``
        inside the LLM CLIENT span.  Span must still be marked ERROR
        with ``error.type=CancelledError`` so trace queries can
        correlate cancelled streams to their LLM-call leg.
        """
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with pytest.raises(asyncio.CancelledError):
            with llm_call_span(tracer, "model", "nim"):
                raise asyncio.CancelledError()

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "CancelledError"

    def test_records_error_type_on_generator_exit(self, otel_provider):
        """``GeneratorExit`` raised inside the LLM CLIENT span must
        also flip the span to ERROR with ``error.type=GeneratorExit``.
        """
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with pytest.raises(GeneratorExit):
            with llm_call_span(tracer, "model", "nim"):
                raise GeneratorExit()

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "GeneratorExit"

    def test_noop_when_tracer_none(self):
        with llm_call_span(None, "model", "nim") as span:
            assert span is None


class TestApiCallSpan:
    def test_creates_client_span(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with api_call_span(tracer, "jailbreak_detection"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].kind == SpanKind.CLIENT
        assert spans[0].name == "api jailbreak_detection"

    def test_sets_api_name(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with api_call_span(tracer, "jailbreak_detection"):
            pass

        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["api.name"] == "jailbreak_detection"
        # Must NOT appear in the gen_ai.* namespace: this is a plain HTTP
        # API call, not a GenAI operation.
        assert "gen_ai.operation.name" not in attrs

    def test_records_error_type(self, otel_provider):
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with pytest.raises(ValueError):
            with api_call_span(tracer, "jailbreak_detection"):
                raise ValueError("bad response")

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "ValueError"

    def test_records_error_type_on_cancelled_error(self, otel_provider):
        """A consumer cancel propagating through an api-call span
        (e.g. an in-flight jailbreak-detection HTTP request) must mark
        it ERROR with ``error.type=CancelledError``."""
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with pytest.raises(asyncio.CancelledError):
            with api_call_span(tracer, "jailbreak_detection"):
                raise asyncio.CancelledError()

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "CancelledError"

    def test_records_error_type_on_generator_exit(self, otel_provider):
        """``GeneratorExit`` propagating through an api-call span must
        mark it ERROR with ``error.type=GeneratorExit``."""
        provider, exporter = otel_provider
        tracer = provider.get_tracer("test")

        with pytest.raises(GeneratorExit):
            with api_call_span(tracer, "jailbreak_detection"):
                raise GeneratorExit()

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "GeneratorExit"

    def test_noop_when_tracer_none(self):
        with api_call_span(None, "jailbreak_detection") as span:
            assert span is None

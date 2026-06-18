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
    set_llm_request_attributes,
    set_llm_response_attributes,
)
from nemoguardrails.types import UsageInfo


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


def _span_attrs(otel_provider, set_fn):
    """Run ``set_fn(span)`` inside a finished CLIENT span and return its
    attributes as a plain dict.

    Mirrors how the helpers are used in production (called on a live span
    inside the ``llm_call_span`` block) and reads the result back off the
    exported span, the same way the span-helper tests above do.
    """
    provider, exporter = otel_provider
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("chat test-model") as span:
        set_fn(span)
    return dict(exporter.get_finished_spans()[-1].attributes)


class TestSetLlmRequestAttributes:
    def test_maps_scalar_params(self, otel_provider):
        """Every supported scalar sampling param maps to its gen_ai.request.* attr."""
        params = {
            "temperature": 0.7,
            "max_tokens": 256,
            "top_p": 0.9,
            "top_k": 40,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.25,
        }
        attrs = _span_attrs(otel_provider, lambda s: set_llm_request_attributes(s, params))
        assert attrs["gen_ai.request.temperature"] == 0.7
        assert attrs["gen_ai.request.max_tokens"] == 256
        assert attrs["gen_ai.request.top_p"] == 0.9
        assert attrs["gen_ai.request.top_k"] == 40
        assert attrs["gen_ai.request.frequency_penalty"] == 0.5
        assert attrs["gen_ai.request.presence_penalty"] == 0.25

    def test_max_completion_tokens_aliases_max_tokens(self, otel_provider):
        """The OpenAI ``max_completion_tokens`` alias lands on gen_ai.request.max_tokens."""
        attrs = _span_attrs(
            otel_provider,
            lambda s: set_llm_request_attributes(s, {"max_completion_tokens": 128}),
        )
        assert attrs["gen_ai.request.max_tokens"] == 128

    @pytest.mark.parametrize(
        "params, expected",
        [
            ({"stop": "END"}, ["END"]),
            ({"stop": ["a", "b"]}, ["a", "b"]),
            ({"stop_sequences": ["x"]}, ["x"]),
            ({"stop": 123}, None),  # malformed type → skipped
            ({"stop": []}, None),  # empty list → skipped
            ({"stop": ""}, None),  # empty string → skipped
        ],
        ids=["string", "list", "stop_sequences_key", "malformed", "empty_list", "empty_string"],
    )
    def test_stop_sequences_normalization(self, otel_provider, params, expected):
        """``stop`` / ``stop_sequences`` normalize to gen_ai.request.stop_sequences:
        a string wraps to a one-element list, a list passes through, and
        empty/malformed values are skipped entirely (no empty attribute, which
        would falsely imply stop tokens were configured)."""
        attrs = _span_attrs(otel_provider, lambda s: set_llm_request_attributes(s, params))
        if expected is None:
            assert "gen_ai.request.stop_sequences" not in attrs
        else:
            assert list(attrs["gen_ai.request.stop_sequences"]) == expected

    def test_unknown_kwargs_ignored(self, otel_provider):
        """Kwargs with no gen_ai.request.* mapping are silently ignored."""
        attrs = _span_attrs(
            otel_provider,
            lambda s: set_llm_request_attributes(s, {"temperature": 0.1, "foo": "bar"}),
        )
        assert attrs["gen_ai.request.temperature"] == 0.1
        assert "foo" not in attrs

    def test_stream_true_sets_attribute(self, otel_provider):
        """``stream=True`` records gen_ai.request.stream (CR iff streaming)."""
        attrs = _span_attrs(otel_provider, lambda s: set_llm_request_attributes(s, {}, stream=True))
        assert attrs["gen_ai.request.stream"] is True

    def test_stream_default_omits_attribute(self, otel_provider):
        """The default (non-streaming) call omits gen_ai.request.stream entirely."""
        attrs = _span_attrs(otel_provider, lambda s: set_llm_request_attributes(s, {}))
        assert "gen_ai.request.stream" not in attrs

    def test_noop_when_span_none(self):
        """``span=None`` is a no-op and must not raise."""
        set_llm_request_attributes(None, {"temperature": 0.5}, stream=True)


class TestSetLlmResponseAttributes:
    def test_sets_all_response_and_usage_attributes(self, otel_provider):
        """A fully-populated response sets every response + usage attr, including
        reasoning tokens, and never emits the spec-removed total_tokens."""
        usage = UsageInfo(input_tokens=12, output_tokens=34, total_tokens=46, reasoning_tokens=7)
        attrs = _span_attrs(
            otel_provider,
            lambda s: set_llm_response_attributes(
                s,
                model="meta/llama-3.3-70b-instruct",
                response_id="chatcmpl-abc123",
                finish_reason="stop",
                usage=usage,
            ),
        )
        assert attrs["gen_ai.response.model"] == "meta/llama-3.3-70b-instruct"
        assert attrs["gen_ai.response.id"] == "chatcmpl-abc123"
        assert list(attrs["gen_ai.response.finish_reasons"]) == ["stop"]
        assert attrs["gen_ai.usage.input_tokens"] == 12
        assert attrs["gen_ai.usage.output_tokens"] == 34
        assert attrs["gen_ai.usage.reasoning.output_tokens"] == 7
        assert "gen_ai.usage.total_tokens" not in attrs

    def test_finish_reason_wrapped_in_list(self, otel_provider):
        """The single finish_reason is wrapped into the spec's finish_reasons string[]."""
        attrs = _span_attrs(
            otel_provider,
            lambda s: set_llm_response_attributes(s, finish_reason="length"),
        )
        assert list(attrs["gen_ai.response.finish_reasons"]) == ["length"]

    def test_reasoning_tokens_omitted_when_none(self, otel_provider):
        """Reasoning tokens are recorded only when present; input/output still set."""
        usage = UsageInfo(input_tokens=1, output_tokens=2, reasoning_tokens=None)
        attrs = _span_attrs(otel_provider, lambda s: set_llm_response_attributes(s, usage=usage))
        assert "gen_ai.usage.reasoning.output_tokens" not in attrs
        assert attrs["gen_ai.usage.input_tokens"] == 1
        assert attrs["gen_ai.usage.output_tokens"] == 2

    def test_usage_none_sets_no_usage_attributes(self, otel_provider):
        """``usage=None`` records no usage attrs; non-usage fields still set."""
        attrs = _span_attrs(
            otel_provider,
            lambda s: set_llm_response_attributes(s, model="m", usage=None),
        )
        assert attrs["gen_ai.response.model"] == "m"
        assert "gen_ai.usage.input_tokens" not in attrs
        assert "gen_ai.usage.output_tokens" not in attrs

    def test_omits_none_response_fields(self, otel_provider):
        """Each response field is omitted when its source value is None."""
        attrs = _span_attrs(
            otel_provider,
            lambda s: set_llm_response_attributes(s, model="only-model"),
        )
        assert attrs["gen_ai.response.model"] == "only-model"
        assert "gen_ai.response.id" not in attrs
        assert "gen_ai.response.finish_reasons" not in attrs

    def test_noop_when_span_none(self):
        """``span=None`` is a no-op and must not raise."""
        set_llm_response_attributes(None, model="m", usage=UsageInfo(input_tokens=1, output_tokens=2))

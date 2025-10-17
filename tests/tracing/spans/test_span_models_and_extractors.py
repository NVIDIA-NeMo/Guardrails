# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time

import pytest

from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.rails.llm.options import ActivatedRail, ExecutedAction
from nemoguardrails.tracing import (
    SpanEvent,
    SpanExtractorV1,
    SpanExtractorV2,
    SpanLegacy,
    SpanOpentelemetry,
    create_span_extractor,
)
from nemoguardrails.tracing.spans import LLMSpan, is_opentelemetry_span


class TestSpanModels:
    def test_span_v1_creation(self):
        span = SpanLegacy(
            span_id="test-123",
            name="test span",
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            metrics={"test_metric": 42},
        )

        assert span.span_id == "test-123"
        assert span.name == "test span"
        assert span.duration == 1.0
        assert span.metrics["test_metric"] == 42

        assert not hasattr(span, "attributes")
        assert not hasattr(span, "events")
        assert not hasattr(span, "otel_metrics")

    def test_span_v2_creation(self):
        """Test creating a v2 span - typed spans with explicit fields."""
        from nemoguardrails.tracing.spans import LLMSpan

        event = SpanEvent(
            name="gen_ai.content.prompt", timestamp=0.5, body={"content": "test prompt"}
        )

        # V2 spans are typed with explicit fields
        span = LLMSpan(
            span_id="test-456",
            name="generate_user_intent gpt-4",
            start_time=0.0,
            end_time=2.0,
            duration=2.0,
            provider_name="openai",
            request_model="gpt-4",
            response_model="gpt-4",
            operation_name="chat.completions",
            usage_input_tokens=10,
            usage_output_tokens=20,
            usage_total_tokens=30,
            events=[event],
        )

        assert span.span_id == "test-456"
        assert span.provider_name == "openai"
        assert span.request_model == "gpt-4"
        assert span.usage_input_tokens == 10
        assert len(span.events) == 1
        assert span.events[0].name == "gen_ai.content.prompt"

        # Check that to_otel_attributes works
        attributes = span.to_otel_attributes()
        assert attributes["gen_ai.provider.name"] == "openai"
        assert attributes["gen_ai.request.model"] == "gpt-4"

        assert not isinstance(span, SpanLegacy)
        # Python 3.9 compatibility: cannot use isinstance with Union types
        # SpanOpentelemetry is TypedSpan which is a Union, so check the actual type
        assert isinstance(span, LLMSpan)

    # Note: V1 and V2 spans are now fundamentally different types
    # V1 is a simple span model, V2 is typed spans with explicit fields
    # No conversion between them is needed or supported


class TestSpanExtractors:
    @pytest.fixture
    def test_data(self):
        llm_call = LLMCallInfo(
            task="generate_user_intent",
            prompt="What is the weather?",
            completion="I cannot provide weather information.",
            llm_model_name="gpt-4",
            llm_provider_name="openai",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            started_at=time.time(),
            finished_at=time.time() + 1.0,
            duration=1.0,
        )

        action = ExecutedAction(
            action_name="generate_user_intent",
            action_params={"temperature": 0.7},
            llm_calls=[llm_call],
            started_at=time.time(),
            finished_at=time.time() + 1.5,
            duration=1.5,
        )

        rail = ActivatedRail(
            type="input",
            name="check_jailbreak",
            decisions=["continue"],
            executed_actions=[action],
            stop=False,
            started_at=time.time(),
            finished_at=time.time() + 2.0,
            duration=2.0,
        )

        activated_rails = [rail]
        return {
            "activated_rails": activated_rails,
            "llm_call": llm_call,
            "action": action,
            "rail": rail,
        }

    def test_span_extractor_v1(self, test_data):
        extractor = SpanExtractorV1()
        spans = extractor.extract_spans(test_data["activated_rails"])

        assert len(spans) > 0

        for span in spans:
            assert isinstance(span, SpanLegacy)
            assert not hasattr(span, "attributes")

        span_names = [s.name for s in spans]
        assert "interaction" in span_names
        assert "rail: check_jailbreak" in span_names
        assert "action: generate_user_intent" in span_names
        assert "LLM: gpt-4" in span_names

    def test_span_extractor_v2_attributes(self, test_data):
        extractor = SpanExtractorV2()
        spans = extractor.extract_spans(test_data["activated_rails"])

        for span in spans:
            # Now we expect typed spans
            assert is_opentelemetry_span(span)

        # In V2, LLM spans follow OpenTelemetry convention: "{operation} {model}"
        llm_span = next(s for s in spans if s.name == "generate_user_intent gpt-4")
        assert isinstance(llm_span, LLMSpan)

        # For typed spans, check the fields directly
        assert llm_span.provider_name == "openai"
        assert llm_span.request_model == "gpt-4"
        assert llm_span.usage_input_tokens == 10

        # Also verify attributes conversion works
        attributes = llm_span.to_otel_attributes()
        assert "gen_ai.provider.name" in attributes
        assert attributes["gen_ai.provider.name"] == "openai"
        assert attributes["gen_ai.request.model"] == "gpt-4"
        assert "gen_ai.usage.input_tokens" in attributes
        assert attributes["gen_ai.usage.input_tokens"] == 10

    def test_span_extractor_v2_events(self, test_data):
        extractor = SpanExtractorV2(enable_content_capture=True)
        spans = extractor.extract_spans(test_data["activated_rails"])

        # In V2, LLM spans follow OpenTelemetry convention: "{operation} {model}"
        llm_span = next(s for s in spans if s.name == "generate_user_intent gpt-4")
        assert len(llm_span.events) > 0

        event_names = [e.name for e in llm_span.events]
        # V2 currently uses deprecated content events for simplicity (TODO: update to newer format)
        assert "gen_ai.content.prompt" in event_names
        assert "gen_ai.content.completion" in event_names

        # Check user message event content (only present when content capture is enabled)
        user_message_event = next(
            e for e in llm_span.events if e.name == "gen_ai.content.prompt"
        )
        assert user_message_event.body["content"] == "What is the weather?"

    def test_span_extractor_v2_metrics(self, test_data):
        extractor = SpanExtractorV2()
        spans = extractor.extract_spans(test_data["activated_rails"])

        # In V2, LLM spans follow OpenTelemetry convention: "{operation} {model}"
        llm_span = next(s for s in spans if s.name == "generate_user_intent gpt-4")
        assert isinstance(llm_span, LLMSpan)

        # Check typed fields
        assert llm_span.usage_input_tokens == 10
        assert llm_span.usage_output_tokens == 20
        assert llm_span.usage_total_tokens == 30
        assert llm_span.provider_name == "openai"

        # Verify attributes conversion
        attributes = llm_span.to_otel_attributes()
        assert attributes["gen_ai.usage.total_tokens"] == 30
        assert attributes["gen_ai.provider.name"] == "openai"

    def test_span_extractor_v2_conversation_events(self, test_data):
        internal_events = [
            {"type": "UtteranceUserActionFinished", "final_transcript": "Hello bot"},
            {"type": "StartUtteranceBotAction", "script": "Hello! How can I help?"},
            {"type": "SystemMessage", "content": "You are a helpful assistant"},
        ]

        # Test with content excluded by default (privacy compliant)
        extractor = SpanExtractorV2(events=internal_events)
        spans = extractor.extract_spans(test_data["activated_rails"])

        interaction_span = next(s for s in spans if s.name == "guardrails.request")
        assert len(interaction_span.events) > 0

        event_names = [e.name for e in interaction_span.events]
        # These are guardrails internal events, not OTel GenAI events
        assert "guardrails.utterance.user.finished" in event_names
        assert "guardrails.utterance.bot.started" in event_names

        user_event = next(
            e
            for e in interaction_span.events
            if e.name == "guardrails.utterance.user.finished"
        )
        # By default, content is NOT included (privacy compliant)
        assert "type" in user_event.body
        assert "final_transcript" not in user_event.body


class TestSpanVersionConfiguration:
    def test_create_span_extractor_legacy(self):
        extractor = create_span_extractor(span_format="legacy")
        assert isinstance(extractor, SpanExtractorV1)

    def test_create_span_extractor_opentelemetry(self):
        extractor = create_span_extractor(span_format="opentelemetry")
        assert isinstance(extractor, SpanExtractorV2)

    def test_create_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid span format"):
            create_span_extractor(span_format="invalid")

    def test_opentelemetry_extractor_with_events(self):
        events = [{"type": "UserMessage", "text": "test"}]
        extractor = create_span_extractor(
            span_format="opentelemetry", events=events, enable_content_capture=False
        )

        assert isinstance(extractor, SpanExtractorV2)
        assert extractor.internal_events == events

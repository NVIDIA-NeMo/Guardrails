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
    SpanExtractorV1,
    SpanExtractorV2,
    SpanLegacy,
    create_span_extractor,
)
from nemoguardrails.tracing.constants import GuardrailsAttributes
from nemoguardrails.tracing.spans import LLMSpan, is_opentelemetry_span


class TestSpanExtractors:
    """Test span extraction for legacy and OpenTelemetry formats."""

    @pytest.fixture
    def test_data(self):
        """Set up test data for span extraction."""
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

        return [rail]

    def test_span_extractor_legacy_format(self, test_data):
        """Test legacy format span extractor produces legacy spans."""
        extractor = SpanExtractorV1()
        spans = extractor.extract_spans(test_data)

        assert len(spans) > 0

        # All spans should be legacy format
        for span in spans:
            assert isinstance(span, SpanLegacy)
            assert not hasattr(span, "attributes")

        span_names = [s.name for s in spans]
        assert "interaction" in span_names
        assert "rail: check_jailbreak" in span_names
        assert "action: generate_user_intent" in span_names
        assert "LLM: gpt-4" in span_names

    def test_span_extractor_opentelemetry_attributes(self, test_data):
        """Test OpenTelemetry span extractor adds semantic convention attributes."""
        extractor = SpanExtractorV2()
        spans = extractor.extract_spans(test_data)

        # All spans should be typed spans
        for span in spans:
            assert is_opentelemetry_span(span)

        # LLM spans follow OpenTelemetry convention: "{operation} {model}"
        llm_span = next(s for s in spans if s.name == "generate_user_intent gpt-4")
        assert isinstance(llm_span, LLMSpan)

        assert llm_span.provider_name == "openai"
        assert llm_span.request_model == "gpt-4"
        assert llm_span.usage_input_tokens == 10

        attributes = llm_span.to_otel_attributes()
        assert "gen_ai.provider.name" in attributes
        assert attributes["gen_ai.provider.name"] == "openai"
        assert attributes["gen_ai.request.model"] == "gpt-4"
        assert "gen_ai.usage.input_tokens" in attributes
        assert attributes["gen_ai.usage.input_tokens"] == 10

    def test_span_extractor_opentelemetry_events(self, test_data):
        """Test OpenTelemetry span extractor adds events."""
        extractor = SpanExtractorV2(enable_content_capture=True)
        spans = extractor.extract_spans(test_data)

        # LLM spans follow OpenTelemetry convention
        llm_span = next(s for s in spans if s.name == "generate_user_intent gpt-4")
        assert len(llm_span.events) > 0

        event_names = [e.name for e in llm_span.events]
        # Currently uses deprecated content events (TODO: update to newer format)
        assert "gen_ai.content.prompt" in event_names
        assert "gen_ai.content.completion" in event_names

        # Check event content (only present when content capture is enabled)
        user_message_event = next(
            e for e in llm_span.events if e.name == "gen_ai.content.prompt"
        )
        assert user_message_event.body["content"] == "What is the weather?"

    def test_span_extractor_opentelemetry_metrics(self, test_data):
        """Test OpenTelemetry span extractor adds metrics as attributes."""
        extractor = SpanExtractorV2()
        spans = extractor.extract_spans(test_data)

        llm_span = next(s for s in spans if s.name == "generate_user_intent gpt-4")
        assert isinstance(llm_span, LLMSpan)

        assert llm_span.usage_input_tokens == 10
        assert llm_span.usage_output_tokens == 20
        assert llm_span.usage_total_tokens == 30

        attributes = llm_span.to_otel_attributes()
        assert "gen_ai.usage.input_tokens" in attributes
        assert "gen_ai.usage.output_tokens" in attributes
        assert "gen_ai.usage.total_tokens" in attributes

        assert attributes["gen_ai.usage.input_tokens"] == 10
        assert attributes["gen_ai.usage.output_tokens"] == 20
        assert attributes["gen_ai.usage.total_tokens"] == 30
        assert attributes["gen_ai.provider.name"] == "openai"

    def test_span_extractor_conversation_events(self, test_data):
        """Test OpenTelemetry span extractor extracts conversation events from internal events."""
        internal_events = [
            {"type": "UtteranceUserActionFinished", "final_transcript": "Hello bot"},
            {"type": "StartUtteranceBotAction", "script": "Hello! How can I help?"},
            {"type": "SystemMessage", "content": "You are a helpful assistant"},
        ]

        extractor = SpanExtractorV2(events=internal_events)
        spans = extractor.extract_spans(test_data)

        interaction_span = next(s for s in spans if s.name == "guardrails.request")
        assert len(interaction_span.events) > 0

        event_names = [e.name for e in interaction_span.events]
        assert "guardrails.utterance.user.finished" in event_names
        assert "guardrails.utterance.bot.started" in event_names

        user_event = next(
            e
            for e in interaction_span.events
            if e.name == "guardrails.utterance.user.finished"
        )
        assert "type" in user_event.body
        # Content not included by default (privacy)
        assert "final_transcript" not in user_event.body

    def test_span_extractor_cache_hit_attribute(self):
        """Test that cached LLM calls are marked with cache_hit typed field."""
        llm_call_cached = LLMCallInfo(
            task="generate_user_intent",
            prompt="What is the weather?",
            completion="I cannot provide weather information.",
            llm_model_name="gpt-4",
            llm_provider_name="openai",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            started_at=time.time(),
            finished_at=time.time() + 0.001,
            duration=0.001,
            from_cache=True,
        )

        llm_call_not_cached = LLMCallInfo(
            task="generate_bot_message",
            prompt="Generate a response",
            completion="Here is a response",
            llm_model_name="gpt-3.5-turbo",
            llm_provider_name="openai",
            prompt_tokens=5,
            completion_tokens=15,
            total_tokens=20,
            started_at=time.time(),
            finished_at=time.time() + 1.0,
            duration=1.0,
            from_cache=False,
        )

        action = ExecutedAction(
            action_name="test_action",
            action_params={},
            llm_calls=[llm_call_cached, llm_call_not_cached],
            started_at=time.time(),
            finished_at=time.time() + 1.5,
            duration=1.5,
        )

        rail = ActivatedRail(
            type="input",
            name="test_rail",
            decisions=["continue"],
            executed_actions=[action],
            stop=False,
            started_at=time.time(),
            finished_at=time.time() + 2.0,
            duration=2.0,
        )

        extractor = SpanExtractorV2()
        spans = extractor.extract_spans([rail])

        llm_spans = [s for s in spans if isinstance(s, LLMSpan)]
        assert len(llm_spans) == 2

        cached_span = next(s for s in llm_spans if "gpt-4" in s.name)
        assert cached_span.cache_hit is True

        attributes = cached_span.to_otel_attributes()
        assert GuardrailsAttributes.LLM_CACHE_HIT in attributes
        assert attributes[GuardrailsAttributes.LLM_CACHE_HIT] is True

        not_cached_span = next(s for s in llm_spans if "gpt-3.5-turbo" in s.name)
        assert not_cached_span.cache_hit is False

        attributes = not_cached_span.to_otel_attributes()
        assert GuardrailsAttributes.LLM_CACHE_HIT in attributes
        assert attributes[GuardrailsAttributes.LLM_CACHE_HIT] is False


class TestSpanFormatConfiguration:
    """Test span format configuration and factory."""

    def test_create_span_extractor_legacy(self):
        """Test creating legacy format span extractor."""
        extractor = create_span_extractor(span_format="legacy")
        assert isinstance(extractor, SpanExtractorV1)

    def test_create_span_extractor_opentelemetry(self):
        """Test creating OpenTelemetry format span extractor."""
        extractor = create_span_extractor(span_format="opentelemetry")
        assert isinstance(extractor, SpanExtractorV2)

    def test_create_invalid_format_raises_error(self):
        """Test invalid span format raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_span_extractor(span_format="invalid")
        assert "Invalid span format" in str(exc_info.value)

    def test_opentelemetry_extractor_with_events(self):
        """Test OpenTelemetry extractor can be created with events."""
        events = [{"type": "UserMessage", "text": "test"}]
        extractor = create_span_extractor(
            span_format="opentelemetry", events=events, enable_content_capture=False
        )

        assert isinstance(extractor, SpanExtractorV2)
        assert extractor.internal_events == events

    def test_legacy_extractor_ignores_extra_params(self):
        """Test legacy extractor ignores OpenTelemetry-specific parameters."""
        # Legacy extractor should ignore events and enable_content_capture
        extractor = create_span_extractor(
            span_format="legacy", events=[{"type": "test"}], enable_content_capture=True
        )

        assert isinstance(extractor, SpanExtractorV1)
        # V1 extractor doesn't have these attributes
        assert not hasattr(extractor, "internal_events")
        assert not hasattr(extractor, "enable_content_capture")

    @pytest.mark.parametrize(
        "format_str,expected_class",
        [
            ("legacy", SpanExtractorV1),
            ("LEGACY", SpanExtractorV1),
            ("opentelemetry", SpanExtractorV2),
            ("OPENTELEMETRY", SpanExtractorV2),
            ("OpenTelemetry", SpanExtractorV2),
        ],
    )
    def test_case_insensitive_format(self, format_str, expected_class):
        """Test that span format is case-insensitive."""
        extractor = create_span_extractor(span_format=format_str)
        assert isinstance(extractor, expected_class)

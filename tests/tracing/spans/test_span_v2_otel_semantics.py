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

"""Tests for SpanOpentelemetry with complete OpenTelemetry semantic convention attributes."""

from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails.rails.llm.options import ActivatedRail, ExecutedAction, LLMCallInfo
from nemoguardrails.tracing.constants import (
    CommonAttributes,
    EventNames,
    GenAIAttributes,
    GuardrailsAttributes,
    OperationNames,
    SpanKind,
    SpanNames,
)
from nemoguardrails.tracing.span_extractors import SpanExtractorV2
from nemoguardrails.tracing.spans import ActionSpan, InteractionSpan, LLMSpan, RailSpan


class TestSpanOpentelemetryOTelAttributes:
    """Test that SpanV2 contains complete OTel semantic convention attributes."""

    def test_interaction_span_has_complete_attributes(self):
        """Test that interaction span has all required OTel attributes."""
        rail = ActivatedRail(
            type="input",
            name="check_jailbreak",
            started_at=1.0,
            finished_at=2.0,
            duration=1.0,
            executed_actions=[],
        )

        extractor = SpanExtractorV2()
        spans = extractor.extract_spans([rail])

        interaction_span = next(s for s in spans if s.parent_id is None)
        assert isinstance(interaction_span, InteractionSpan)

        attrs = interaction_span.to_otel_attributes()
        assert attrs[CommonAttributes.SPAN_KIND] == SpanKind.SERVER
        assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == OperationNames.GUARDRAILS
        assert "service.name" in attrs
        assert interaction_span.name == SpanNames.GUARDRAILS_REQUEST

        assert GenAIAttributes.GEN_AI_PROVIDER_NAME not in attrs
        assert GenAIAttributes.GEN_AI_SYSTEM not in attrs

    def test_rail_span_has_complete_attributes(self):
        """Test that rail spans have all required attributes."""
        rail = ActivatedRail(
            type="input",
            name="check_jailbreak",
            started_at=1.0,
            finished_at=2.0,
            duration=1.0,
            stop=True,
            decisions=["blocked"],
            executed_actions=[],
        )

        extractor = SpanExtractorV2()
        spans = extractor.extract_spans([rail])

        rail_span = next(s for s in spans if s.name == SpanNames.GUARDRAILS_RAIL)
        assert isinstance(rail_span, RailSpan)

        attrs = rail_span.to_otel_attributes()
        assert attrs[CommonAttributes.SPAN_KIND] == SpanKind.INTERNAL
        assert attrs[GuardrailsAttributes.RAIL_TYPE] == "input"
        assert attrs[GuardrailsAttributes.RAIL_NAME] == "check_jailbreak"
        assert attrs[GuardrailsAttributes.RAIL_STOP] is True
        assert attrs[GuardrailsAttributes.RAIL_DECISIONS] == ["blocked"]

    def test_llm_span_has_complete_attributes(self):
        """Test that LLM spans have all required OTel GenAI attributes."""
        llm_call = LLMCallInfo(
            task="generate",
            llm_model_name="gpt-4",
            llm_provider_name="openai",
            prompt="Hello, world!",
            completion="Hi there!",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            started_at=1.5,
            finished_at=1.8,
            duration=0.3,
            raw_response={
                "id": "chatcmpl-123",
                "choices": [{"finish_reason": "stop"}],
                "temperature": 0.7,
                "max_tokens": 100,
                "top_p": 0.9,
            },
        )

        action = ExecutedAction(
            action_name="generate_user_intent",
            started_at=1.0,
            finished_at=2.0,
            duration=1.0,
            llm_calls=[llm_call],
        )

        rail = ActivatedRail(
            type="dialog",
            name="generate_next_step",
            started_at=1.0,
            finished_at=2.0,
            duration=1.0,
            executed_actions=[action],
        )

        extractor = SpanExtractorV2()
        spans = extractor.extract_spans([rail])

        llm_span = next(s for s in spans if "gpt-4" in s.name)
        assert isinstance(llm_span, LLMSpan)

        attrs = llm_span.to_otel_attributes()
        assert attrs[CommonAttributes.SPAN_KIND] == SpanKind.CLIENT
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"
        assert attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4"
        assert attrs[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gpt-4"
        assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "generate"
        assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 10
        assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 5
        assert attrs[GenAIAttributes.GEN_AI_USAGE_TOTAL_TOKENS] == 15
        assert attrs[GenAIAttributes.GEN_AI_RESPONSE_ID] == "chatcmpl-123"
        assert attrs[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]
        assert attrs[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.7
        assert attrs[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 100
        assert attrs[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.9

        assert GenAIAttributes.GEN_AI_SYSTEM not in attrs

    def test_llm_span_events_are_complete(self):
        """Test that LLM span events follow OTel GenAI conventions."""
        llm_call = LLMCallInfo(
            task="chat",
            llm_model_name="claude-3",
            prompt="What is the weather?",
            completion="I cannot access real-time weather data.",
            started_at=1.5,
            finished_at=1.8,
            duration=0.3,
            raw_response={"stop_reason": "end_turn"},
        )

        action = ExecutedAction(
            action_name="llm_generate",
            started_at=1.0,
            finished_at=2.0,
            duration=1.0,
            llm_calls=[llm_call],
        )

        rail = ActivatedRail(
            type="dialog",
            name="chat",
            started_at=1.0,
            finished_at=2.0,
            duration=1.0,
            executed_actions=[action],
        )

        extractor = SpanExtractorV2(enable_content_capture=True)
        spans = extractor.extract_spans([rail])

        llm_span = next(s for s in spans if "claude" in s.name)
        assert isinstance(llm_span, LLMSpan)

        assert len(llm_span.events) >= 2  # at least user and assistant messages

        user_event = next(e for e in llm_span.events if e.name == EventNames.GEN_AI_CONTENT_PROMPT)
        assert user_event.body["content"] == "What is the weather?"

        assistant_event = next(e for e in llm_span.events if e.name == EventNames.GEN_AI_CONTENT_COMPLETION)
        assert assistant_event.body["content"] == "I cannot access real-time weather data."

        finish_events = [e for e in llm_span.events if e.name == "gen_ai.choice.finish"]
        if finish_events:
            finish_event = finish_events[0]
            assert "finish_reason" in finish_event.body
            assert "index" in finish_event.body

    def test_action_span_with_error_attributes(self):
        """Test that action spans include error information when present."""
        # TODO: Figure out how errors are properly attached to actions
        action = ExecutedAction(
            action_name="failed_action",
            started_at=1.0,
            finished_at=2.0,
            duration=1.0,
            llm_calls=[],
        )
        # skip setting error for now since ExecutedAction doesn't have that field
        # action.error = ValueError("Something went wrong")

        rail = ActivatedRail(
            type="input",
            name="check_input",
            started_at=1.0,
            finished_at=2.0,
            duration=1.0,
            executed_actions=[action],
        )

        extractor = SpanExtractorV2()
        spans = extractor.extract_spans([rail])

        action_span = next(s for s in spans if s.name == SpanNames.GUARDRAILS_ACTION)
        assert isinstance(action_span, ActionSpan)

        attrs = action_span.to_otel_attributes()
        # since we didn't set an error, these shouldn't be present
        assert "error" not in attrs or attrs["error"] is None
        assert "error.type" not in attrs
        assert "error.message" not in attrs

    def test_span_names_are_low_cardinality(self):
        """Test that span names follow low-cardinality convention."""
        rails = [
            ActivatedRail(
                type="input",
                name=f"rail_{i}",
                started_at=float(i),
                finished_at=float(i + 1),
                duration=1.0,
                executed_actions=[
                    ExecutedAction(
                        action_name=f"action_{i}",
                        started_at=float(i),
                        finished_at=float(i + 1),
                        duration=1.0,
                        llm_calls=[
                            LLMCallInfo(
                                task=f"task_{i}",
                                llm_model_name=f"model_{i}",
                                started_at=float(i),
                                finished_at=float(i + 1),
                                duration=1.0,
                            )
                        ],
                    )
                ],
            )
            for i in range(3)
        ]

        extractor = SpanExtractorV2()
        all_spans = []
        for rail in rails:
            spans = extractor.extract_spans([rail])
            all_spans.extend(spans)

        expected_patterns = {
            SpanNames.GUARDRAILS_REQUEST,
            SpanNames.GUARDRAILS_RAIL,
            SpanNames.GUARDRAILS_ACTION,
        }

        for span in all_spans:
            if not any(f"model_{i}" in span.name for i in range(3)):
                assert span.name in expected_patterns

        rail_spans = [s for s in all_spans if s.name == SpanNames.GUARDRAILS_RAIL]
        rail_names = {s.to_otel_attributes()[GuardrailsAttributes.RAIL_NAME] for s in rail_spans}
        assert len(rail_names) == 3

    def test_no_semantic_logic_in_adapter(self):
        """Verify adapter is just an API bridge by checking it doesn't modify attributes."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from nemoguardrails.tracing import InteractionLog
        from nemoguardrails.tracing.adapters.opentelemetry import OpenTelemetryAdapter

        # create a mock exporter to capture spans
        class MockExporter:
            def __init__(self):
                self.spans = []

            def export(self, spans):
                self.spans.extend(spans)
                return 0

            def shutdown(self):
                pass

        # setup OTel
        exporter = MockExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        # create adapter
        adapter = OpenTelemetryAdapter()

        # create a simple rail
        rail = ActivatedRail(
            type="input",
            name="test_rail",
            started_at=1.0,
            finished_at=2.0,
            duration=1.0,
            executed_actions=[],
        )

        # extract spans with V2 extractor
        extractor = SpanExtractorV2()
        spans = extractor.extract_spans([rail])

        # create interaction log
        interaction_log = InteractionLog(
            id="test-trace-123",
            activated_rails=[rail],
            trace=spans,
        )

        # transform through adapter
        adapter.transform(interaction_log)

        assert len(exporter.spans) > 0

        for otel_span in exporter.spans:
            attrs = dict(otel_span.attributes)

            if otel_span.name == SpanNames.GUARDRAILS_REQUEST:
                assert GenAIAttributes.GEN_AI_OPERATION_NAME in attrs
                assert GenAIAttributes.GEN_AI_PROVIDER_NAME not in attrs
                assert GenAIAttributes.GEN_AI_SYSTEM not in attrs


class TestOpenTelemetryAdapterAsTheBridge:
    """Test that OpenTelemetryAdapter is a pure API bridge."""

    def test_adapter_handles_span_kind_mapping(self):
        """Test that adapter correctly maps span.kind string to OTel enum."""
        from opentelemetry.trace import SpanKind as OTelSpanKind

        from nemoguardrails.tracing import InteractionLog
        from nemoguardrails.tracing.adapters.opentelemetry import OpenTelemetryAdapter

        # mock provider to capture span creation
        created_spans = []

        class MockTracer:
            def start_span(self, name, context=None, start_time=None, kind=None):
                created_spans.append({"name": name, "kind": kind})
                return MagicMock()

        provider = MagicMock()
        provider.get_tracer = MagicMock(return_value=MockTracer())

        with patch("opentelemetry.trace.get_tracer_provider", return_value=provider):
            adapter = OpenTelemetryAdapter()

            spans = [
                InteractionSpan(
                    span_id="1",
                    name="server_span",
                    start_time=0.0,
                    end_time=1.0,
                    duration=1.0,
                ),
                LLMSpan(
                    span_id="2",
                    name="client_span",
                    start_time=0.0,
                    end_time=1.0,
                    duration=1.0,
                    provider_name="openai",
                    request_model="gpt-4",
                    response_model="gpt-4",
                    operation_name="chat.completions",
                ),
                RailSpan(
                    span_id="3",
                    name="internal_span",
                    start_time=0.0,
                    end_time=1.0,
                    duration=1.0,
                    rail_type="input",
                    rail_name="test_rail",
                ),
            ]

            interaction_log = InteractionLog(
                id="test-123",
                activated_rails=[],
                trace=spans,
            )

            adapter.transform(interaction_log)

            assert created_spans[0]["kind"] == OTelSpanKind.SERVER
            assert created_spans[1]["kind"] == OTelSpanKind.CLIENT
            assert created_spans[2]["kind"] == OTelSpanKind.INTERNAL


class TestContentPrivacy:
    """Test that sensitive content is handled according to OTel GenAI conventions."""

    def test_content_not_included_by_default(self):
        """Test that content is NOT included by default per OTel spec."""
        events = [
            {"type": "UserMessage", "text": "My SSN is 123-45-6789"},
            {
                "type": "UtteranceBotActionFinished",
                "final_script": "I cannot process SSN",
            },
        ]
        extractor = SpanExtractorV2(events=events, enable_content_capture=False)

        activated_rail = ActivatedRail(
            type="action",
            name="generate",
            started_at=0.0,
            finished_at=1.0,
            duration=1.0,
            executed_actions=[
                ExecutedAction(
                    action_name="generate",
                    started_at=0.0,
                    finished_at=1.0,
                    duration=1.0,
                    llm_calls=[
                        LLMCallInfo(
                            task="general",
                            prompt="User sensitive prompt",
                            completion="Bot response with PII",
                            duration=0.5,
                            total_tokens=100,
                            prompt_tokens=50,
                            completion_tokens=50,
                            raw_response={"model": "gpt-3.5-turbo"},
                        )
                    ],
                )
            ],
        )

        spans = extractor.extract_spans([activated_rail])

        llm_span = next((s for s in spans if isinstance(s, LLMSpan)), None)
        assert llm_span is not None

        for event in llm_span.events:
            if event.name in ["gen_ai.content.prompt", "gen_ai.content.completion"]:
                assert event.body == {}
                assert "content" not in event.body

    def test_content_included_when_explicitly_enabled(self):
        """Test that content IS included when explicitly enabled."""
        # Create extractor with enable_content_capture=True
        events = [
            {"type": "UserMessage", "text": "Hello bot"},
            {"type": "UtteranceBotActionFinished", "final_script": "Hello user"},
        ]
        extractor = SpanExtractorV2(events=events, enable_content_capture=True)

        activated_rail = ActivatedRail(
            type="action",
            name="generate",
            started_at=0.0,
            finished_at=1.0,
            duration=1.0,
            executed_actions=[
                ExecutedAction(
                    action_name="generate",
                    started_at=0.0,
                    finished_at=1.0,
                    duration=1.0,
                    llm_calls=[
                        LLMCallInfo(
                            task="general",
                            prompt="Test prompt",
                            completion="Test response",
                            duration=0.5,
                            total_tokens=100,
                            prompt_tokens=50,
                            completion_tokens=50,
                            raw_response={"model": "gpt-3.5-turbo"},
                        )
                    ],
                )
            ],
        )

        spans = extractor.extract_spans([activated_rail])

        llm_span = next((s for s in spans if isinstance(s, LLMSpan)), None)
        assert llm_span is not None

        prompt_event = next((e for e in llm_span.events if e.name == "gen_ai.content.prompt"), None)
        assert prompt_event is not None
        assert prompt_event.body.get("content") == "Test prompt"

        completion_event = next((e for e in llm_span.events if e.name == "gen_ai.content.completion"), None)
        assert completion_event is not None
        assert completion_event.body.get("content") == "Test response"

    def test_conversation_events_respect_privacy_setting(self):
        """Test that guardrails internal events respect the privacy setting."""
        events = [
            {"type": "UserMessage", "text": "Private message"},
            {
                "type": "UtteranceBotActionFinished",
                "final_script": "Private response",
                "is_success": True,
            },
        ]

        extractor_no_content = SpanExtractorV2(events=events, enable_content_capture=False)
        activated_rail = ActivatedRail(type="dialog", name="main", started_at=0.0, finished_at=1.0, duration=1.0)

        spans = extractor_no_content.extract_spans([activated_rail])
        interaction_span = spans[0]  # First span is the interaction span

        user_event = next(
            (e for e in interaction_span.events if e.name == "guardrails.user_message"),
            None,
        )
        assert user_event is not None
        assert user_event.body["type"] == "UserMessage"
        assert "content" not in user_event.body

        bot_event = next(
            (e for e in interaction_span.events if e.name == "guardrails.utterance.bot.finished"),
            None,
        )
        assert bot_event is not None
        assert bot_event.body["type"] == "UtteranceBotActionFinished"
        assert bot_event.body["is_success"]
        assert "content" not in bot_event.body  # Content excluded

        extractor_with_content = SpanExtractorV2(events=events, enable_content_capture=True)
        spans = extractor_with_content.extract_spans([activated_rail])
        interaction_span = spans[0]

        user_event = next(
            (e for e in interaction_span.events if e.name == "guardrails.user_message"),
            None,
        )
        assert user_event is not None
        assert user_event.body.get("content") == "Private message"

        bot_event = next(
            (e for e in interaction_span.events if e.name == "guardrails.utterance.bot.finished"),
            None,
        )
        assert bot_event is not None
        assert bot_event.body.get("content") == "Private response"
        assert bot_event.body.get("type") == "UtteranceBotActionFinished"
        assert bot_event.body.get("is_success")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

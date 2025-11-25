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


from nemoguardrails.tracing import SpanEvent, SpanLegacy
from nemoguardrails.tracing.spans import LLMSpan


class TestSpanModels:
    """Test the span models for legacy and OpenTelemetry formats."""

    def test_span_legacy_creation(self):
        """Test creating a legacy format span."""
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

        # Legacy spans don't have OpenTelemetry attributes
        assert not hasattr(span, "attributes")
        assert not hasattr(span, "events")
        assert not hasattr(span, "otel_metrics")

    def test_span_opentelemetry_creation(self):
        """Test creating an OpenTelemetry format span - typed spans with explicit fields."""
        event = SpanEvent(name="gen_ai.content.prompt", timestamp=0.5, body={"content": "test prompt"})

        # OpenTelemetry spans are typed with explicit fields
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

        attributes = span.to_otel_attributes()
        assert attributes["gen_ai.provider.name"] == "openai"
        assert attributes["gen_ai.request.model"] == "gpt-4"

    def test_span_legacy_model_is_simple(self):
        """Test that Legacy span model is a simple span without OpenTelemetry features."""
        legacy_span = SpanLegacy(
            span_id="legacy-123",
            name="test",
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            metrics={"metric": 1},
        )

        assert isinstance(legacy_span, SpanLegacy)
        assert legacy_span.span_id == "legacy-123"
        assert legacy_span.metrics["metric"] == 1

        # Legacy spans don't have OpenTelemetry attributes or events
        assert not hasattr(legacy_span, "attributes")
        assert not hasattr(legacy_span, "events")

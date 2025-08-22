# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest
from unittest.mock import MagicMock, patch

from nemoguardrails.tracing import (
    InteractionLog,
    SpanEvent,
    SpanLegacy,
    SpanOpentelemetry,
)
from nemoguardrails.tracing.adapters.opentelemetry import OpenTelemetryAdapter
from nemoguardrails.tracing.spans import InteractionSpan, LLMSpan


class TestOpenTelemetryAdapterV2(unittest.TestCase):
    """Test OpenTelemetryAdapter handling of v2 spans."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the tracer
        self.mock_tracer = MagicMock()
        self.mock_tracer_provider = MagicMock()
        self.mock_tracer_provider.get_tracer.return_value = self.mock_tracer

        # Patch trace.get_tracer_provider
        patcher = patch("opentelemetry.trace.get_tracer_provider")
        self.mock_get_tracer_provider = patcher.start()
        self.mock_get_tracer_provider.return_value = self.mock_tracer_provider
        self.addCleanup(patcher.stop)

        self.adapter = OpenTelemetryAdapter()

    def test_v1_span_compatibility(self):
        """Test that v1 spans still work correctly."""
        mock_span = MagicMock()
        self.mock_tracer.start_span.return_value = mock_span

        v1_span = SpanLegacy(
            name="test_v1",
            span_id="v1_123",
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            metrics={"metric1": 42},
        )

        interaction_log = InteractionLog(
            id="test_v1_log", activated_rails=[], events=[], trace=[v1_span]
        )

        self.adapter.transform(interaction_log)

        # Verify span was created
        self.mock_tracer.start_span.assert_called_once()

        # Verify metrics were set as attributes without prefix
        mock_span.set_attribute.assert_any_call("metric1", 42)

        # Should not try to add events
        mock_span.add_event.assert_not_called()

    def test_v2_span_attributes(self):
        """Test that v2 span attributes are properly handled."""
        mock_span = MagicMock()
        self.mock_tracer.start_span.return_value = mock_span

        from nemoguardrails.tracing.spans import LLMSpan

        v2_span = LLMSpan(
            name="LLM: gpt-4",
            span_id="v2_123",
            start_time=0.0,
            end_time=2.0,
            duration=2.0,
            provider_name="openai",
            request_model="gpt-4",
            response_model="gpt-4",
            operation_name="chat.completions",
            usage_total_tokens=150,
            custom_attributes={
                "rail.decisions": ["continue", "allow"],  # List attribute in custom
            },
        )

        interaction_log = InteractionLog(
            id="test_v2_log", activated_rails=[], events=[], trace=[v2_span]
        )

        self.adapter.transform(interaction_log)

        # Verify OpenTelemetry attributes were set
        mock_span.set_attribute.assert_any_call("gen_ai.provider.name", "openai")
        mock_span.set_attribute.assert_any_call("gen_ai.request.model", "gpt-4")
        mock_span.set_attribute.assert_any_call("gen_ai.usage.total_tokens", 150)

        # Verify list was passed directly
        # Note: OTel Python SDK automatically converts lists to strings
        mock_span.set_attribute.assert_any_call("rail.decisions", ["continue", "allow"])

    def test_v2_span_events(self):
        """Test that v2 span events are properly added."""
        mock_span = MagicMock()
        self.mock_tracer.start_span.return_value = mock_span

        events = [
            SpanEvent(
                name="gen_ai.content.prompt",
                timestamp=0.5,
                body={"content": "What is AI?"},
            ),
            SpanEvent(
                name="gen_ai.content.completion",
                timestamp=1.5,
                body={"content": "AI stands for Artificial Intelligence..."},
            ),
            SpanEvent(
                name="gen_ai.choice",
                timestamp=1.6,
                body={"finish_reason": "stop", "index": 0},
            ),
        ]

        v2_span = LLMSpan(
            name="LLM: gpt-4",
            span_id="v2_events",
            start_time=0.0,
            end_time=2.0,
            duration=2.0,
            provider_name="openai",
            request_model="gpt-4",
            response_model="gpt-4",
            operation_name="chat.completions",
            events=events,
        )

        interaction_log = InteractionLog(
            id="test_events", activated_rails=[], events=[], trace=[v2_span]
        )

        self.adapter.transform(interaction_log)

        # Verify events were added
        self.assertEqual(mock_span.add_event.call_count, 3)

        # Check first event (prompt)
        call_args = mock_span.add_event.call_args_list[0]
        self.assertEqual(call_args[1]["name"], "gen_ai.content.prompt")
        # In new implementation, body content is merged directly into attributes
        self.assertIn("content", call_args[1]["attributes"])
        self.assertEqual(call_args[1]["attributes"]["content"], "What is AI?")

        # Check choice event has finish reason
        call_args = mock_span.add_event.call_args_list[2]
        self.assertEqual(call_args[1]["name"], "gen_ai.choice")
        # In new implementation, body fields are merged directly into attributes
        self.assertIn("finish_reason", call_args[1]["attributes"])

    def test_v2_span_metrics(self):
        """Test that v2 span token usage is properly recorded as attributes."""
        mock_span = MagicMock()
        self.mock_tracer.start_span.return_value = mock_span

        # In the new implementation, token usage is in attributes, not otel_metrics
        v2_span = LLMSpan(
            name="completion gpt-4",  # Following new naming convention
            span_id="v2_metrics",
            start_time=0.0,
            end_time=2.0,
            duration=2.0,
            provider_name="openai",
            request_model="gpt-4",
            response_model="gpt-4",
            operation_name="completion",
            usage_input_tokens=50,
            usage_output_tokens=100,
            usage_total_tokens=150,
        )

        interaction_log = InteractionLog(
            id="test_metrics", activated_rails=[], events=[], trace=[v2_span]
        )

        self.adapter.transform(interaction_log)

        # Verify token usage is recorded as standard attributes per OpenTelemetry GenAI conventions
        mock_span.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 50)
        mock_span.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 100)
        mock_span.set_attribute.assert_any_call("gen_ai.usage.total_tokens", 150)
        mock_span.set_attribute.assert_any_call("gen_ai.provider.name", "openai")
        mock_span.set_attribute.assert_any_call("gen_ai.request.model", "gpt-4")

    def test_mixed_v1_v2_spans(self):
        """Test handling of mixed v1 and v2 spans in the same trace."""
        mock_span = MagicMock()
        self.mock_tracer.start_span.return_value = mock_span

        v1_span = SpanLegacy(
            name="action: check_input",
            span_id="v1_span",
            start_time=0.0,
            end_time=0.5,
            duration=0.5,
            metrics={"action_total": 1},  # Will be set as action_total (no prefix)
        )

        v2_span = LLMSpan(
            name="LLM: gpt-4",
            span_id="v2_span",
            parent_id="v1_span",
            start_time=0.1,
            end_time=0.4,
            duration=0.3,
            provider_name="openai",
            request_model="gpt-4",
            response_model="gpt-4",
            operation_name="chat.completions",
            events=[
                SpanEvent(
                    name="gen_ai.content.prompt",
                    timestamp=0.1,
                    body={"content": "test"},
                )
            ],
        )

        interaction_log = InteractionLog(
            id="test_mixed", activated_rails=[], events=[], trace=[v1_span, v2_span]
        )

        self.adapter.transform(interaction_log)

        # Verify both spans were created
        self.assertEqual(self.mock_tracer.start_span.call_count, 2)

        # Verify v2 span had events added (v1 should not)
        # Only the second span should have events
        event_calls = [call for call in mock_span.add_event.call_args_list]
        self.assertEqual(len(event_calls), 1)  # Only v2 span has events

    def test_event_content_passthrough(self):
        """Test that event content is passed through as-is by the adapter."""
        mock_span = MagicMock()
        self.mock_tracer.start_span.return_value = mock_span

        from nemoguardrails.tracing.spans import InteractionSpan

        long_content = "x" * 2000

        v2_span = InteractionSpan(
            name="test",
            span_id="truncate_test",
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            events=[
                SpanEvent(
                    name="gen_ai.content.prompt",
                    timestamp=0.5,
                    body={"content": long_content},
                )
            ],
        )

        interaction_log = InteractionLog(
            id="test_truncate", activated_rails=[], events=[], trace=[v2_span]
        )

        self.adapter.transform(interaction_log)

        # Verify content was passed through as-is
        # The adapter is now a thin bridge and doesn't truncate
        # Truncation should be done by the extractor if needed
        call_args = mock_span.add_event.call_args_list[0]
        content = call_args[1]["attributes"]["content"]
        self.assertEqual(len(content), 2000)  # Full content passed through
        self.assertEqual(content, "x" * 2000)

    def test_unique_span_timestamps_regression_fix(self):
        """Test that each span gets unique timestamps - regression test for timestamp bug.

        This test would FAIL with the old buggy logic where all end_time_ns were identical.
        It PASSES with the correct logic where each span has unique timestamps.
        """
        created_spans = []

        def track_span(*args, **kwargs):
            span = MagicMock()
            created_spans.append(span)
            return span

        self.mock_tracer.start_span.side_effect = track_span

        # Create multiple V2 spans with different timings
        from nemoguardrails.tracing.spans import ActionSpan, RailSpan

        spans = [
            InteractionSpan(
                name="span_1",
                span_id="1",
                start_time=0.0,  # Starts at trace beginning
                end_time=1.0,  # Ends after 1 second
                duration=1.0,
                custom_attributes={"type": "first"},
            ),
            RailSpan(
                name="span_2",
                span_id="2",
                start_time=0.5,  # Starts 0.5s after trace start
                end_time=2.0,  # Ends after 2 seconds
                duration=1.5,
                rail_type="input",
                rail_name="test_rail",
                custom_attributes={"type": "second"},
            ),
            ActionSpan(
                name="span_3",
                span_id="3",
                start_time=1.0,  # Starts 1s after trace start
                end_time=1.5,  # Ends after 1.5 seconds
                duration=0.5,
                action_name="test_action",
                custom_attributes={"type": "third"},
            ),
        ]

        interaction_log = InteractionLog(
            id="test_timestamps",
            activated_rails=[],
            events=[],
            trace=spans,
        )

        # Use a fixed base time for predictable results
        import time

        with unittest.mock.patch("time.time_ns", return_value=1700000000_000_000_000):
            self.adapter.transform(interaction_log)

        # Verify that each span was created
        self.assertEqual(len(created_spans), 3)

        # Extract the end times for each span
        end_times = []
        for span_mock in created_spans:
            end_call = span_mock.end.call_args
            end_times.append(end_call[1]["end_time"])

        # CRITICAL TEST: All end times should be DIFFERENT
        # With the bug, all end_times would be identical (base_time_ns)
        unique_end_times = set(end_times)
        self.assertEqual(
            len(unique_end_times),
            3,
            f"End times should be unique but got: {end_times}. "
            f"This indicates the timestamp calculation bug has regressed!",
        )

        # Verify correct absolute timestamps
        base_ns = 1700000000_000_000_000
        expected_end_times = [
            base_ns + 1_000_000_000,  # span_1 ends at 1s
            base_ns + 2_000_000_000,  # span_2 ends at 2s
            base_ns + 1_500_000_000,  # span_3 ends at 1.5s
        ]

        self.assertEqual(end_times, expected_end_times)

    def test_multiple_interactions_different_base_times(self):
        """Test that multiple interactions get different base times."""
        mock_span = MagicMock()
        self.mock_tracer.start_span.return_value = mock_span

        span1 = InteractionSpan(
            name="span1",
            span_id="1",
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            custom_attributes={"interaction": "first"},
        )

        span2 = InteractionSpan(
            name="span2",
            span_id="2",
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            custom_attributes={"interaction": "second"},
        )

        log1 = InteractionLog(id="log1", activated_rails=[], events=[], trace=[span1])
        log2 = InteractionLog(id="log2", activated_rails=[], events=[], trace=[span2])

        # First interaction
        import time

        with unittest.mock.patch("time.time_ns", return_value=1000000000_000_000_000):
            self.adapter.transform(log1)

        first_start = self.mock_tracer.start_span.call_args[1]["start_time"]

        # Reset mock
        self.mock_tracer.start_span.reset_mock()

        # Second interaction (100ms later)
        with unittest.mock.patch("time.time_ns", return_value=1000000100_000_000_000):
            self.adapter.transform(log2)

        second_start = self.mock_tracer.start_span.call_args[1]["start_time"]

        # The two interactions should have different base times
        self.assertNotEqual(first_start, second_start)
        self.assertEqual(
            second_start - first_start, 100_000_000_000
        )  # 100ms difference

    def test_uses_actual_interaction_start_time_from_rails(self):
        """Test that adapter uses the actual start time from activated rails, not current time."""
        import time

        from nemoguardrails.rails.llm.options import ActivatedRail

        one_hour_ago = time.time() - 3600

        rail = ActivatedRail(
            type="input",
            name="test_rail",
            started_at=one_hour_ago,
            finished_at=one_hour_ago + 2.0,
            duration=2.0,
        )

        span = InteractionSpan(
            name="test_span",
            span_id="test_123",
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            operation_name="test",
            service_name="test_service",
        )

        interaction_log = InteractionLog(
            id="test_actual_time", activated_rails=[rail], events=[], trace=[span]
        )

        mock_span = MagicMock()
        self.mock_tracer.start_span.return_value = mock_span

        self.adapter.transform(interaction_log)

        call_args = self.mock_tracer.start_span.call_args
        actual_start_time_ns = call_args[1]["start_time"]

        expected_start_time_ns = int(one_hour_ago * 1_000_000_000)
        self.assertEqual(
            actual_start_time_ns,
            expected_start_time_ns,
            "Should use the actual interaction start time from rails, not current time",
        )

        end_call = mock_span.end.call_args
        actual_end_time_ns = end_call[1]["end_time"]
        expected_end_time_ns = expected_start_time_ns + 1_000_000_000

        self.assertEqual(
            actual_end_time_ns,
            expected_end_time_ns,
            "End time should be calculated relative to the actual interaction start",
        )

    def test_fallback_when_no_rail_timestamp(self):
        """Test that adapter falls back to current time when rails have no timestamp."""
        span = InteractionSpan(
            name="test_span",
            span_id="test_no_rails",
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            operation_name="test",
            service_name="test_service",
        )

        interaction_log = InteractionLog(
            id="test_no_rails", activated_rails=[], events=[], trace=[span]
        )

        mock_span = MagicMock()
        self.mock_tracer.start_span.return_value = mock_span

        with patch("time.time_ns", return_value=9999999999_000_000_000):
            self.adapter.transform(interaction_log)

        call_args = self.mock_tracer.start_span.call_args
        actual_start_time_ns = call_args[1]["start_time"]

        self.assertEqual(
            actual_start_time_ns,
            9999999999_000_000_000,
            "Should fall back to current time when no rail timestamps available",
        )


if __name__ == "__main__":
    unittest.main()

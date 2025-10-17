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

import asyncio
import unittest
import warnings
from importlib.metadata import version
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import NoOpTracerProvider

from nemoguardrails.tracing import (
    InteractionLog,
    SpanEvent,
    SpanLegacy,
    SpanOpentelemetry,
)
from nemoguardrails.tracing.adapters.opentelemetry import OpenTelemetryAdapter


class TestOpenTelemetryAdapter(unittest.TestCase):
    def setUp(self):
        # Set up a mock tracer provider for testing
        self.mock_tracer_provider = MagicMock(spec=TracerProvider)
        self.mock_tracer = MagicMock()
        self.mock_tracer_provider.get_tracer.return_value = self.mock_tracer

        # Patch the global tracer provider
        patcher_get_tracer_provider = patch("opentelemetry.trace.get_tracer_provider")
        self.mock_get_tracer_provider = patcher_get_tracer_provider.start()
        self.mock_get_tracer_provider.return_value = self.mock_tracer_provider
        self.addCleanup(patcher_get_tracer_provider.stop)

        # Patch get_tracer to return our mock
        patcher_get_tracer = patch("opentelemetry.trace.get_tracer")
        self.mock_get_tracer = patcher_get_tracer.start()
        self.mock_get_tracer.return_value = self.mock_tracer
        self.addCleanup(patcher_get_tracer.stop)

        # Get the actual version for testing
        self.actual_version = version("nemoguardrails")

        # Create the adapter - it should now use the global tracer
        self.adapter = OpenTelemetryAdapter()

    def test_initialization(self):
        """Test that the adapter initializes correctly using the global tracer."""

        self.mock_get_tracer.assert_called_once_with(
            "nemo_guardrails",
            instrumenting_library_version=self.actual_version,
            schema_url="https://opentelemetry.io/schemas/1.26.0",
        )
        # Verify that the adapter has the mock tracer
        self.assertEqual(self.adapter.tracer, self.mock_tracer)

    def test_transform(self):
        """Test that transform creates spans correctly with proper timing."""
        mock_span = MagicMock()
        self.mock_tracer.start_span.return_value = mock_span

        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[
                SpanLegacy(
                    name="test_span",
                    span_id="span_1",
                    parent_id=None,
                    start_time=1234567890.5,  # historical timestamp
                    end_time=1234567891.5,  # historical timestamp
                    duration=1.0,
                    metrics={"key": 123},
                )
            ],
        )

        self.adapter.transform(interaction_log)

        # Verify that start_span was called with proper timing (not start_as_current_span)
        call_args = self.mock_tracer.start_span.call_args
        self.assertEqual(call_args[0][0], "test_span")  # name
        self.assertEqual(call_args[1]["context"], None)  # no parent context
        # Verify start_time is a reasonable absolute timestamp in nanoseconds
        start_time_ns = call_args[1]["start_time"]
        self.assertIsInstance(start_time_ns, int)
        self.assertGreater(
            start_time_ns, 1e15
        )  # Should be realistic Unix timestamp in ns

        # V1 span metrics are set directly without prefix
        mock_span.set_attribute.assert_any_call("key", 123)
        # The adapter no longer sets intrinsic IDs as attributes
        # (span_id, trace_id, duration are intrinsic to OTel spans)

        # Verify span was ended with correct end time
        end_call_args = mock_span.end.call_args
        end_time_ns = end_call_args[1]["end_time"]
        self.assertIsInstance(end_time_ns, int)
        self.assertGreater(end_time_ns, start_time_ns)  # End should be after start
        # Verify duration is approximately correct (allowing for conversion precision)
        duration_ns = end_time_ns - start_time_ns
        expected_duration_ns = int(1.0 * 1_000_000_000)  # 1 second
        self.assertAlmostEqual(
            duration_ns, expected_duration_ns, delta=1000000
        )  # 1ms tolerance

    def test_transform_span_attributes_various_types(self):
        """Test that different attribute types are handled correctly."""
        mock_span = MagicMock()
        self.mock_tracer.start_span.return_value = mock_span

        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[
                SpanLegacy(
                    name="test_span",
                    span_id="span_1",
                    parent_id=None,
                    start_time=1234567890.0,
                    end_time=1234567891.0,
                    duration=1.0,
                    metrics={
                        "int_key": 42,
                        "float_key": 3.14,
                        "str_key": 123,  # Changed to a numeric value
                        "bool_key": 1,  # Changed to a numeric value
                    },
                )
            ],
        )

        self.adapter.transform(interaction_log)

        mock_span.set_attribute.assert_any_call("int_key", 42)
        mock_span.set_attribute.assert_any_call("float_key", 3.14)
        mock_span.set_attribute.assert_any_call("str_key", 123)
        mock_span.set_attribute.assert_any_call("bool_key", 1)
        # The adapter no longer sets intrinsic IDs as attributes
        # (span_id, trace_id, duration are intrinsic to OTel spans)
        # Verify span was ended
        mock_span.end.assert_called_once()
        end_call_args = mock_span.end.call_args
        self.assertIn("end_time", end_call_args[1])
        self.assertIsInstance(end_call_args[1]["end_time"], int)

    def test_transform_with_empty_trace(self):
        """Test transform with empty trace."""
        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[],
        )

        self.adapter.transform(interaction_log)

        self.mock_tracer.start_span.assert_not_called()

    def test_transform_with_tracer_failure(self):
        """Test transform when tracer fails."""
        self.mock_tracer.start_span.side_effect = Exception("Tracer failure")

        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[
                SpanLegacy(
                    name="test_span",
                    span_id="span_1",
                    parent_id=None,
                    start_time=1234567890.0,
                    end_time=1234567891.0,
                    duration=1.0,
                    metrics={"key": 123},
                )
            ],
        )

        with self.assertRaises(Exception) as context:
            self.adapter.transform(interaction_log)

        self.assertIn("Tracer failure", str(context.exception))

    def test_transform_with_parent_child_relationships(self):
        """Test that parent-child relationships are preserved with correct timing."""
        parent_mock_span = MagicMock()
        child_mock_span = MagicMock()
        self.mock_tracer.start_span.side_effect = [parent_mock_span, child_mock_span]

        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[
                SpanLegacy(
                    name="parent_span",
                    span_id="span_1",
                    parent_id=None,
                    start_time=1234567890.0,
                    end_time=1234567892.0,
                    duration=2.0,
                    metrics={"parent_key": 1},
                ),
                SpanLegacy(
                    name="child_span",
                    span_id="span_2",
                    parent_id="span_1",
                    start_time=1234567890.5,  # child starts after parent
                    end_time=1234567891.5,  # child ends before parent
                    duration=1.0,
                    metrics={"child_key": 2},
                ),
            ],
        )

        with patch(
            "opentelemetry.trace.set_span_in_context"
        ) as mock_set_span_in_context:
            mock_set_span_in_context.return_value = "parent_context"

            self.adapter.transform(interaction_log)

            # verify parent span created first with no context
            self.assertEqual(self.mock_tracer.start_span.call_count, 2)
            first_call = self.mock_tracer.start_span.call_args_list[0]
            self.assertEqual(first_call[0][0], "parent_span")  # name
            self.assertEqual(first_call[1]["context"], None)  # no parent context
            # Verify start_time is a reasonable absolute timestamp
            start_time_ns = first_call[1]["start_time"]
            self.assertIsInstance(start_time_ns, int)
            self.assertGreater(
                start_time_ns, 1e15
            )  # Should be realistic Unix timestamp in ns

            # verify child span created with parent context
            second_call = self.mock_tracer.start_span.call_args_list[1]
            self.assertEqual(second_call[0][0], "child_span")  # name
            self.assertEqual(
                second_call[1]["context"], "parent_context"
            )  # parent context
            # Verify child start_time is also a reasonable absolute timestamp
            child_start_time_ns = second_call[1]["start_time"]
            self.assertIsInstance(child_start_time_ns, int)
            self.assertGreater(
                child_start_time_ns, 1e15
            )  # Should be realistic Unix timestamp in ns

            # verify parent context was set correctly
            mock_set_span_in_context.assert_called_once_with(parent_mock_span)

            # verify both spans ended with reasonable times
            parent_mock_span.end.assert_called_once()
            child_mock_span.end.assert_called_once()
            parent_end_time = parent_mock_span.end.call_args[1]["end_time"]
            child_end_time = child_mock_span.end.call_args[1]["end_time"]
            self.assertIsInstance(parent_end_time, int)
            self.assertIsInstance(child_end_time, int)
            self.assertGreater(parent_end_time, 1e15)
            self.assertGreater(child_end_time, 1e15)

    def test_transform_async(self):
        """Test async transform functionality."""

        async def run_test():
            mock_span = MagicMock()
            self.mock_tracer.start_span.return_value = mock_span

            interaction_log = InteractionLog(
                id="test_id",
                activated_rails=[],
                events=[],
                trace=[
                    SpanLegacy(
                        name="test_span",
                        span_id="span_1",
                        parent_id=None,
                        start_time=1234567890.5,
                        end_time=1234567891.5,
                        duration=1.0,
                        metrics={"key": 123},
                    )
                ],
            )

            await self.adapter.transform_async(interaction_log)

            call_args = self.mock_tracer.start_span.call_args
            self.assertEqual(call_args[0][0], "test_span")
            self.assertEqual(call_args[1]["context"], None)
            # Verify start_time is reasonable
            self.assertIsInstance(call_args[1]["start_time"], int)
            self.assertGreater(call_args[1]["start_time"], 1e15)

            mock_span.set_attribute.assert_any_call("key", 123)
            # The adapter no longer sets intrinsic IDs as attributes
            # (span_id, trace_id, duration are intrinsic to OTel spans)
            mock_span.end.assert_called_once()
            self.assertIn("end_time", mock_span.end.call_args[1])
            self.assertIsInstance(mock_span.end.call_args[1]["end_time"], int)

        asyncio.run(run_test())

    def test_transform_async_with_empty_trace(self):
        """Test async transform with empty trace."""

        async def run_test():
            interaction_log = InteractionLog(
                id="test_id",
                activated_rails=[],
                events=[],
                trace=[],
            )

            await self.adapter.transform_async(interaction_log)

            self.mock_tracer.start_span.assert_not_called()

        asyncio.run(run_test())

    def test_transform_async_with_tracer_failure(self):
        """Test async transform when tracer fails."""
        self.mock_tracer.start_span.side_effect = Exception("Tracer failure")

        async def run_test():
            interaction_log = InteractionLog(
                id="test_id",
                activated_rails=[],
                events=[],
                trace=[
                    SpanLegacy(
                        name="test_span",
                        span_id="span_1",
                        parent_id=None,
                        start_time=1234567890.0,
                        end_time=1234567891.0,
                        duration=1.0,
                        metrics={"key": 123},
                    )
                ],
            )

            with self.assertRaises(Exception) as context:
                await self.adapter.transform_async(interaction_log)

            self.assertIn("Tracer failure", str(context.exception))

        asyncio.run(run_test())

    def test_no_op_tracer_provider_warning(self):
        """Test that a warning is issued when NoOpTracerProvider is detected."""

        with patch("opentelemetry.trace.get_tracer_provider") as mock_get_provider:
            mock_get_provider.return_value = NoOpTracerProvider()

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                _adapter = OpenTelemetryAdapter()

                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[0].category, UserWarning))
                self.assertIn(
                    "No OpenTelemetry TracerProvider configured", str(w[0].message)
                )
                self.assertIn("Traces will not be exported", str(w[0].message))

    def test_no_warnings_with_proper_configuration(self):
        """Test that no warnings are issued when properly configured."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # adapter without deprecated parameters
            _adapter = OpenTelemetryAdapter(service_name="test_service")

            # no warnings is issued
            self.assertEqual(len(w), 0)

    def test_v1_spans_unique_timestamps_regression(self):
        """Regression test: V1 spans should have unique timestamps.

        This test ensures the timestamp bug is fixed for V1 spans.
        With the bug, all spans would have the same end_time_ns.
        """
        created_spans = []

        def track_span(*args, **kwargs):
            span = MagicMock()
            created_spans.append(span)
            return span

        self.mock_tracer.start_span.side_effect = track_span

        # Create multiple V1 spans with different end times
        spans = []
        for i in range(5):
            spans.append(
                SpanLegacy(
                    name=f"v1_span_{i}",
                    span_id=str(i),
                    start_time=float(i * 0.1),  # 0, 0.1, 0.2, 0.3, 0.4
                    end_time=float(0.5 + i * 0.2),  # 0.5, 0.7, 0.9, 1.1, 1.3
                    duration=float(0.5 + i * 0.2 - i * 0.1),
                    metrics={"index": i},
                )
            )

        interaction_log = InteractionLog(
            id="v1_regression_test",
            activated_rails=[],
            events=[],
            trace=spans,
        )

        # Use fixed time for predictable results
        import time

        with patch("time.time_ns", return_value=8000000000_000_000_000):
            self.adapter.transform(interaction_log)

        # Extract all end times
        end_times = []
        for span_mock in created_spans:
            end_time = span_mock.end.call_args[1]["end_time"]
            end_times.append(end_time)

        # CRITICAL: All end times MUST be different
        unique_end_times = set(end_times)
        self.assertEqual(
            len(unique_end_times),
            5,
            f"REGRESSION DETECTED: All V1 span end times should be unique! "
            f"Got {len(unique_end_times)} unique values from {end_times}. "
            f"The timestamp calculation bug has regressed.",
        )

        # Verify expected values
        base_ns = 8000000000_000_000_000
        expected_end_times = [
            base_ns + int(0.5 * 1_000_000_000),
            base_ns + int(0.7 * 1_000_000_000),
            base_ns + int(0.9 * 1_000_000_000),
            base_ns + int(1.1 * 1_000_000_000),
            base_ns + int(1.3 * 1_000_000_000),
        ]

        self.assertEqual(end_times, expected_end_times)

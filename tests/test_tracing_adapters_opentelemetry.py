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

import asyncio
import unittest
from unittest.mock import MagicMock, patch

# TODO: check to see if we can add it as a dependency
# but now we try to import opentelemetry and set a flag if it's not available
try:
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

    from nemoguardrails.tracing.adapters.opentelemetry import OpenTelemetryAdapter

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

from nemoguardrails.eval.models import Span
from nemoguardrails.tracing import InteractionLog


@unittest.skipIf(not OPENTELEMETRY_AVAILABLE, "opentelemetry is not available")
class TestOpenTelemetryAdapter(unittest.TestCase):
    def setUp(self):
        patcher_get_tracer = patch("opentelemetry.trace.get_tracer")
        self.mock_get_tracer = patcher_get_tracer.start()
        self.addCleanup(patcher_get_tracer.stop)

        # Create a mock tracer
        self.mock_tracer = MagicMock()
        self.mock_get_tracer.return_value = self.mock_tracer

        patcher_console_exporter = patch(
            "opentelemetry.sdk.trace.export.ConsoleSpanExporter"
        )
        self.mock_console_exporter_cls = patcher_console_exporter.start()
        self.addCleanup(patcher_console_exporter.stop)

        patcher_batch_span_processor = patch(
            "opentelemetry.sdk.trace.export.BatchSpanProcessor"
        )
        self.mock_batch_span_processor_cls = patcher_batch_span_processor.start()
        self.addCleanup(patcher_batch_span_processor.stop)

        patcher_add_span_processor = patch(
            "opentelemetry.sdk.trace.TracerProvider.add_span_processor"
        )
        self.mock_add_span_processor = patcher_add_span_processor.start()
        self.addCleanup(patcher_add_span_processor.stop)

        self.adapter = OpenTelemetryAdapter(
            span_processor=self.mock_batch_span_processor_cls,
            exporter_cls=self.mock_console_exporter_cls,
        )

    def test_initialization(self):
        self.assertIsInstance(self.adapter.tracer_provider, SDKTracerProvider)
        self.mock_add_span_processor.assert_called_once_with(
            self.mock_batch_span_processor_cls
        )

    def test_transform(self):
        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[
                Span(
                    name="test_span",
                    span_id="span_1",
                    parent_id=None,
                    start_time=0.0,
                    end_time=1.0,
                    duration=1.0,
                    metrics={"key": 123},
                )
            ],
        )

        self.adapter.transform(interaction_log)

        self.mock_tracer.start_as_current_span.assert_called_once_with(
            "test_span",
            context=None,
        )

        # We retrieve the mock span instance here
        span_instance = (
            self.mock_tracer.start_as_current_span.return_value.__enter__.return_value
        )

        span_instance.set_attribute.assert_any_call("key", 123)
        span_instance.set_attribute.assert_any_call("span_id", "span_1")
        span_instance.set_attribute.assert_any_call("trace_id", "test_id")
        span_instance.set_attribute.assert_any_call("start_time", 0.0)
        span_instance.set_attribute.assert_any_call("end_time", 1.0)
        span_instance.set_attribute.assert_any_call("duration", 1.0)

    def test_transform_span_attributes_various_types(self):
        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[
                Span(
                    name="test_span",
                    span_id="span_1",
                    parent_id=None,
                    start_time=0.0,
                    end_time=1.0,
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

        span_instance = (
            self.mock_tracer.start_as_current_span.return_value.__enter__.return_value
        )

        span_instance.set_attribute.assert_any_call("int_key", 42)
        span_instance.set_attribute.assert_any_call("float_key", 3.14)
        span_instance.set_attribute.assert_any_call("str_key", 123)
        span_instance.set_attribute.assert_any_call("bool_key", 1)
        span_instance.set_attribute.assert_any_call("span_id", "span_1")
        span_instance.set_attribute.assert_any_call("trace_id", "test_id")
        span_instance.set_attribute.assert_any_call("start_time", 0.0)
        span_instance.set_attribute.assert_any_call("end_time", 1.0)
        span_instance.set_attribute.assert_any_call("duration", 1.0)

    def test_transform_with_empty_trace(self):
        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[],
        )

        self.adapter.transform(interaction_log)

        self.mock_tracer.start_as_current_span.assert_not_called()

    def test_transform_with_exporter_failure(self):
        self.mock_tracer.start_as_current_span.side_effect = Exception(
            "Exporter failure"
        )

        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[
                Span(
                    name="test_span",
                    span_id="span_1",
                    parent_id=None,
                    start_time=0.0,
                    end_time=1.0,
                    duration=1.0,
                    metrics={"key": 123},
                )
            ],
        )

        with self.assertRaises(Exception) as context:
            self.adapter.transform(interaction_log)

        self.assertIn("Exporter failure", str(context.exception))

    def test_transform_async(self):
        async def run_test():
            interaction_log = InteractionLog(
                id="test_id",
                activated_rails=[],
                events=[],
                trace=[
                    Span(
                        name="test_span",
                        span_id="span_1",
                        parent_id=None,
                        start_time=0.0,
                        end_time=1.0,
                        duration=1.0,
                        metrics={"key": 123},
                    )
                ],
            )

            await self.adapter.transform_async(interaction_log)

            self.mock_tracer.start_as_current_span.assert_called_once_with(
                "test_span",
                context=None,
            )

            # We retrieve the mock span instance here
            span_instance = (
                self.mock_tracer.start_as_current_span.return_value.__enter__.return_value
            )

            span_instance.set_attribute.assert_any_call("key", 123)
            span_instance.set_attribute.assert_any_call("span_id", "span_1")
            span_instance.set_attribute.assert_any_call("trace_id", "test_id")
            span_instance.set_attribute.assert_any_call("start_time", 0.0)
            span_instance.set_attribute.assert_any_call("end_time", 1.0)
            span_instance.set_attribute.assert_any_call("duration", 1.0)

        asyncio.run(run_test())

    def test_transform_async_with_empty_trace(self):
        async def run_test():
            interaction_log = InteractionLog(
                id="test_id",
                activated_rails=[],
                events=[],
                trace=[],
            )

            await self.adapter.transform_async(interaction_log)

            self.mock_tracer.start_as_current_span.assert_not_called()

        asyncio.run(run_test())

    def test_transform_async_with_exporter_failure(self):
        self.mock_tracer.start_as_current_span.side_effect = Exception(
            "Exporter failure"
        )

        async def run_test():
            interaction_log = InteractionLog(
                id="test_id",
                activated_rails=[],
                events=[],
                trace=[
                    Span(
                        name="test_span",
                        span_id="span_1",
                        parent_id=None,
                        start_time=0.0,
                        end_time=1.0,
                        duration=1.0,
                        metrics={"key": 123},
                    )
                ],
            )

            with self.assertRaises(Exception) as context:
                await self.adapter.transform_async(interaction_log)

            self.assertIn("Exporter failure", str(context.exception))

        asyncio.run(run_test())

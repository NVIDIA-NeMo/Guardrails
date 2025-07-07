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

"""
OpenTelemetry Adapter for NeMo Guardrails

This adapter follows OpenTelemetry best practices for libraries:
- Uses only the OpenTelemetry API (not SDK)
- Does not modify global state
- Relies on the application to configure the SDK

Usage:
    Applications using NeMo Guardrails with OpenTelemetry should configure
    the OpenTelemetry SDK before using this adapter:

    ```python
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    # application configures the SDK
    trace.set_tracer_provider(TracerProvider())
    tracer_provider = trace.get_tracer_provider()

    exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
    span_processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(span_processor)

    # now NeMo Guardrails can use the configured tracer
    config = RailsConfig.from_content(
        config={
            "tracing": {
                "enabled": True,
                "adapters": [{"name": "OpenTelemetry"}]
            }
        }
    )
    ```
"""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nemoguardrails.tracing import InteractionLog
try:
    from opentelemetry import trace

except ImportError:
    raise ImportError(
        "OpenTelemetry API is not installed. Please install NeMo Guardrails with tracing support: "
        "`pip install nemoguardrails[tracing]` or install the API directly: `pip install opentelemetry-api`."
    )

from nemoguardrails.tracing.adapters.base import InteractionLogAdapter


class OpenTelemetryAdapter(InteractionLogAdapter):
    """
    OpenTelemetry adapter that follows library best practices.

    This adapter uses only the OpenTelemetry API and relies on the application
    to configure the SDK. It does not modify global state or create its own
    tracer provider.
    """

    name = "OpenTelemetry"

    def __init__(
        self,
        service_name: str = "nemo_guardrails",
        **kwargs,
    ):
        """
        Initialize the OpenTelemetry adapter.

        Args:
            service_name: Service name for instrumentation scope (not used for resource)
            **kwargs: Additional arguments (for backward compatibility)

        Note:
            Applications must configure the OpenTelemetry SDK before using this adapter.
            The adapter will use the globally configured tracer provider.
        """
        self.tracer = trace.get_tracer(
            service_name,
            instrumenting_library_version=version("nemoguardrails"),
            schema_url="https://opentelemetry.io/schemas/1.26.0",
        )

    def transform(self, interaction_log: "InteractionLog"):
        """Transforms the InteractionLog into OpenTelemetry spans."""
        spans = {}

        for span_data in interaction_log.trace:
            parent_span = spans.get(span_data.parent_id)
            parent_context = (
                trace.set_span_in_context(parent_span) if parent_span else None
            )

            self._create_span(
                span_data,
                parent_context,
                spans,
                interaction_log.id,  # trace_id
            )

    async def transform_async(self, interaction_log: "InteractionLog"):
        """Transforms the InteractionLog into OpenTelemetry spans asynchronously."""
        spans = {}
        for span_data in interaction_log.trace:
            parent_span = spans.get(span_data.parent_id)
            parent_context = (
                trace.set_span_in_context(parent_span) if parent_span else None
            )
            self._create_span(
                span_data,
                parent_context,
                spans,
                interaction_log.id,  # trace_id
            )

    def _create_span(
        self,
        span_data,
        parent_context,
        spans,
        trace_id,
    ):
        with self.tracer.start_as_current_span(
            span_data.name,
            context=parent_context,
        ) as span:
            for key, value in span_data.metrics.items():
                span.set_attribute(key, value)

            span.set_attribute("span_id", span_data.span_id)
            span.set_attribute("trace_id", trace_id)
            span.set_attribute("start_time", span_data.start_time)
            span.set_attribute("end_time", span_data.end_time)
            span.set_attribute("duration", span_data.duration)

            spans[span_data.span_id] = span

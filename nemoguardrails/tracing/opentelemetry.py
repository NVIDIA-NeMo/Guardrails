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

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import SpanKind

from nemoguardrails.tracing import InteractionLog, InteractionLogAdapter


class OpenTelemetryAdapter(InteractionLogAdapter):
    def __init__(
        self,
        service_name="nemo_guardrails_service",
        span_processor=None,
        exporter=None,
        resource_attributes=None,
    ):
        resource_attributes = resource_attributes or {}
        resource = Resource.create(
            {"service.name": service_name, **resource_attributes}
        )

        # Set up the tracer provider
        provider = SDKTracerProvider(resource=resource)

        # Init the span processor and exporter
        if exporter is None:
            exporter = ConsoleSpanExporter()
        if span_processor is None:
            span_processor = BatchSpanProcessor(exporter)

        provider.add_span_processor(span_processor)
        trace.set_tracer_provider(provider)

        self.tracer_provider = provider
        self.tracer = trace.get_tracer(__name__)

    def transform(self, interaction_log: InteractionLog):
        """Transforms the InteractionLog into OpenTelemetry spans."""
        spans = {}

        for span_data in interaction_log.trace:
            parent_span = spans.get(span_data.parent_id)
            parent_context = (
                trace.set_span_in_context(parent_span) if parent_span else None
            )

            # Convert time to nanoseconds
            start_time_ns = int(span_data.start_time * 1e9)
            end_time_ns = int(span_data.end_time * 1e9)

            with self.tracer.start_as_current_span(
                name=span_data.name,
                context=parent_context,
                kind=SpanKind.INTERNAL,
                start_time=start_time_ns,
            ) as span:
                for key, value in span_data.metrics.items():
                    span.set_attribute(key, value)

                span.set_attribute("span_id", span_data.span_id)
                span.set_attribute("trace_id", interaction_log.id)
                span.set_attribute("start_time", span_data.start_time)
                span.set_attribute("end_time", span_data.end_time)
                span.set_attribute("duration", span_data.duration)

                spans[span_data.span_id] = span

                # End the span at the correct time
                span.end(end_time=end_time_ns)

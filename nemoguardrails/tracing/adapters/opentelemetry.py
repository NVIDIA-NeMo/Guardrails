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

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Type

from opentelemetry.sdk.trace.export import SpanExporter

if TYPE_CHECKING:
    from nemoguardrails.tracing import InteractionLog
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Attributes, Resource
    from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

except ImportError:
    raise ImportError(
        "opentelemetry is not installed. Please install it using `pip install opentelemetry-api opentelemetry-sdk`."
    )

from nemoguardrails.tracing.adapters.base import InteractionLogAdapter

# Global dictionary to store registered exporters
_exporter_name_cls_map: Dict[str, Type[SpanExporter]] = {
    "console": ConsoleSpanExporter,
}


def register_otel_exporter(name: str, exporter_cls: Type[SpanExporter]):
    """Register a new exporter."""
    _exporter_name_cls_map[name] = exporter_cls


class OpenTelemetryAdapter(InteractionLogAdapter):
    name = "OpenTelemetry"

    def __init__(
        self,
        service_name="nemo_guardrails_service",
        span_processor: Optional[SpanProcessor] = None,
        exporter: Optional[str] = None,
        exporter_cls: Optional[SpanExporter] = None,
        resource_attributes: Optional[Attributes] = None,
        **kwargs,
    ):
        resource_attributes = resource_attributes or {}
        resource = Resource.create(
            {"service.name": service_name, **resource_attributes}
        )

        if exporter_cls and exporter:
            raise ValueError(
                "Only one of 'exporter' or 'exporter_name' should be provided"
            )
        # Set up the tracer provider
        provider = TracerProvider(resource=resource)

        # Init the span processor and exporter
        exporter_cls = None
        if exporter:
            exporter_cls = self.get_exporter(exporter, **kwargs)

        if exporter_cls is None:
            exporter_cls = ConsoleSpanExporter()

        if span_processor is None:
            span_processor = BatchSpanProcessor(exporter_cls)

        provider.add_span_processor(span_processor)
        trace.set_tracer_provider(provider)

        self.tracer_provider = provider
        self.tracer = trace.get_tracer(__name__)

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

    @staticmethod
    def get_exporter(exporter: str, **kwargs) -> SpanExporter:
        if exporter == "zipkin":
            try:
                from opentelemetry.exporter.zipkin.json import ZipkinExporter

                _exporter_name_cls_map["zipkin"] = ZipkinExporter
            except ImportError:
                raise ImportError(
                    "The opentelemetry-exporter-zipkin package is not installed. Please install it using 'pip install opentelemetry-exporter-zipkin'."
                )

        exporter_cls = _exporter_name_cls_map.get(exporter)
        if not exporter_cls:
            raise ValueError(f"Unknown exporter: {exporter}")
        return exporter_cls(**kwargs)

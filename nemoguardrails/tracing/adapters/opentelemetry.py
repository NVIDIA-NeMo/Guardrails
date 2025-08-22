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

import warnings
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from nemoguardrails.tracing import InteractionLog
try:
    from opentelemetry import trace  # type: ignore
    from opentelemetry.trace import NoOpTracerProvider  # type: ignore

except ImportError:
    raise ImportError(
        "OpenTelemetry API is not installed. Please install NeMo Guardrails with tracing support: "
        "`pip install nemoguardrails[tracing]` or install the API directly: `pip install opentelemetry-api`."
    )

from nemoguardrails.tracing.adapters.base import InteractionLogAdapter
from nemoguardrails.tracing.span_formatting import extract_span_attributes


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
    ):
        """
        Initialize the OpenTelemetry adapter.

        Args:
            service_name: Service name for instrumentation scope (not used for resource)

        Note:
            Applications must configure the OpenTelemetry SDK before using this adapter.
            The adapter will use the globally configured tracer provider.
        """

        # validate that OpenTelemetry is properly configured
        provider = trace.get_tracer_provider()  # type: ignore
        if provider is None or isinstance(provider, NoOpTracerProvider):
            warnings.warn(
                "No OpenTelemetry TracerProvider configured. Traces will not be exported. "
                "Please configure OpenTelemetry in your application code before using NeMo Guardrails. "
                "See setup guide at: "
                "https://github.com/NVIDIA/NeMo-Guardrails/blob/main/examples/configs/tracing/README.md#opentelemetry-setup",
                UserWarning,
                stacklevel=2,
            )

        self.tracer = trace.get_tracer(  # type: ignore
            service_name,
            instrumenting_library_version=version("nemoguardrails"),
            schema_url="https://opentelemetry.io/schemas/1.26.0",
        )

    def transform(self, interaction_log: "InteractionLog"):
        """Transforms the InteractionLog into OpenTelemetry spans."""
        # get the actual interaction start time from the first rail
        # all span times are relative offsets from this timestamp
        base_time_ns = _get_base_time_ns(interaction_log)

        spans: Dict[str, Any] = {}

        for span_data in interaction_log.trace:
            parent_span = (
                spans.get(span_data.parent_id) if span_data.parent_id else None
            )
            parent_context = (
                trace.set_span_in_context(parent_span) if parent_span else None
            )

            self._create_span(
                span_data,
                parent_context,
                spans,
                base_time_ns,
            )

    async def transform_async(self, interaction_log: "InteractionLog"):
        """Transforms the InteractionLog into OpenTelemetry spans asynchronously."""
        # get the actual interaction start time from the first rail
        # all span times are relative offsets from this timestamp
        base_time_ns = _get_base_time_ns(interaction_log)

        spans: Dict[str, Any] = {}

        for span_data in interaction_log.trace:
            parent_span = (
                spans.get(span_data.parent_id) if span_data.parent_id else None
            )
            parent_context = (
                trace.set_span_in_context(parent_span) if parent_span else None
            )
            self._create_span(
                span_data,
                parent_context,
                spans,
                base_time_ns,
            )

    def _create_span(
        self,
        span_data,
        parent_context,
        spans,
        base_time_ns,
    ):
        """Create OTel span from a span.

        This is a pure API bridge - all semantic attributes are extracted
        by the formatting function. We only handle:
        1. Timestamp conversion (relative to absolute)
        2. Span kind mapping (string to enum)
        3. API calls to create spans and events
        """
        # convert relative times to absolute timestamps
        # the span times are relative offsets from the start of the trace
        # base_time_ns represents the start time of the trace
        # we simply add the relative offsets to get absolute times
        relative_start_ns = int(span_data.start_time * 1_000_000_000)
        relative_end_ns = int(span_data.end_time * 1_000_000_000)

        start_time_ns = base_time_ns + relative_start_ns
        end_time_ns = base_time_ns + relative_end_ns

        attributes = extract_span_attributes(span_data)

        from opentelemetry.trace import SpanKind as OTelSpanKind

        span_kind_map = {
            "server": OTelSpanKind.SERVER,
            "client": OTelSpanKind.CLIENT,
            "internal": OTelSpanKind.INTERNAL,
        }

        span_kind_str = attributes.get("span.kind", "internal")
        otel_span_kind = span_kind_map.get(span_kind_str, OTelSpanKind.INTERNAL)

        span = self.tracer.start_span(
            span_data.name,
            context=parent_context,
            start_time=start_time_ns,
            kind=otel_span_kind,
        )

        if attributes:
            for key, value in attributes.items():
                if key == "span.kind":
                    continue
                span.set_attribute(key, value)

        if hasattr(span_data, "events") and span_data.events:
            for event in span_data.events:
                relative_event_ns = int(event.timestamp * 1_000_000_000)
                event_time_ns = base_time_ns + relative_event_ns

                event_attrs = event.attributes.copy() if event.attributes else {}

                if event.body and isinstance(event.body, dict):
                    # merge body content into attributes for OTel compatibility
                    # (OTel events don't have separate body, just attributes)
                    for body_key, body_value in event.body.items():
                        if body_key not in event_attrs:
                            event_attrs[body_key] = body_value

                span.add_event(
                    name=event.name, attributes=event_attrs, timestamp=event_time_ns
                )

        spans[span_data.span_id] = span

        span.end(end_time=end_time_ns)


def _get_base_time_ns(interaction_log: InteractionLog) -> int:
    """Get the base time in nanoseconds for tracing spans.

    Args:
        interaction_log: The interaction log containing rail timing information

    Returns:
        Base time in nanoseconds, either from the first activated rail or current time
    """
    if (
        interaction_log.activated_rails
        and interaction_log.activated_rails[0].started_at
    ):
        return int(interaction_log.activated_rails[0].started_at * 1_000_000_000)
    else:
        # This shouldn't happen in normal operation, but provide a fallback
        import time

        return time.time_ns()

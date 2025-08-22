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

"""Simple span formatting functions for different output formats."""

from typing import Any, Dict

from nemoguardrails.tracing.spans import SpanLegacy, is_opentelemetry_span


def get_schema_version_for_filesystem(span) -> str:
    """Return the schema version string based on the span type."""
    if isinstance(span, SpanLegacy):
        return "1.0"
    if is_opentelemetry_span(span):
        return "2.0"
    raise ValueError(f"Unknown span type: {type(span).__name__}.")


def format_span_for_filesystem(span) -> Dict[str, Any]:
    """Format any span type for JSON filesystem storage.

    Args:
        span: Either SpanLegacy or typed span (InteractionSpan, RailSpan, etc.)

    Returns:
        Dictionary with all span data for JSON serialization
    """
    if not isinstance(span, SpanLegacy) and not is_opentelemetry_span(span):
        raise ValueError(
            f"Unknown span type: {type(span).__name__}. "
            f"Only SpanLegacy and typed spans are supported."
        )

    result = {
        "name": span.name,
        "span_id": span.span_id,
        "parent_id": span.parent_id,
        "start_time": span.start_time,
        "end_time": span.end_time,
        "duration": span.duration,
        "span_type": span.__class__.__name__,
    }

    if isinstance(span, SpanLegacy):
        if hasattr(span, "metrics") and span.metrics:
            result["metrics"] = span.metrics

    else:  # is_typed_span(span)
        result["span_kind"] = span.span_kind
        result["attributes"] = span.to_otel_attributes()

        if hasattr(span, "events") and span.events:
            result["events"] = [
                {
                    "name": event.name,
                    "timestamp": event.timestamp,
                    "attributes": event.attributes,
                }
                for event in span.events
            ]

        if hasattr(span, "error") and span.error:
            result["error"] = {
                "occurred": span.error,
                "type": getattr(span, "error_type", None),
                "message": getattr(span, "error_message", None),
            }

        if hasattr(span, "custom_attributes") and span.custom_attributes:
            result["custom_attributes"] = span.custom_attributes

    return result


def extract_span_attributes(span) -> Dict[str, Any]:
    """Extract OpenTelemetry attributes from any span type.

    Args:
        span: Either SpanLegacy or typed span

    Returns:
        Dictionary of OpenTelemetry attributes
    """
    if isinstance(span, SpanLegacy):
        return span.metrics.copy() if hasattr(span, "metrics") and span.metrics else {}

    elif is_opentelemetry_span(span):
        return span.to_otel_attributes()

    else:
        raise ValueError(
            f"Unknown span type: {type(span).__name__}. "
            f"Only SpanLegacy and typed spans are supported."
        )

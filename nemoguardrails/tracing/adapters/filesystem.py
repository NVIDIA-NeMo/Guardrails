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

import json
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from nemoguardrails.tracing import InteractionLog

from nemoguardrails.tracing.adapters.base import InteractionLogAdapter


class FileSystemAdapter(InteractionLogAdapter):
    name = "FileSystem"
    SCHEMA_VERSION = "2.0"

    def __init__(self, filepath: Optional[str] = None):
        if not filepath:
            self.filepath = "./.traces/trace.jsonl"
        else:
            self.filepath = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def _extract_span_data(self, span_data) -> Dict[str, Any]:
        """Extract all available data from a span."""
        # Start with common fields that all spans have
        span_dict = {
            "name": span_data.name,
            "span_id": span_data.span_id,
            "parent_id": span_data.parent_id,
            "start_time": span_data.start_time,
            "end_time": span_data.end_time,
            "duration": span_data.duration,
        }

        # Add span type for debugging
        span_dict["span_type"] = span_data.__class__.__name__

        # Handle v1 spans (SpanFlat) - they have metrics
        if hasattr(span_data, "metrics") and span_data.metrics:
            span_dict["metrics"] = span_data.metrics

        # Handle v2 spans - they have span_kind
        if hasattr(span_data, "span_kind"):
            span_dict["span_kind"] = span_data.span_kind

        # Extract events if present
        if hasattr(span_data, "events") and span_data.events:
            span_dict["events"] = [
                {
                    "name": event.name,
                    "timestamp": event.timestamp,
                    "attributes": event.attributes,
                }
                for event in span_data.events
            ]

        # Extract error information if present
        if hasattr(span_data, "error") and span_data.error:
            span_dict["error"] = {
                "occurred": span_data.error,
                "type": getattr(span_data, "error_type", None),
                "message": getattr(span_data, "error_message", None),
            }

        # Extract OpenTelemetry attributes if available
        if hasattr(span_data, "to_otel_attributes"):
            span_dict["attributes"] = span_data.to_otel_attributes()

        # Include custom attributes if present
        if hasattr(span_data, "custom_attributes") and span_data.custom_attributes:
            span_dict["custom_attributes"] = span_data.custom_attributes

        return span_dict

    def transform(self, interaction_log: "InteractionLog"):
        """Transforms the InteractionLog into a JSON string."""
        spans = []

        for span_data in interaction_log.trace:
            span_dict = self._extract_span_data(span_data)
            spans.append(span_dict)

        log_dict = {
            "schema_version": self.SCHEMA_VERSION,
            "trace_id": interaction_log.id,
            "spans": spans,
        }

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_dict) + "\n")

    async def transform_async(self, interaction_log: "InteractionLog"):
        try:
            import aiofiles
        except ImportError:
            raise ImportError(
                "aiofiles is required for async file writing. Please install it using `pip install aiofiles`"
            )

        spans = []

        for span_data in interaction_log.trace:
            span_dict = self._extract_span_data(span_data)
            spans.append(span_dict)

        log_dict = {
            "schema_version": self.SCHEMA_VERSION,
            "trace_id": interaction_log.id,
            "spans": spans,
        }

        async with aiofiles.open(self.filepath, "a", encoding="utf-8") as f:
            await f.write(json.dumps(log_dict) + "\n")

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

import pytest

from nemoguardrails.tracing.span_formatting import (
    extract_span_attributes,
    format_span_for_filesystem,
)
from nemoguardrails.tracing.spans import (
    ActionSpan,
    InteractionSpan,
    LLMSpan,
    RailSpan,
    SpanEvent,
    SpanLegacy,
)


class TestFormatSpanForFilesystem:
    def test_format_legacy_span_with_metrics(self):
        span = SpanLegacy(
            name="llm_call",
            span_id="span_1",
            parent_id="parent_1",
            start_time=0.5,
            end_time=1.5,
            duration=1.0,
            metrics={"input_tokens": 10, "output_tokens": 20},
        )

        result = format_span_for_filesystem(span)

        assert result["name"] == "llm_call"
        assert result["span_id"] == "span_1"
        assert result["parent_id"] == "parent_1"
        assert result["start_time"] == 0.5
        assert result["end_time"] == 1.5
        assert result["duration"] == 1.0
        assert result["span_type"] == "SpanLegacy"
        assert result["metrics"] == {"input_tokens": 10, "output_tokens": 20}
        assert "span_kind" not in result
        assert "attributes" not in result

    def test_format_legacy_span_without_metrics(self):
        span = SpanLegacy(
            name="test",
            span_id="span_1",
            parent_id=None,
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            metrics={},
        )

        result = format_span_for_filesystem(span)

        assert result["span_type"] == "SpanLegacy"
        assert "metrics" not in result

    def test_format_interaction_span(self):
        span = InteractionSpan(
            name="interaction",
            span_id="span_1",
            parent_id=None,
            start_time=0.0,
            end_time=2.0,
            duration=2.0,
            span_kind="server",
            request_model="gpt-4",
        )

        result = format_span_for_filesystem(span)

        assert result["span_type"] == "InteractionSpan"
        assert result["span_kind"] == "server"
        assert "attributes" in result
        assert result["attributes"]["gen_ai.operation.name"] == "guardrails"

    def test_format_span_with_events(self):
        events = [
            SpanEvent(
                name="test_event",
                timestamp=0.5,
                attributes={"key": "value"},
            )
        ]
        span = InteractionSpan(
            name="interaction",
            span_id="span_1",
            parent_id=None,
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            span_kind="server",
            events=events,
        )

        result = format_span_for_filesystem(span)

        assert "events" in result
        assert len(result["events"]) == 1
        assert result["events"][0]["name"] == "test_event"
        assert result["events"][0]["timestamp"] == 0.5
        assert result["events"][0]["attributes"] == {"key": "value"}

    def test_format_span_with_error(self):
        span = ActionSpan(
            name="action",
            span_id="span_1",
            parent_id=None,
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            span_kind="internal",
            action_name="fetch",
            error=True,
            error_type="ConnectionError",
            error_message="Failed",
        )

        result = format_span_for_filesystem(span)

        assert "error" in result
        assert result["error"]["occurred"] is True
        assert result["error"]["type"] == "ConnectionError"
        assert result["error"]["message"] == "Failed"

    def test_format_span_with_custom_attributes(self):
        span = LLMSpan(
            name="llm",
            span_id="span_1",
            parent_id=None,
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            span_kind="client",
            provider_name="openai",
            operation_name="chat.completions",
            request_model="gpt-4",
            response_model="gpt-4",
            custom_attributes={"custom": "value"},
        )

        result = format_span_for_filesystem(span)

        assert "custom_attributes" in result
        assert result["custom_attributes"] == {"custom": "value"}

    def test_format_unknown_span_type_raises(self):
        class UnknownSpan:
            def __init__(self):
                self.name = "unknown"

        with pytest.raises(ValueError) as exc_info:
            format_span_for_filesystem(UnknownSpan())

        assert "Unknown span type: UnknownSpan" in str(exc_info.value)
        assert "Only SpanLegacy and typed spans are supported" in str(exc_info.value)


class TestExtractSpanAttributes:
    def test_extract_from_legacy_span_with_metrics(self):
        span = SpanLegacy(
            name="test",
            span_id="1",
            parent_id=None,
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            metrics={"tokens": 100, "latency": 0.5},
        )

        attrs = extract_span_attributes(span)

        assert attrs == {"tokens": 100, "latency": 0.5}
        assert attrs is not span.metrics

    def test_extract_from_legacy_span_without_metrics(self):
        span = SpanLegacy(
            name="test",
            span_id="1",
            parent_id=None,
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            metrics={},
        )

        attrs = extract_span_attributes(span)

        assert attrs == {}

    def test_extract_from_interaction_span(self):
        span = InteractionSpan(
            name="interaction",
            span_id="1",
            parent_id=None,
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            span_kind="server",
            request_model="gpt-4",
        )

        attrs = extract_span_attributes(span)

        assert "span.kind" in attrs
        assert attrs["span.kind"] == "server"
        assert "gen_ai.operation.name" in attrs

    def test_extract_from_rail_span(self):
        span = RailSpan(
            name="check",
            span_id="1",
            parent_id=None,
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            span_kind="internal",
            rail_type="input",
            rail_name="check_jailbreak",
            rail_stop=False,
        )

        attrs = extract_span_attributes(span)

        assert attrs["rail.type"] == "input"
        assert attrs["rail.name"] == "check_jailbreak"
        assert attrs["rail.stop"] is False

    def test_extract_from_llm_span(self):
        span = LLMSpan(
            name="llm",
            span_id="1",
            parent_id=None,
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            span_kind="client",
            provider_name="openai",
            operation_name="chat.completions",
            request_model="gpt-4",
            response_model="gpt-4",
            temperature=0.7,
            usage_input_tokens=50,
            usage_output_tokens=100,
        )

        attrs = extract_span_attributes(span)

        assert attrs["gen_ai.request.model"] == "gpt-4"
        assert attrs["gen_ai.request.temperature"] == 0.7
        assert attrs["gen_ai.usage.input_tokens"] == 50
        assert attrs["gen_ai.usage.output_tokens"] == 100

    def test_extract_unknown_span_type_raises(self):
        class UnknownSpan:
            pass

        with pytest.raises(ValueError) as exc_info:
            extract_span_attributes(UnknownSpan())

        assert "Unknown span type: UnknownSpan" in str(exc_info.value)

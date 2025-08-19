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

import json
from typing import Any

import pytest

from nemoguardrails.tracing.span_format import (
    SpanFormat,
    SpanFormatType,
    validate_span_format,
)


class TestSpanFormat:
    """Test cases for SpanFormat enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        assert SpanFormat.LEGACY.value == "legacy"
        assert SpanFormat.OPENTELEMETRY.value == "opentelemetry"

    def test_string_inheritance(self):
        """Test that SpanFormat inherits from str."""
        assert isinstance(SpanFormat.LEGACY, str)
        assert isinstance(SpanFormat.OPENTELEMETRY, str)

    def test_string_comparison(self):
        """Test direct string comparison works."""
        assert SpanFormat.LEGACY == "legacy"
        assert SpanFormat.OPENTELEMETRY == "opentelemetry"
        assert SpanFormat.LEGACY != "opentelemetry"

    def test_json_serialization(self):
        """Test that enum values can be JSON serialized."""
        data = {"format": SpanFormat.LEGACY}
        json_str = json.dumps(data)
        assert '"format": "legacy"' in json_str

        parsed = json.loads(json_str)
        assert parsed["format"] == "legacy"

    def test_str_method(self):
        """Test __str__ method returns value."""
        assert str(SpanFormat.LEGACY) == "legacy"
        assert str(SpanFormat.OPENTELEMETRY) == "opentelemetry"

    def test_from_string_valid_values(self):
        """Test from_string with valid values."""
        assert SpanFormat.from_string("legacy") == SpanFormat.LEGACY
        assert SpanFormat.from_string("opentelemetry") == SpanFormat.OPENTELEMETRY

        assert SpanFormat.from_string("LEGACY") == SpanFormat.LEGACY
        assert SpanFormat.from_string("OpenTelemetry") == SpanFormat.OPENTELEMETRY
        assert SpanFormat.from_string("OPENTELEMETRY") == SpanFormat.OPENTELEMETRY

    def test_from_string_invalid_value(self):
        """Test from_string with invalid value raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SpanFormat.from_string("invalid")

        error_msg = str(exc_info.value)
        assert "Invalid span format: 'invalid'" in error_msg
        assert "Valid formats are: legacy, opentelemetry" in error_msg

    def test_from_string_empty_value(self):
        """Test from_string with empty string raises ValueError."""
        with pytest.raises(ValueError):
            SpanFormat.from_string("")

    def test_from_string_none_value(self):
        """Test from_string with None raises appropriate error."""
        with pytest.raises(AttributeError):
            SpanFormat.from_string(None)


class TestValidateSpanFormat:
    """Test cases for validate_span_format function."""

    def test_validate_span_format_enum(self):
        """Test validation with SpanFormat enum."""
        result = validate_span_format(SpanFormat.LEGACY)
        assert result == SpanFormat.LEGACY
        assert isinstance(result, SpanFormat)

        result = validate_span_format(SpanFormat.OPENTELEMETRY)
        assert result == SpanFormat.OPENTELEMETRY
        assert isinstance(result, SpanFormat)

    def test_validate_span_format_string(self):
        """Test validation with string values."""
        result = validate_span_format("legacy")
        assert result == SpanFormat.LEGACY
        assert isinstance(result, SpanFormat)

        result = validate_span_format("opentelemetry")
        assert result == SpanFormat.OPENTELEMETRY
        assert isinstance(result, SpanFormat)

        result = validate_span_format("LEGACY")
        assert result == SpanFormat.LEGACY

    def test_validate_span_format_invalid_string(self):
        """Test validation with invalid string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_span_format("invalid")

        error_msg = str(exc_info.value)
        assert "Invalid span format: 'invalid'" in error_msg

    def test_validate_span_format_invalid_type(self):
        """Test validation with invalid type raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            validate_span_format(123)

        error_msg = str(exc_info.value)
        assert "Span format must be a string or SpanFormat enum" in error_msg
        assert "got <class 'int'>" in error_msg

    def test_validate_span_format_none(self):
        """Test validation with None raises TypeError."""
        with pytest.raises(TypeError):
            validate_span_format(None)

    def test_validate_span_format_list(self):
        """Test validation with list raises TypeError."""
        with pytest.raises(TypeError):
            validate_span_format(["legacy"])

    def test_validate_span_format_dict(self):
        """Test validation with dict raises TypeError."""
        with pytest.raises(TypeError):
            validate_span_format({"format": "legacy"})


class TestSpanFormatType:
    """Test cases for SpanFormatType type alias."""

    def test_type_alias_accepts_enum(self):
        """Test that type alias accepts SpanFormat enum."""

        def test_function(format_type: SpanFormatType) -> SpanFormat:
            return validate_span_format(format_type)

        result = test_function(SpanFormat.LEGACY)
        assert result == SpanFormat.LEGACY

    def test_type_alias_accepts_string(self):
        """Test that type alias accepts string values."""

        def test_function(format_type: SpanFormatType) -> SpanFormat:
            return validate_span_format(format_type)

        result = test_function("legacy")
        assert result == SpanFormat.LEGACY

        result = test_function("opentelemetry")
        assert result == SpanFormat.OPENTELEMETRY


class TestSpanFormatIntegration:
    """Integration tests for span format functionality."""

    def test_config_usage_pattern(self):
        """Test typical configuration usage pattern."""
        config_value = "opentelemetry"
        format_enum = validate_span_format(config_value)

        if format_enum == SpanFormat.OPENTELEMETRY:
            assert True  # Expected path
        else:
            pytest.fail("Unexpected format")

    def test_function_parameter_pattern(self):
        """Test typical function parameter usage pattern."""

        def process_spans(span_format: SpanFormatType = SpanFormat.LEGACY):
            validated_format = validate_span_format(span_format)
            return validated_format

        result = process_spans()
        assert result == SpanFormat.LEGACY

        result = process_spans("opentelemetry")
        assert result == SpanFormat.OPENTELEMETRY

        result = process_spans(SpanFormat.OPENTELEMETRY)
        assert result == SpanFormat.OPENTELEMETRY

    def test_all_enum_values_have_tests(self):
        """Ensure all enum values are tested."""
        tested_values = {"legacy", "opentelemetry"}
        actual_values = {format_enum.value for format_enum in SpanFormat}
        assert (
            tested_values == actual_values
        ), f"Missing tests for: {actual_values - tested_values}"

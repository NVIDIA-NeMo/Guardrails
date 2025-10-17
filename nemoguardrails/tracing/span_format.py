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

"""Span format definitions for NeMo Guardrails tracing."""

from enum import Enum
from typing import Literal, Union


class SpanFormat(str, Enum):
    """Supported span formats for tracing.

    Inherits from str to allow direct string comparison and JSON serialization.
    """

    # legacy structure with metrics dictionary (simple, minimal overhead)
    LEGACY = "legacy"

    # OpenTelemetry Semantic Conventions compliant format
    # see https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/
    OPENTELEMETRY = "opentelemetry"

    @classmethod
    def from_string(cls, value: str) -> "SpanFormat":
        """Create SpanFormat from string value.

        Args:
            value: String representation of span format

        Returns:
            SpanFormat enum value

        Raises:
            ValueError: If value is not a valid span format
        """
        try:
            return cls(value.lower())
        except ValueError:
            valid_formats = [f.value for f in cls]
            raise ValueError(
                f"Invalid span format: '{value}'. "
                f"Valid formats are: {', '.join(valid_formats)}"
            )

    def __str__(self) -> str:
        """Return string value for use in configs."""
        return self.value


# Type alias for function signatures
SpanFormatType = Union[SpanFormat, Literal["legacy", "opentelemetry"], str]


def validate_span_format(value: SpanFormatType) -> SpanFormat:
    """Validate and convert span format value to SpanFormat enum.

    Args:
        value: Span format as enum, literal, or string

    Returns:
        Validated SpanFormat enum value

    Raises:
        ValueError: If value is not a valid span format
    """
    if isinstance(value, SpanFormat):
        return value
    elif isinstance(value, str):
        return SpanFormat.from_string(value)
    else:
        raise TypeError(
            f"Span format must be a string or SpanFormat enum, got {type(value)}"
        )

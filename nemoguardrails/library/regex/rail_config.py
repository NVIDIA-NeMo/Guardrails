# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re
from typing import List, Optional

from nemoguardrails.library.regex.safety import (
    UnsafeRegexPatternError,
    ensure_pattern_is_safe,
)
from nemoguardrails.manifests.config_schema import (
    Field,
    PrivateAttr,
    RailConfigBaseModel,
    RailConfigSpec,
    model_validator,
    rail_field,
)


class RegexDetectionOptions(RailConfigBaseModel):
    """Configuration options for regex pattern detection on a specific source."""

    patterns: List[str] = Field(
        default_factory=list,
        description="List of regex patterns to match against the text.",
    )
    case_insensitive: bool = Field(
        default=False,
        description="Whether to perform case-insensitive matching.",
    )

    _compiled_patterns: List["re.Pattern[str]"] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def compile_patterns(self) -> "RegexDetectionOptions":
        """Pre-compile regex patterns at config load time.

        Beyond syntax validation, each pattern is checked against the
        documented safe subset so that a configured pattern cannot exhibit
        catastrophic backtracking at request time (ReDoS, #2203).
        """
        flags = re.IGNORECASE if self.case_insensitive else 0
        compiled = []
        for i, pattern in enumerate(self.patterns):
            try:
                compiled.append(re.compile(pattern, flags))
            except re.error as e:
                raise ValueError(f"Invalid regex pattern at index {i} ({pattern!r}): {e}") from e
            try:
                ensure_pattern_is_safe(pattern, flags)
            except UnsafeRegexPatternError as e:
                raise ValueError(f"Unsafe regex pattern at index {i} ({pattern!r}): {e}") from e
        object.__setattr__(self, "_compiled_patterns", compiled)
        return self

    @property
    def compiled_patterns(self) -> List["re.Pattern[str]"]:
        """Return the pre-compiled regex patterns."""
        return self._compiled_patterns


class RegexDetection(RailConfigBaseModel):
    """Configuration for regex pattern detection."""

    input: RegexDetectionOptions = Field(
        default_factory=RegexDetectionOptions,
        description="Configuration for regex patterns to detect on user input.",
    )
    output: RegexDetectionOptions = Field(
        default_factory=RegexDetectionOptions,
        description="Configuration for regex patterns to detect on bot output.",
    )
    retrieval: RegexDetectionOptions = Field(
        default_factory=RegexDetectionOptions,
        description="Configuration for regex patterns to detect on retrieved relevant chunks.",
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[RegexDetection],
        field_info=rail_field(
            default_factory=RegexDetection,
            description="Configuration for regex pattern detection.",
        ),
        exports={
            "RegexDetection": RegexDetection,
            "RegexDetectionOptions": RegexDetectionOptions,
        },
    )

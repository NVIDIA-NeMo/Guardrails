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

import asyncio
import logging

import pytest
from pydantic import ValidationError

from nemoguardrails.library.regex.actions import (
    MAX_REGEX_SCAN_CHARS,
    detect_regex_pattern,
)
from nemoguardrails.library.regex.rail_config import RegexDetectionOptions


@pytest.mark.unit
@pytest.mark.parametrize(
    "pattern",
    [
        r"(a+)+",
        r"(a*)*",
        r"(a+)+$",
        r"(?:\w+\s*)*$",
        r"(a|a+)*b",
        r"(?:a*b+)*c",
        r"((a+)b)+",
        r"x|(a+)*y",
        r"(a+){2,}b",
    ],
)
def test_unsafe_nested_quantifier_rejected_at_config_load(pattern):
    """Patterns that nest unbounded quantifiers must fail config load."""
    with pytest.raises(ValidationError, match="Unsafe regex pattern"):
        RegexDetectionOptions(patterns=[pattern])


@pytest.mark.unit
def test_invalid_pattern_error_unchanged():
    """Syntax errors keep their existing message and take precedence."""
    with pytest.raises(ValidationError, match="Invalid regex pattern"):
        RegexDetectionOptions(patterns=["(a+"])


@pytest.mark.unit
@pytest.mark.parametrize(
    "pattern",
    [
        r"\d{3}-\d{2}-\d{4}",
        r"(?:cat|dog)s?",
        r"a*b*c*",
        r"[abc]+",
        r"\b\w+\b",
        r"(?:ab)*c",
        r"^abc$",
        r"x(?:y|z)?w",
        r"(a{0,2}){0,2}",
        r"(a{2})+",
        r"(?<=pre)post",
        r".*?",
        r"(19|20)\d{2}",
    ],
)
def test_safe_patterns_accepted(pattern):
    """Ordinary patterns, bounded nesting, and lookarounds stay allowed."""
    options = RegexDetectionOptions(patterns=[pattern])
    assert len(options.compiled_patterns) == 1


@pytest.mark.unit
def test_case_insensitive_flag_still_applied():
    """The safe-subset check must not disturb case_insensitive behavior."""
    options = RegexDetectionOptions(
        patterns=[r"ssn:\s+\d+"],
        case_insensitive=True,
    )
    (compiled,) = options.compiled_patterns
    assert compiled.search("SSN:   123-45-6789") is not None


def _config_with_patterns(*patterns: str):
    from nemoguardrails import RailsConfig

    # Escape backslashes for YAML double-quoted scalars.
    lines = "\n".join(f'                      - "{p.replace(chr(92), chr(92) * 2)}"' for p in patterns)
    return RailsConfig.from_content(
        yaml_content=f"""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
{lines}
        """,
    )


@pytest.mark.unit
def test_detect_regex_pattern_still_matches_after_vetting():
    """End-to-end action behavior on a vetted pattern is unchanged."""
    config = _config_with_patterns(r"\d{3}-\d{2}-\d{4}")
    result = asyncio.run(detect_regex_pattern(source="input", text="My SSN is 123-45-6789", config=config))
    assert result.metadata["is_match"] is True
    assert result.metadata["detections"] == [r"\d{3}-\d{2}-\d{4}"]


@pytest.mark.unit
def test_scan_cap_limits_text_and_logs(monkeypatch, caplog):
    """Oversized texts are truncated to the cap with a warning; matches
    beyond the cap are not reported (documented trade-off)."""
    monkeypatch.setattr("nemoguardrails.library.regex.actions.MAX_REGEX_SCAN_CHARS", 40)
    config = _config_with_patterns(r"SECRET-\d+")

    inside = "A" * 30 + " SECRET-1"
    outside = "A" * 45 + " SECRET-9"

    with caplog.at_level(logging.WARNING):
        hit = asyncio.run(detect_regex_pattern(source="input", text=inside, config=config))
        miss = asyncio.run(detect_regex_pattern(source="input", text=outside, config=config))

    assert hit.metadata["is_match"] is True
    assert miss.metadata["is_match"] is False
    assert any("scan cap" in r.message for r in caplog.records)


@pytest.mark.unit
def test_default_scan_cap_constant_is_bounded():
    """The shipped default keeps worst-case work bounded (#2203)."""
    assert 0 < MAX_REGEX_SCAN_CHARS <= 1_000_000

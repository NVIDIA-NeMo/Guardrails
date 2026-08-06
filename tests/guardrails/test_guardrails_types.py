# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for guardrails_types module."""

import pytest

from nemoguardrails.guardrails.guardrails_types import (
    LLMMessage,
    LLMMessages,
    RailResult,
    client_reason,
    display_reason,
    truncate,
)


class TestRailResult:
    """Tests for the RailResult frozen dataclass."""

    def test_safe_result_defaults(self):
        """Test creating a safe result with default reason=None."""
        result = RailResult(is_safe=True)
        assert result.is_safe is True
        assert result.reason is None

    def test_safe_result_explicit_none(self):
        """Test creating a safe result with explicit reason=None."""
        result = RailResult(is_safe=True, reason=None)
        assert result.is_safe is True
        assert result.reason is None

    def test_unsafe_result_with_reason(self):
        """Test creating an unsafe result with a reason string."""
        result = RailResult(is_safe=False, reason="Content safety violation")
        assert result.is_safe is False
        assert result.reason == "Content safety violation"

    def test_unsafe_result_without_reason(self):
        """Test creating an unsafe result without a reason."""
        result = RailResult(is_safe=False)
        assert result.is_safe is False
        assert result.reason is None

    def test_equality_same_values(self):
        """Test that two RailResults with the same values are equal."""
        a = RailResult(is_safe=True)
        b = RailResult(is_safe=True)
        assert a == b

    def test_equality_with_reason(self):
        """Test equality when both have the same reason."""
        a = RailResult(is_safe=False, reason="blocked")
        b = RailResult(is_safe=False, reason="blocked")
        assert a == b

    def test_inequality_different_is_safe(self):
        """Test inequality when is_safe differs."""
        a = RailResult(is_safe=True)
        b = RailResult(is_safe=False)
        assert a != b

    def test_inequality_different_reason(self):
        """Test inequality when reason differs."""
        a = RailResult(is_safe=False, reason="reason1")
        b = RailResult(is_safe=False, reason="reason2")
        assert a != b

    def test_repr(self):
        """Test the string representation."""
        result = RailResult(is_safe=False, reason="jailbreak")
        assert "is_safe=False" in repr(result)
        assert "reason='jailbreak'" in repr(result)

    def test_reason_with_empty_string(self):
        """Test that empty string reason is distinct from None."""
        result = RailResult(is_safe=False, reason="")
        assert result.reason == ""
        assert result != RailResult(is_safe=False, reason=None)


class TestTruncate:
    """Tests for the truncate helper."""

    def test_short_string_unchanged(self):
        assert truncate("hello", 10) == "hello"

    def test_exact_length_unchanged(self):
        assert truncate("hello", 5) == "hello"

    def test_long_string_truncated(self):
        assert truncate("hello world", 5) == "hello..."

    def test_max_len_zero_truncates_everything(self):
        assert truncate("hello", 0) == "..."

    def test_none_max_len_uses_default(self):
        short = "x" * 200
        assert truncate(short, None) == short
        long = "x" * 201
        assert truncate(long, None) == "x" * 200 + "..."

    def test_non_string_input_converted(self):
        assert truncate(12345, 3) == "123..."


class TestDisplayReason:
    """Tests for the display_reason helper."""

    @pytest.mark.parametrize(
        "result, expected",
        [
            (RailResult(is_safe=False, reason="Safety categories: S1: Violence"), "Safety categories: S1: Violence"),
            (
                RailResult(
                    is_safe=False,
                    reason="Safety categories: S1: Violence",
                    triggered_rail="content safety check input",
                    return_value={"allowed": False, "policy_violations": ["S2: Sexual"]},
                ),
                "Safety categories: S1: Violence",
            ),
            (
                RailResult(
                    is_safe=False,
                    triggered_rail="content safety check input",
                    return_value={"allowed": False, "policy_violations": ["S1: Violence", "S2: Sexual"]},
                ),
                "policy_violations: S1: Violence, S2: Sexual",
            ),
            (
                RailResult(
                    is_safe=False,
                    triggered_rail="content safety check input",
                    return_value={"allowed": False, "policy_violations": [], "score": 0},
                ),
                "score: 0",
            ),
            (
                RailResult(
                    is_safe=False,
                    triggered_rail="content safety check input",
                    return_value={"allowed": False, "policy_violations": ["S1: Violence"], "score": 0.97},
                ),
                "policy_violations: S1: Violence; score: 0.97",
            ),
            (
                RailResult(
                    is_safe=False,
                    triggered_rail="content safety check input",
                    return_value={"allowed": False, "policy_violations": None},
                ),
                "content safety check input",
            ),
            (
                RailResult(
                    is_safe=False,
                    triggered_rail="jailbreak detection model",
                    return_value={"allowed": False},
                ),
                "jailbreak detection model",
            ),
            (RailResult(is_safe=False), "unspecified"),
        ],
        ids=[
            "reason",
            "reason-beats-evidence",
            "evidence",
            "empty-evidence-skipped",
            "multiple-fields-joined",
            "null-evidence-skipped",
            "rail-name",
            "nothing-at-all",
        ],
    )
    def test_prefers_the_most_specific_source(self, result, expected):
        """A stated reason wins, then the verdict's non-empty evidence, then the rail's name."""
        assert display_reason(result) == expected


class TestClientReason:
    """What reaches the caller in a block payload, as opposed to what reaches the log."""

    CROWDSTRIKE_VERDICT = {
        "allowed": False,
        "blocked": True,
        "guard_output": "policy 7",
        "user_message": "my private question",
        "bot_message": "the draft reply",
    }

    @pytest.mark.parametrize(
        "result, expected",
        [
            (RailResult(is_safe=False, reason="Safety categories: S1: Violence"), "Safety categories: S1: Violence"),
            (
                RailResult(
                    is_safe=False,
                    triggered_rail="content safety check input",
                    return_value={"allowed": False, "policy_violations": ["S1: Violence"]},
                ),
                "policy_violations: S1: Violence",
            ),
            (
                RailResult(
                    is_safe=False,
                    triggered_rail="content safety check input",
                    return_value={"allowed": False, "policy_violations": ["S1: Violence"], "raw_response": "..."},
                ),
                "policy_violations: S1: Violence",
            ),
        ],
        ids=["reason-passes-through", "listed-key", "unlisted-key-dropped-from-a-mix"],
    )
    def test_only_listed_verdict_fields_reach_the_caller(self, result, expected):
        """A rail-authored reason passes through; verdict fields reach the caller only by name."""
        assert client_reason(result) == expected

    def test_an_entirely_unlisted_verdict_falls_back_to_the_rail_name(self):
        """A rail whose evidence is all unlisted names itself rather than disclosing nothing useful."""
        result = RailResult(
            is_safe=False,
            triggered_rail="crowdstrike aidr guard input",
            return_value=self.CROWDSTRIKE_VERDICT,
        )

        assert client_reason(result) == "crowdstrike aidr guard input"

    def test_the_log_still_carries_what_the_caller_does_not(self):
        """The restriction is on the payload only; diagnosis keeps the full verdict."""
        result = RailResult(
            is_safe=False,
            triggered_rail="crowdstrike aidr guard input",
            return_value=self.CROWDSTRIKE_VERDICT,
        )

        rendered = display_reason(result)

        assert "my private question" in rendered
        assert "the draft reply" in rendered
        assert "my private question" not in client_reason(result)


class TestTypeAliases:
    """Tests for the LLMMessage and LLMMessages type aliases."""

    def test_llm_message_is_dict(self):
        """Test that LLMMessage is a dict type alias."""
        msg: LLMMessage = {"role": "user", "content": "hello"}
        assert isinstance(msg, dict)

    def test_llm_messages_is_list(self):
        """Test that LLMMessages is a list of dicts."""
        msgs: LLMMessages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert isinstance(msgs, list)
        assert all(isinstance(m, dict) for m in msgs)

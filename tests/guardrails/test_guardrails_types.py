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

from nemoguardrails.guardrails.guardrails_types import LLMMessage, LLMMessages, RailResult, truncate


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

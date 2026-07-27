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

import contextvars
from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails.guardrails.guardrails_types import (
    LLMMessage,
    LLMMessages,
    RailResult,
    get_http_headers,
    request_http_headers,
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


class TestRequestHttpHeaders:
    """Tests for the request-scoped inference-time HTTP header ContextVar."""

    def test_defaults_to_none_outside_a_request(self):
        """Without an active request scope there are no inference-time headers."""
        assert get_http_headers() is None

    def test_headers_visible_inside_scope(self):
        """Headers bound by the context manager are readable for the duration of the scope."""
        with request_http_headers({"X-Tenant": "acme"}):
            assert get_http_headers() == {"X-Tenant": "acme"}

    def test_headers_cleared_after_scope(self):
        """Leaving the scope restores the previous (absent) value so headers don't leak between requests."""
        with request_http_headers({"X-Tenant": "acme"}):
            pass
        assert get_http_headers() is None

    def test_headers_cleared_when_scope_raises(self):
        """A failing request still resets the ContextVar."""
        with pytest.raises(RuntimeError):
            with request_http_headers({"X-Tenant": "acme"}):
                raise RuntimeError("request failed")
        assert get_http_headers() is None

    def test_none_binds_none(self):
        """Passing None binds None rather than an empty dict."""
        with request_http_headers(None):
            assert get_http_headers() is None

    def test_empty_mapping_binds_empty_dict(self):
        """An empty mapping is preserved as an empty dict, distinct from None."""
        with request_http_headers({}):
            assert get_http_headers() == {}

    def test_values_coerced_to_str(self):
        """Non-string names and values are coerced, matching config default_headers handling."""
        with request_http_headers({"X-Count": 3, "X-Flag": True}):
            headers = get_http_headers()
        assert headers == {"X-Count": "3", "X-Flag": "True"}

    def test_bound_headers_do_not_alias_the_caller_mapping(self):
        """The bound headers are a copy, so later caller mutation cannot change the request's headers."""
        caller_headers = {"X-Tenant": "acme"}
        with request_http_headers(caller_headers):
            caller_headers["X-Tenant"] = "other"
            assert get_http_headers() == {"X-Tenant": "acme"}

    def test_nested_scope_restores_outer_headers(self):
        """An inner scope shadows the outer headers and restores them on exit."""
        with request_http_headers({"X-Tenant": "outer"}):
            with request_http_headers({"X-Tenant": "inner"}):
                assert get_http_headers() == {"X-Tenant": "inner"}
            assert get_http_headers() == {"X-Tenant": "outer"}

    def test_reset_from_a_different_context_is_tolerated(self):
        """Exiting the scope from another context, as async-generator teardown does, does not raise."""
        scope = request_http_headers({"X-Tenant": "acme"})
        contextvars.Context().run(scope.__enter__)

        scope.__exit__(None, None, None)

    def test_unexpected_reset_error_is_reraised(self):
        """A reset failure other than the cross-context one surfaces instead of being swallowed."""
        fake_var = MagicMock()
        fake_var.reset.side_effect = ValueError("boom")

        with patch("nemoguardrails.guardrails.guardrails_types._http_headers_var", fake_var):
            with pytest.raises(ValueError, match="boom"):
                with request_http_headers({"X-Tenant": "acme"}):
                    pass

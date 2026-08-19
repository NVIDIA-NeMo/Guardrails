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

"""Unit tests for _http module (shared aiohttp helpers)."""

from unittest.mock import AsyncMock

import pytest

from nemoguardrails.guardrails._http import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT_CONNECT,
    DEFAULT_TIMEOUT_TOTAL,
    RETRYABLE_STATUS_CODES,
    merge_headers_case_insensitive,
    normalize_query_params,
    safe_read_body,
)


class TestSafeReadBody:
    """Test safe_read_body error body extraction and truncation."""

    @pytest.mark.asyncio
    async def test_reads_short_body(self):
        """Short body is returned as-is."""
        mock_response = AsyncMock()
        mock_response.text = AsyncMock(return_value="short error message")
        result = await safe_read_body(mock_response)
        assert result == "short error message"

    @pytest.mark.asyncio
    async def test_handles_read_failure(self):
        """Returns fallback string when response.text() raises."""
        mock_response = AsyncMock()
        mock_response.text = AsyncMock(side_effect=Exception("read failed"))
        result = await safe_read_body(mock_response)
        assert result == "<could not read response body>"

    @pytest.mark.asyncio
    async def test_exactly_500_chars_not_truncated(self):
        """Body of exactly 500 chars is not truncated."""
        text = "a" * 500
        mock_response = AsyncMock()
        mock_response.text = AsyncMock(return_value=text)
        result = await safe_read_body(mock_response)
        assert len(result) == 500

    @pytest.mark.asyncio
    async def test_501_chars_truncated(self):
        """Body of 501 chars is truncated to 500."""
        text = "a" * 501
        mock_response = AsyncMock()
        mock_response.text = AsyncMock(return_value=text)
        result = await safe_read_body(mock_response)
        assert len(result) == 500

    @pytest.mark.asyncio
    async def test_explicit_max_chars_overrides_the_default(self):
        """Error classification reads further than the default, so an oversized JSON body still parses."""
        text = "a" * 5000
        mock_response = AsyncMock()
        mock_response.text = AsyncMock(return_value=text)
        result = await safe_read_body(mock_response, max_chars=8192)
        assert result == text


class TestMergeHeadersCaseInsensitive:
    """Test case-insensitive header merging (override wins, base preserved)."""

    def test_overrides_added_to_base(self):
        """Non-colliding override keys are added to the base headers."""
        base = {"Content-Type": "application/json"}
        result = merge_headers_case_insensitive(base, {"X-Tenant": "acme"})
        assert result == {"Content-Type": "application/json", "X-Tenant": "acme"}

    def test_override_replaces_case_insensitive_match(self):
        """An override replaces a base header that differs only by case, keeping one entry."""
        base = {"Content-Type": "application/json", "Authorization": "Bearer base"}
        result = merge_headers_case_insensitive(base, {"authorization": "Bearer override"})
        auth_keys = [key for key in result if key.lower() == "authorization"]
        assert auth_keys == ["authorization"]
        assert result["authorization"] == "Bearer override"

    def test_none_overrides_returns_base_copy(self):
        """None overrides yields an unchanged copy that does not alias the base."""
        base = {"Content-Type": "application/json"}
        result = merge_headers_case_insensitive(base, None)
        assert result == base
        assert result is not base

    def test_empty_overrides_returns_base_copy(self):
        """Empty overrides yields an unchanged copy that does not alias the base."""
        base = {"Content-Type": "application/json"}
        result = merge_headers_case_insensitive(base, {})
        assert result == base
        assert result is not base

    def test_base_not_mutated(self):
        """Merging does not mutate the caller's base dict."""
        base = {"Authorization": "Bearer base"}
        merge_headers_case_insensitive(base, {"authorization": "Bearer override"})
        assert base == {"Authorization": "Bearer base"}

    def test_removes_all_case_equivalent_base_keys(self):
        """An override collapses every case variant of a name in the base into a single header."""
        base = {"Authorization": "base-a", "authorization": "base-b"}
        result = merge_headers_case_insensitive(base, {"Authorization": "override"})
        auth_keys = [key for key in result if key.lower() == "authorization"]
        assert auth_keys == ["Authorization"]
        assert result["Authorization"] == "override"


class TestNormalizeQueryParams:
    """Test coercion of configured default_query values into wire-form (name, value) pairs."""

    def test_none_yields_no_pairs(self):
        """An unset default_query normalizes to an empty tuple."""
        assert normalize_query_params(None) == ()

    def test_empty_mapping_yields_no_pairs(self):
        """An empty default_query block normalizes to an empty tuple."""
        assert normalize_query_params({}) == ()

    def test_string_value_yields_one_pair(self):
        """A string value becomes a single pair."""
        assert normalize_query_params({"tenant": "acme"}) == (("tenant", "acme"),)

    def test_numeric_values_stringify(self):
        """Int and float values are stringified."""
        assert normalize_query_params({"retries": 3, "ratio": 1.5}) == (("retries", "3"), ("ratio", "1.5"))

    def test_bool_values_lowercase(self):
        """Booleans use the wire form 'true'/'false' rather than Python's 'True'/'False'."""
        assert normalize_query_params({"beta": True, "cache": False}) == (("beta", "true"), ("cache", "false"))

    def test_none_value_becomes_empty_string(self):
        """A null value becomes a valueless parameter."""
        assert normalize_query_params({"flag": None}) == (("flag", ""),)

    def test_list_value_repeats_the_key_in_order(self):
        """A list value yields one pair per element, preserving order."""
        assert normalize_query_params({"tag": ["b", "a"]}) == (("tag", "b"), ("tag", "a"))

    def test_list_elements_are_coerced_individually(self):
        """Elements of a list value get the same scalar coercion as a bare value."""
        assert normalize_query_params({"tag": [True, 2, None]}) == (("tag", "true"), ("tag", "2"), ("tag", ""))

    def test_string_value_is_not_split_into_characters(self):
        """A string is treated as one scalar, not as a sequence of characters."""
        assert normalize_query_params({"tenant": "abc"}) == (("tenant", "abc"),)

    def test_bytes_value_is_not_split_into_ints(self):
        """A bytes value is treated as one scalar, not as a sequence of ints."""
        assert normalize_query_params({"raw": b"ab"}) == (("raw", "b'ab'"),)

    def test_non_string_keys_stringify(self):
        """A non-string key, such as a bare YAML number, is stringified."""
        assert normalize_query_params({2024: "x"}) == (("2024", "x"),)

    def test_key_order_is_preserved(self):
        """Pairs follow the configured key order."""
        assert normalize_query_params({"b": "1", "a": "2"}) == (("b", "1"), ("a", "2"))

    def test_nested_mapping_value_raises_naming_the_key(self):
        """A nested object value is rejected with a message naming the field and the offending key."""
        with pytest.raises(ValueError) as exc_info:
            normalize_query_params({"meta": {"region": "us"}})
        assert "default_query" in str(exc_info.value)
        assert "meta" in str(exc_info.value)

    def test_nested_mapping_inside_a_list_raises_naming_the_key(self):
        """A nested object inside a list value is rejected the same way as a bare one."""
        with pytest.raises(ValueError) as exc_info:
            normalize_query_params({"meta": ["ok", {"region": "us"}]})
        assert "meta" in str(exc_info.value)

    def test_non_mapping_query_raises(self):
        """A default_query that is not a mapping at all is rejected."""
        with pytest.raises(ValueError) as exc_info:
            normalize_query_params("tenant=acme")
        assert "default_query" in str(exc_info.value)


class TestSharedConstants:
    """Test values of shared HTTP constants."""

    def test_retryable_status_codes_is_frozenset(self):
        """Retryable codes include 429 and 5xx server errors."""
        assert isinstance(RETRYABLE_STATUS_CODES, frozenset)
        assert 429 in RETRYABLE_STATUS_CODES
        assert 500 in RETRYABLE_STATUS_CODES

    def test_default_timeout_total(self):
        assert DEFAULT_TIMEOUT_TOTAL == 30

    def test_default_timeout_connect(self):
        assert DEFAULT_TIMEOUT_CONNECT == 5

    def test_default_max_attempts(self):
        assert DEFAULT_MAX_ATTEMPTS == 3

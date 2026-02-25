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

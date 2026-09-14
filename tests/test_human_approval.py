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

from unittest.mock import MagicMock

import pytest

from nemoguardrails.library.human_approval.actions import (
    human_approval_check,
    human_approval_matches_keywords,
)
from nemoguardrails.rails.llm.config import HumanApprovalConfig, RailsConfigData


def _make_config(patterns=None, **overrides):
    config = MagicMock()
    kwargs = {}
    if patterns is not None:
        kwargs["patterns"] = patterns
    kwargs.update(overrides)
    config.rails.config = RailsConfigData(human_approval=HumanApprovalConfig(**kwargs))
    return config


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_default_values(self):
        cfg = HumanApprovalConfig()
        assert cfg.patterns == []
        assert cfg.approval_keywords == ["approve", "yes", "approved"]
        assert "approval" in cfg.approval_message
        assert "rejected" in cfg.rejection_message.lower()

    def test_custom_patterns(self):
        cfg = HumanApprovalConfig(patterns=["delete", "drop"])
        assert len(cfg.compiled_patterns) == 2

    def test_invalid_regex_raises(self):
        with pytest.raises(ValueError, match="Invalid regex"):
            HumanApprovalConfig(patterns=["[invalid"])

    def test_registered_in_rails_config_data(self):
        data = RailsConfigData()
        assert data.human_approval is not None


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


class TestPatternMatching:
    @pytest.mark.asyncio
    async def test_matches_pattern(self):
        config = _make_config(patterns=["delete|drop"])
        result = await human_approval_check("DROP TABLE users", config)
        assert result["needs_approval"] is True
        assert "delete|drop" in result["matched_patterns"]

    @pytest.mark.asyncio
    async def test_no_match(self):
        config = _make_config(patterns=["delete|drop"])
        result = await human_approval_check("SELECT * FROM users", config)
        assert result["needs_approval"] is False
        assert result["matched_patterns"] == []

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        config = _make_config(patterns=["sudo"])
        result = await human_approval_check("SUDO rm -rf /", config)
        assert result["needs_approval"] is True

    @pytest.mark.asyncio
    async def test_multiple_patterns_matched(self):
        config = _make_config(patterns=["delete", "sudo"])
        result = await human_approval_check("sudo delete everything", config)
        assert result["needs_approval"] is True
        assert len(result["matched_patterns"]) == 2

    @pytest.mark.asyncio
    async def test_empty_patterns(self):
        config = _make_config(patterns=[])
        result = await human_approval_check("DROP TABLE users", config)
        assert result["needs_approval"] is False

    @pytest.mark.asyncio
    async def test_empty_text(self):
        config = _make_config(patterns=["delete"])
        result = await human_approval_check("", config)
        assert result["needs_approval"] is False

    @pytest.mark.asyncio
    async def test_no_config(self):
        config = MagicMock()
        config.rails.config = RailsConfigData()
        result = await human_approval_check("DROP TABLE users", config)
        assert result["needs_approval"] is False

    @pytest.mark.asyncio
    async def test_text_returned_unchanged(self):
        config = _make_config(patterns=["delete"])
        text = "delete this record"
        result = await human_approval_check(text, config)
        assert result["text"] == text

    @pytest.mark.asyncio
    async def test_regex_pattern(self):
        config = _make_config(patterns=[r"rm\s+-rf"])
        result = await human_approval_check("rm -rf /home", config)
        assert result["needs_approval"] is True

    @pytest.mark.asyncio
    async def test_regex_no_match(self):
        config = _make_config(patterns=[r"rm\s+-rf"])
        result = await human_approval_check("remove files", config)
        assert result["needs_approval"] is False


class TestApprovalKeywords:
    @pytest.mark.asyncio
    async def test_matches_approve_keyword(self):
        config = _make_config()
        assert await human_approval_matches_keywords("approve", config) is True

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        config = _make_config()
        assert await human_approval_matches_keywords("YES", config) is True

    @pytest.mark.asyncio
    async def test_rejects_unknown_response(self):
        config = _make_config()
        assert await human_approval_matches_keywords("no", config) is False

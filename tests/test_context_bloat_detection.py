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

import logging
from unittest.mock import MagicMock

import pytest

from nemoguardrails.library.context_bloat_detection.actions import (
    _longest_run_ratio,
    _repetition_ratio,
    _shannon_entropy,
    _validate_config,
    context_bloat_detection,
)
from nemoguardrails.rails.llm.config import (
    ContextBloatDetectionConfig,
    RailsConfigData,
)


def _make_config(**overrides):
    config = MagicMock()
    config.rails.config = RailsConfigData(context_bloat_detection=ContextBloatDetectionConfig(**overrides))
    return config


# ---------------------------------------------------------------------------
# Config default values
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_default_values(self):
        cfg = ContextBloatDetectionConfig()
        assert cfg.max_chars == 5000
        assert cfg.min_entropy == 3.5
        assert cfg.max_repetition_ratio == 0.4
        assert cfg.max_run_ratio == 0.1
        assert cfg.ngram_size == 3
        assert cfg.action == "reject"

    def test_custom_values(self):
        cfg = ContextBloatDetectionConfig(max_chars=10000, ngram_size=4, action="truncate")
        assert cfg.max_chars == 10000
        assert cfg.ngram_size == 4
        assert cfg.action == "truncate"

    def test_registered_in_rails_config_data(self):
        data = RailsConfigData()
        assert data.context_bloat_detection is not None

    def test_accessible_when_configured(self):
        data = RailsConfigData(context_bloat_detection=ContextBloatDetectionConfig())
        assert data.context_bloat_detection is not None
        assert data.context_bloat_detection.max_chars == 5000


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_action_reject(self):
        _validate_config(_make_config(action="reject"))

    def test_valid_action_truncate(self):
        _validate_config(_make_config(action="truncate"))

    def test_valid_action_warn(self):
        _validate_config(_make_config(action="warn"))

    def test_invalid_action_raises_at_config_time(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="literal_error"):
            ContextBloatDetectionConfig(action="bad_value")

    def test_default_config_is_valid(self):
        config = MagicMock()
        config.rails.config = RailsConfigData()
        _validate_config(config)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestShannonEntropy:
    def test_empty_string(self):
        assert _shannon_entropy("") == 0.0

    def test_single_char_repeated(self):
        assert _shannon_entropy("aaaaaaa") == pytest.approx(0.0, abs=0.01)

    def test_normal_english(self):
        entropy = _shannon_entropy("The quick brown fox jumps over the lazy dog")
        assert entropy > 3.5

    def test_large_input_uses_sampling(self):
        text = "abcdefghij" * 2000
        entropy = _shannon_entropy(text)
        assert entropy > 0.0


class TestRepetitionRatio:
    def test_no_repetition(self):
        assert _repetition_ratio("one two three four five six") == 0.0

    def test_high_repetition(self):
        repeated = " ".join(["foo bar baz"] * 20)
        ratio = _repetition_ratio(repeated)
        assert ratio > 0.4

    def test_too_few_tokens(self):
        assert _repetition_ratio("one two") == 0.0

    def test_custom_ngram_size(self):
        text = "a b c d a b c d a b c d"
        ratio_3 = _repetition_ratio(text, n=3)
        ratio_4 = _repetition_ratio(text, n=4)
        assert ratio_3 > 0.0
        assert ratio_4 > 0.0


class TestLongestRunRatio:
    def test_empty_string(self):
        assert _longest_run_ratio("") == 0.0

    def test_no_runs(self):
        ratio = _longest_run_ratio("abcdef")
        assert ratio == pytest.approx(1 / 6)

    def test_long_run(self):
        ratio = _longest_run_ratio("a" * 90 + "bcdefghijk")
        assert ratio == 0.9

    def test_all_same(self):
        assert _longest_run_ratio("aaaa") == 1.0


# ---------------------------------------------------------------------------
# Main action — detection paths
# ---------------------------------------------------------------------------


class TestDetectSizeCap:
    @pytest.mark.asyncio
    async def test_reject_oversized(self):
        config = _make_config(max_chars=100, action="reject")
        result = await context_bloat_detection("x" * 200, config)
        assert result["is_bloat"] is True
        assert "size_cap_exceeded" in result["detections"]

    @pytest.mark.asyncio
    async def test_under_size_passes(self):
        config = _make_config(max_chars=1000, action="reject")
        result = await context_bloat_detection("short text", config)
        assert "size_cap_exceeded" not in result["detections"]


class TestDetectEntropy:
    @pytest.mark.asyncio
    async def test_low_entropy_detected(self):
        config = _make_config(max_chars=50000, min_entropy=3.5, action="reject")
        result = await context_bloat_detection("ab" * 500, config)
        assert result["is_bloat"] is True
        assert "low_entropy" in result["detections"]

    @pytest.mark.asyncio
    async def test_normal_entropy_passes(self):
        config = _make_config(max_chars=50000, min_entropy=3.5, action="reject")
        text = "The quick brown fox jumps over the lazy dog. " * 10
        result = await context_bloat_detection(text, config)
        assert "low_entropy" not in result["detections"]


class TestDetectLongRun:
    @pytest.mark.asyncio
    async def test_long_run_detected(self):
        config = _make_config(max_chars=50000, min_entropy=0.1, max_run_ratio=0.1, action="reject")
        padded = "a" * 90 + "hello world"
        result = await context_bloat_detection(padded, config)
        assert result["is_bloat"] is True
        assert "long_run" in result["detections"]

    @pytest.mark.asyncio
    async def test_short_run_passes(self):
        config = _make_config(max_chars=50000, min_entropy=0.1, max_run_ratio=0.5, action="reject")
        result = await context_bloat_detection("hello world", config)
        assert "long_run" not in result["detections"]


class TestDetectRepetition:
    @pytest.mark.asyncio
    async def test_high_repetition_detected(self):
        config = _make_config(
            max_chars=50000,
            min_entropy=1.0,
            max_run_ratio=0.5,
            max_repetition_ratio=0.4,
            action="warn",
        )
        repeated = " ".join(["foo bar baz"] * 50)
        result = await context_bloat_detection(repeated, config)
        assert result["is_bloat"] is True
        assert "high_repetition" in result["detections"]

    @pytest.mark.asyncio
    async def test_unique_text_passes(self):
        config = _make_config(
            max_chars=50000,
            min_entropy=1.0,
            max_run_ratio=0.5,
            max_repetition_ratio=0.4,
            action="reject",
        )
        result = await context_bloat_detection(
            "every single word here is different and unique in this sentence", config
        )
        assert "high_repetition" not in result["detections"]


# ---------------------------------------------------------------------------
# Action modes
# ---------------------------------------------------------------------------


class TestTruncateMode:
    @pytest.mark.asyncio
    async def test_truncates_to_max_chars(self):
        text = ("The quick brown fox jumps over the lazy dog. " * 10)[:200]
        config = _make_config(max_chars=50, action="truncate")
        result = await context_bloat_detection(text, config)
        assert result["is_bloat"] is True
        assert result["action"] == "truncate"
        assert len(result["text"]) == 50

    @pytest.mark.asyncio
    async def test_truncate_early_returns_on_entropy(self):
        config = _make_config(max_chars=50000, min_entropy=5.0, action="truncate")
        result = await context_bloat_detection("ab" * 500, config)
        assert result["is_bloat"] is True
        assert result["action"] == "reject"
        assert "low_entropy" in result["detections"]
        assert "longest_run_ratio" not in result["metrics"]


class TestWarnMode:
    @pytest.mark.asyncio
    async def test_warn_does_not_block(self, caplog):
        config = _make_config(max_chars=10, action="warn")
        with caplog.at_level(logging.INFO):
            result = await context_bloat_detection("x" * 200, config)
        assert result["is_bloat"] is True
        assert result["action"] == "warn"
        assert result["text"] == "x" * 200
        assert any("context bloat detected" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_warn_continues_all_checks(self):
        config = _make_config(
            max_chars=10,
            min_entropy=5.0,
            max_run_ratio=0.01,
            max_repetition_ratio=0.01,
            action="warn",
        )
        repeated = " ".join(["foo bar baz"] * 50)
        result = await context_bloat_detection(repeated, config)
        assert result["is_bloat"] is True
        assert result["action"] == "warn"
        assert len(result["detections"]) > 1


# ---------------------------------------------------------------------------
# Normal text passes all checks
# ---------------------------------------------------------------------------


class TestNormalTextPasses:
    @pytest.mark.asyncio
    async def test_normal_text_not_flagged(self):
        config = _make_config()
        result = await context_bloat_detection("Hello, how can I help you today?", config)
        assert result["is_bloat"] is False
        assert result["reason"] is None
        assert result["detections"] == []

    @pytest.mark.asyncio
    async def test_short_messages_not_flagged(self):
        config = _make_config()
        for msg in ["Hi", "Hello", "Yes", "Yes please", "No thanks"]:
            result = await context_bloat_detection(msg, config)
            assert result["is_bloat"] is False, f"'{msg}' should not be flagged"

    @pytest.mark.asyncio
    async def test_moderate_text_not_flagged(self):
        config = _make_config()
        text = (
            "The weather today is sunny with a high of 75 degrees. "
            "Scientists have discovered a new species of butterfly in the Amazon. "
            "The stock market closed higher on strong earnings reports from tech companies. "
            "A new study suggests that regular exercise can improve cognitive function. "
        )
        result = await context_bloat_detection(text, config)
        assert result["is_bloat"] is False

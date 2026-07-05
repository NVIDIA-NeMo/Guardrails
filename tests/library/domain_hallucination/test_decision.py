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

"""Tests for decision module."""

import unittest

from nemoguardrails.library.domain_hallucination import decision
from nemoguardrails.library.domain_hallucination.config import (
    DomainHallucinationGuardConfig,
    EnforcementConfig,
    get_config,
    set_config,
)


class TestDecisionEngine(unittest.TestCase):
    """Test pass/warn/refine/block decisions."""

    def test_make_decision_block(self):
        """Test block decision at high score."""
        result = decision.make_decision({"score": 85.0, "level": "L4"})
        assert result["action"] == "block"
        assert result["verification_level"] == "dns"

    def test_make_decision_refine(self):
        """Test refine decision in middle range."""
        result = decision.make_decision({"score": 45.0, "level": "L2"})
        assert result["action"] == "refine"

    def test_make_decision_warn(self):
        """Test warn decision in low-risk range."""
        result = decision.make_decision({"score": 25.0, "level": "L1"})
        assert result["action"] == "warn"

    def test_make_decision_pass(self):
        """Test pass decision below warn threshold."""
        result = decision.make_decision({"score": 10.0, "level": "L0"})
        assert result["action"] == "pass"

    def test_make_decision_downgrades_without_verification(self):
        """Test no-verification mode downgrades severe actions to warn."""
        result = decision.make_decision(
            {"score": 90.0, "level": "L4"},
            verification_level="none",
        )
        assert result["action"] == "warn"
        assert "downgraded" in result["reason"]

    def test_make_decision_uses_recalibrated_score(self):
        """Test recalibrated score overrides raw score."""
        result = decision.make_decision(
            {"score": 85.0, "level": "L4"},
            recalibrated_score={"recalibrated_score": 15.0, "recalibrated_level": "L0"},
        )
        assert result["action"] == "pass"
        assert result["level"] == "L0"
        assert result["score"] == 15.0

    def test_make_decision_keeps_recalibrated_zero_score(self):
        """Test recalibrated 0.0 is treated as a real score."""
        result = decision.make_decision(
            {"score": 85.0, "level": "L4"},
            recalibrated_score={"recalibrated_score": 0.0, "recalibrated_level": "L0"},
        )
        assert result["action"] == "pass"
        assert result["level"] == "L0"
        assert result["score"] == 0.0

    def test_make_decision_full_verification_keeps_threshold_logic(self):
        """Test full verification mode does not bypass configured thresholds."""
        result = decision.make_decision(
            {"score": 70.0, "level": "L3"},
            verification_level="full",
        )
        assert result["action"] == "block"

    def test_make_decision_http_verification_level(self):
        """Test HTTP verification level preserves threshold-based action."""
        result = decision.make_decision(
            {"score": 50.0, "level": "L2"},
            verification_level="http",
        )
        assert result["action"] == "refine"
        assert result["verification_level"] == "http"


class TestDecisionApplication(unittest.TestCase):
    """Test answer enforcement output."""

    def setUp(self):
        self._original_config = get_config()

    def tearDown(self):
        set_config(self._original_config)

    def test_apply_block(self):
        """Test block output formatting."""
        result = decision.apply_decision({"action": "block", "reason": "unsafe"}, answer="hello")
        assert result["enforced"] is True
        assert result["modified_answer"].startswith("[BLOCKED]")

    def test_apply_refine(self):
        """Test refine output formatting."""
        result = decision.apply_decision({"action": "refine", "reason": "review"}, answer="hello")
        assert result["enforced"] is True
        assert "[Refined by domain guard]" in result["modified_answer"]

    def test_apply_warn(self):
        """Test warn output formatting."""
        result = decision.apply_decision({"action": "warn", "reason": "review"}, answer="hello")
        assert result["enforced"] is True
        assert result["modified_answer"].startswith("[WARNING]")

    def test_apply_decision_uses_enforcement_config_messages(self):
        """Test enforcement messages are sourced from config."""
        set_config(
            DomainHallucinationGuardConfig(
                enforcement=EnforcementConfig(
                    block_message="[CUSTOM BLOCK]",
                    refine_message="[CUSTOM REFINE]",
                    warn_message="[CUSTOM WARN]",
                    append_verification_notice=False,
                )
            )
        )

        block = decision.apply_decision({"action": "block", "reason": "unsafe"}, answer="hello")
        refine = decision.apply_decision({"action": "refine", "reason": "review"}, answer="hello")
        warn = decision.apply_decision({"action": "warn", "reason": "review"}, answer="hello")

        assert block["modified_answer"] == "[CUSTOM BLOCK]"
        assert refine["modified_answer"] == "[CUSTOM REFINE]\n\nhello"
        assert warn["modified_answer"] == "[CUSTOM WARN]\n\nhello"

    def test_apply_pass(self):
        """Test pass keeps original answer intact."""
        result = decision.apply_decision({"action": "pass", "reason": "safe"}, answer="hello")
        assert result["enforced"] is False
        assert result["modified_answer"] == "hello"


if __name__ == "__main__":
    unittest.main()

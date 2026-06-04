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

    def test_make_decision_full_verification_passes_below_80(self):
        """Test full verification mode relaxes sub-80 scores."""
        result = decision.make_decision(
            {"score": 65.0, "level": "L3"},
            verification_level="full",
        )
        assert result["action"] == "pass"
        assert result["reason"] == "Full verification required but score below threshold"


class TestDecisionApplication(unittest.TestCase):
    """Test answer enforcement output."""

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

    def test_apply_pass(self):
        """Test pass keeps original answer intact."""
        result = decision.apply_decision({"action": "pass", "reason": "safe"}, answer="hello")
        assert result["enforced"] is False
        assert result["modified_answer"] == "hello"


if __name__ == "__main__":
    unittest.main()

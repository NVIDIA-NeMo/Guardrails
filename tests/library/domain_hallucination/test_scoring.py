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

"""Tests for scoring module."""

import unittest

from nemoguardrails.library.domain_hallucination import scoring


class TestRiskScoring(unittest.TestCase):
    """Test risk scoring."""

    def test_score_no_issues(self):
        """Test scoring with no issues."""
        detection = {
            "has_issues": False,
            "issues": [],
            "issue_summary": {"total": 0},
        }
        result = scoring.calculate_risk_score(detection)
        assert result["score"] == 0.0
        assert result["level"] == "L0"

    def test_score_single_high_issue(self):
        """Test scoring with single high severity issue."""
        detection = {
            "has_issues": True,
            "issues": [
                {
                    "type": "non_existent_domain",
                    "severity": "high",
                    "confidence": "high",
                    "target": "fake.com",
                }
            ],
            "issue_summary": {"total": 1},
        }
        result = scoring.calculate_risk_score(detection)
        assert result["score"] > 0
        assert result["level"] != "L0"

    def test_score_multiple_issues(self):
        """Test scoring with multiple issues."""
        detection = {
            "has_issues": True,
            "issues": [
                {
                    "type": "fake_github_repo",
                    "severity": "high",
                    "confidence": "high",
                    "target": "fake/repo",
                },
                {
                    "type": "non_existent_domain",
                    "severity": "high",
                    "confidence": "high",
                    "target": "fake.com",
                },
            ],
            "issue_summary": {"total": 2},
        }
        result = scoring.calculate_risk_score(detection)
        assert result["score"] > 60  # Should be in high range

    def test_score_to_level(self):
        """Test score to level conversion."""
        assert scoring._score_to_level(0.0) == ("L0", "Normal")
        assert scoring._score_to_level(20.0) == ("L1", "Low")
        assert scoring._score_to_level(40.0) == ("L2", "Medium")
        assert scoring._score_to_level(60.0) == ("L3", "High")
        assert scoring._score_to_level(80.0) == ("L4", "Critical")

    def test_calculate_bonus_critical_issues(self):
        """Test bonus calculation with critical severity issues."""
        detection = {
            "has_issues": True,
            "issues": [
                {
                    "type": "fake_github_repo",
                    "severity": "critical",
                    "confidence": "high",
                },
                {
                    "type": "fake_github_repo",
                    "severity": "critical",
                    "confidence": "high",
                },
            ],
            "issue_summary": {"total": 2, "by_severity": {"critical": 2}},
        }
        result = scoring.calculate_risk_score(detection)
        assert result["bonus"] == 20.0
        assert result["score"] == 100.0

    def test_calculate_bonus_high_issues(self):
        """Test bonus calculation with three high severity issues."""
        detection = {
            "has_issues": True,
            "issues": [
                {"type": "recent_domain", "severity": "high", "confidence": "low"},
                {
                    "type": "tls_certificate_expiring_soon",
                    "severity": "high",
                    "confidence": "low",
                },
                {
                    "type": "no_local_kb_evidence",
                    "severity": "high",
                    "confidence": "low",
                },
            ],
            "issue_summary": {"total": 3, "by_severity": {"high": 3}},
        }
        result = scoring.calculate_risk_score(detection)
        assert result["bonus"] == 5.0

    def test_calculate_bonus_combined_types(self):
        """Test bonus for combined GitHub and domain failures."""
        detection = {
            "has_issues": True,
            "issues": [
                {"type": "fake_github_repo", "severity": "low", "confidence": "low"},
                {"type": "non_existent_domain", "severity": "low", "confidence": "low"},
            ],
            "issue_summary": {"total": 2},
        }
        result = scoring.calculate_risk_score(detection)
        assert result["bonus"] == 5.0


class TestRecalibration(unittest.TestCase):
    """Test score recalibration."""

    def test_recalibration_with_successful_verification(self):
        """Test recalibrating with successful verification."""
        risk_score = {"score": 50.0, "level": "L2"}
        verification = {
            "dns": [{"resolves": True}],
            "http": [{"reachable": True}],
            "github": [{"exists": True}],
        }

        recalibrated = scoring.recalibrate_score(risk_score, verification_results=verification)
        assert recalibrated["recalibrated_score"] < risk_score["score"]

    def test_recalibration_no_change(self):
        """Test recalibration with no verification."""
        risk_score = {"score": 50.0, "level": "L2"}
        recalibrated = scoring.recalibrate_score(risk_score)
        assert recalibrated["recalibrated_score"] == 50.0


if __name__ == "__main__":
    unittest.main()

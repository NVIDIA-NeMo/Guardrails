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


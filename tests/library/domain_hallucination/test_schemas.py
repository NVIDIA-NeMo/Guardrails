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

"""Tests for domain hallucination schema helpers."""

from nemoguardrails.library.domain_hallucination.schemas import (
    Decision,
    DetectionResult,
    Issue,
    RiskScore,
)


def test_issue_minimal_to_dict():
    issue = Issue("dns_failure", "domain", "example.invalid")

    assert issue.to_dict() == {
        "type": "dns_failure",
        "target_type": "domain",
        "target": "example.invalid",
        "severity": "low",
        "confidence": "low",
        "evidence_source": "",
        "evidence": {},
        "message": "",
        "signals": [],
    }


def test_issue_full_to_dict():
    issue = Issue(
        "github_missing",
        "github_repo",
        "owner/repo",
        severity="high",
        confidence="high",
        evidence_source="github",
        evidence={"exists": False},
        message="Repository not found",
        signals=["repo_not_found"],
    )

    data = issue.to_dict()

    assert data["severity"] == "high"
    assert data["confidence"] == "high"
    assert data["evidence"] == {"exists": False}
    assert data["signals"] == ["repo_not_found"]


def test_detection_result_defaults_and_custom_values():
    default = DetectionResult().to_dict()
    custom = DetectionResult(
        has_issues=True,
        issues=[{"type": "dns_failure"}],
        issue_summary={"total": 1, "highest_severity": "high"},
    ).to_dict()

    assert default["has_issues"] is False
    assert default["issues"] == []
    assert default["issue_summary"]["highest_severity"] == "none"
    assert custom["has_issues"] is True
    assert custom["issues"] == [{"type": "dns_failure"}]
    assert custom["issue_summary"]["total"] == 1


def test_risk_score_defaults_and_custom_values():
    default = RiskScore().to_dict()
    custom = RiskScore(
        score=75.0,
        raw_score=70.0,
        level="L3",
        label="High",
        bonus=5.0,
        score_details=[{"reason": "missing_repo"}],
    ).to_dict()

    assert default == {
        "score": 0.0,
        "raw_score": 0.0,
        "level": "L0",
        "label": "Normal",
        "bonus": 0.0,
        "score_details": [],
    }
    assert custom["score"] == 75.0
    assert custom["score_details"] == [{"reason": "missing_repo"}]


def test_decision_defaults_and_custom_values():
    default = Decision().to_dict()
    custom = Decision(action="block", reason="high risk", level="L3").to_dict()

    assert default == {"action": "pass", "reason": "", "level": "L0"}
    assert custom == {"action": "block", "reason": "high risk", "level": "L3"}

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

"""Data schemas for domain hallucination detection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class Issue:
    """Represents a domain hallucination issue."""

    def __init__(
        self,
        issue_type: str,
        target_type: str,
        target: str,
        severity: str = "low",
        confidence: str = "low",
        evidence_source: str = "",
        evidence: Any = None,
        message: str = "",
        signals: Optional[List[str]] = None,
    ):
        self.type = issue_type
        self.target_type = target_type
        self.target = target
        self.severity = severity
        self.confidence = confidence
        self.evidence_source = evidence_source
        self.evidence = evidence or {}
        self.message = message
        self.signals = signals or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "target_type": self.target_type,
            "target": self.target,
            "severity": self.severity,
            "confidence": self.confidence,
            "evidence_source": self.evidence_source,
            "evidence": self.evidence,
            "message": self.message,
            "signals": self.signals,
        }


class DetectionResult:
    """Result from domain hallucination detection."""

    def __init__(
        self,
        has_issues: bool = False,
        issues: Optional[List[Dict[str, Any]]] = None,
        issue_summary: Optional[Dict[str, Any]] = None,
    ):
        self.has_issues = has_issues
        self.issues = issues or []
        self.issue_summary = issue_summary or {
            "total": 0,
            "by_type": {},
            "by_severity": {},
            "highest_severity": "none",
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_issues": self.has_issues,
            "issues": self.issues,
            "issue_summary": self.issue_summary,
        }


class RiskScore:
    """Risk scoring result."""

    def __init__(
        self,
        score: float = 0.0,
        raw_score: float = 0.0,
        level: str = "L0",
        label: str = "Normal",
        bonus: float = 0.0,
        score_details: Optional[List[Dict[str, Any]]] = None,
    ):
        self.score = score
        self.raw_score = raw_score
        self.level = level
        self.label = label
        self.bonus = bonus
        self.score_details = score_details or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "raw_score": self.raw_score,
            "level": self.level,
            "label": self.label,
            "bonus": self.bonus,
            "score_details": self.score_details,
        }


class Decision:
    """Enforcement decision."""

    def __init__(
        self,
        action: str = "pass",
        reason: str = "",
        level: str = "L0",
    ):
        self.action = action
        self.reason = reason
        self.level = level

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "level": self.level,
        }

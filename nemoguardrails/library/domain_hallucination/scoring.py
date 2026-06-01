"""Risk scoring and recalibration for domain hallucination."""

from __future__ import annotations

from typing import Any, Dict, List


ISSUE_TYPE_SCORES = {
    "fake_github_repo": 85,
    "non_existent_domain": 80,
    "delegated_no_address_record": 50,
    "blacklisted_domain": 95,
    "recent_domain": 20,
    "no_local_kb_evidence": 15,
    "semantic_mismatch": 30,
    "advanced_verification_failed": 40,
}

SEVERITY_WEIGHTS = {
    "critical": 1.5,
    "high": 1.3,
    "medium": 1.0,
    "low": 0.7,
}

CONFIDENCE_BOOSTS = {
    "high": 1.0,
    "medium": 0.8,
    "low": 0.6,
}


def calculate_risk_score(detection_result: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate risk score from detection result."""
    issues = detection_result.get("issues", []) or []
    issue_summary = detection_result.get("issue_summary", {}) or {}

    if not issues:
        return {
            "score": 0.0,
            "raw_score": 0.0,
            "level": "L0",
            "label": "Normal",
            "bonus": 0.0,
            "score_details": [],
        }

    score_details: List[Dict[str, Any]] = []
    total_score = 0.0

    for issue in issues:
        issue_type = str(issue.get("type", "unknown"))
        severity = str(issue.get("severity", "low"))
        confidence = str(issue.get("confidence", "low"))

        base_score = float(ISSUE_TYPE_SCORES.get(issue_type, 25))
        severity_weight = float(SEVERITY_WEIGHTS.get(severity, 1.0))
        confidence_boost = float(CONFIDENCE_BOOSTS.get(confidence, 0.6))

        weighted_score = base_score * severity_weight * confidence_boost
        total_score += weighted_score

        score_details.append({
            "type": issue_type,
            "base_score": base_score,
            "severity_weight": severity_weight,
            "confidence_boost": confidence_boost,
            "weighted_score": weighted_score,
            "target": issue.get("target", ""),
        })

    raw_score = min(100.0, total_score)
    bonus = _calculate_bonus(issues, issue_summary)
    final_score = min(100.0, raw_score + bonus)

    level, label = _score_to_level(final_score)

    return {
        "score": round(final_score, 2),
        "raw_score": round(raw_score, 2),
        "level": level,
        "label": label,
        "bonus": round(bonus, 2),
        "score_details": score_details,
    }


def _calculate_bonus(issues: List[Dict[str, Any]], summary: Dict[str, Any]) -> float:
    """Calculate bonus adjustments."""
    bonus = 0.0

    critical_count = summary.get("by_severity", {}).get("critical", 0)
    if critical_count > 0:
        bonus += 10.0 * critical_count

    high_count = summary.get("by_severity", {}).get("high", 0)
    if high_count >= 3:
        bonus += 5.0

    issue_types = {str(issue.get("type")) for issue in issues if isinstance(issue, dict)}
    if {"fake_github_repo", "non_existent_domain"}.issubset(issue_types):
        bonus += 5.0

    return min(20.0, bonus)


def _score_to_level(score: float) -> tuple[str, str]:
    """Convert numeric score to risk level."""
    if score >= 80:
        return "L4", "Critical"
    elif score >= 60:
        return "L3", "High"
    elif score >= 40:
        return "L2", "Medium"
    elif score >= 20:
        return "L1", "Low"
    else:
        return "L0", "Normal"


def recalibrate_score(
    risk_score: Dict[str, Any],
    verification_results: Dict[str, Any] | None = None,
    rag_results: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Recalibrate risk score based on additional evidence."""
    verification_results = verification_results or {}
    rag_results = rag_results or {}

    score = float(risk_score.get("score", 0.0))
    adjustments: List[Dict[str, Any]] = []

    # DNS success reduces score
    dns_results = verification_results.get("dns", []) or []
    successful_dns = sum(1 for r in dns_results if r.get("resolves") is True)
    if successful_dns > 0:
        reduction = successful_dns * 10.0
        score = max(0.0, score - reduction)
        adjustments.append({
            "type": "dns_success",
            "count": successful_dns,
            "reduction": reduction,
            "new_score": score,
        })

    # HTTP success reduces score
    http_results = verification_results.get("http", []) or []
    successful_http = sum(1 for r in http_results if r.get("reachable") is True)
    if successful_http > 0:
        reduction = successful_http * 15.0
        score = max(0.0, score - reduction)
        adjustments.append({
            "type": "http_success",
            "count": successful_http,
            "reduction": reduction,
            "new_score": score,
        })

    # GitHub success reduces score
    github_results = verification_results.get("github", []) or []
    existing_repos = sum(1 for r in github_results if r.get("exists") is True)
    if existing_repos > 0:
        reduction = existing_repos * 20.0
        score = max(0.0, score - reduction)
        adjustments.append({
            "type": "github_success",
            "count": existing_repos,
            "reduction": reduction,
            "new_score": score,
        })

    # KB evidence boosts confidence (slight reduction)
    kb_evidence = rag_results.get("domain_evidence", {}) or {}
    evidence_count = len(kb_evidence)
    if evidence_count > 0:
        reduction = min(10.0, evidence_count * 2.0)
        score = max(0.0, score - reduction)
        adjustments.append({
            "type": "kb_evidence",
            "count": evidence_count,
            "reduction": reduction,
            "new_score": score,
        })

    recalibrated_score = round(max(0.0, min(100.0, score)), 2)
    level, label = _score_to_level(recalibrated_score)

    return {
        "original_score": risk_score.get("score", 0.0),
        "recalibrated_score": recalibrated_score,
        "level": level,
        "label": label,
        "adjustments": adjustments,
        "original_level": risk_score.get("level", "L0"),
        "recalibrated_level": level,
    }

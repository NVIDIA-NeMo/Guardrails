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

"""Expert semantic review using the NeMo-configured LLM."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

try:  # pragma: no cover - available inside NeMo runtime
    from nemoguardrails.actions.llm.utils import llm_call
except Exception:  # pragma: no cover - import fallback for standalone tests
    llm_call = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

RISK_VALUES = {"low", "medium", "high", "critical", "unknown"}
RELEVANCE_VALUES = {"match", "partial_match", "mismatch", "unknown"}

EXPERT_REVIEW_INSTRUCTIONS = """You are a domain trust expert.
Judge whether the model answer misleads the user about domains, URLs, official
sources, GitHub repositories, downloads, login/payment flows, or other
security-sensitive actions.

Use only the provided evidence. Do not claim live browsing. Return strict JSON
only, with no Markdown and no hidden reasoning.

Required JSON fields:
{
  "semantic_risk": "low|medium|high|critical|unknown",
  "is_answer_misleading": true|false|null,
  "domain_relevance": "match|partial_match|mismatch|unknown",
  "official_candidates": [{"domain": "string", "confidence": 0.0, "source": "string"}],
  "corrected_answer": "string|null",
  "explanation": "string"
}
Keep explanation under two sentences.
""".strip()


def build_disabled_review(
    reason: str = "Expert review disabled; rule-based decision used.",
) -> Dict[str, Any]:
    return {
        "enabled": False,
        "status": "disabled",
        "semantic_risk": "unknown",
        "is_answer_misleading": None,
        "domain_relevance": "unknown",
        "official_candidates": [],
        "corrected_answer": None,
        "explanation": reason,
    }


def build_expert_review_payload(
    *,
    question: str,
    model_answer: str,
    extracted: Dict[str, Any],
    verification_result: Dict[str, Any],
    rag_result: Dict[str, Any],
    detection_result: Dict[str, Any],
    score: float,
    level: str,
    decision: Dict[str, Any],
) -> Dict[str, Any]:
    domains = sorted(
        {
            str(item.get("host") or item.get("registered_domain") or "").strip().lower()
            for item in extracted.get("domains", [])
            if isinstance(item, dict) and str(item.get("host") or item.get("registered_domain") or "").strip()
        }
    )
    urls = sorted(
        {
            str(item.get("normalized") or item.get("raw") or "").strip()
            for item in extracted.get("urls", [])
            if isinstance(item, dict) and str(item.get("normalized") or item.get("raw") or "").strip()
        }
    )
    return {
        "question": question,
        "model_answer": model_answer,
        "extracted_domains": domains,
        "extracted_urls": urls,
        "github_repos": _compact_github_repos(extracted.get("github_repos", [])),
        "domain_evidence": _domain_evidence(domains, verification_result, rag_result),
        "local_detection_issues": _compact_issues(detection_result.get("issues", [])),
        "rule_score": score,
        "rule_level": level,
        "rule_decision": decision,
    }


async def review_with_nemo_llm(
    *,
    llm: Any,
    question: str,
    model_answer: str,
    extracted: Dict[str, Any],
    verification_result: Dict[str, Any],
    rag_result: Dict[str, Any],
    detection_result: Dict[str, Any],
    score: float,
    level: str,
    decision: Dict[str, Any],
    llm_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run expert semantic review through NeMo's configured LLM."""
    if llm is None or llm_call is None:
        return build_disabled_review("NeMo LLM is unavailable; rule-based decision used.")

    payload = build_expert_review_payload(
        question=question,
        model_answer=model_answer,
        extracted=extracted,
        verification_result=verification_result,
        rag_result=rag_result,
        detection_result=detection_result,
        score=score,
        level=level,
        decision=decision,
    )
    prompt = f"{EXPERT_REVIEW_INSTRUCTIONS}\n\nEvidence JSON:\n{json.dumps(payload, ensure_ascii=False)}"

    try:
        response = await llm_call(
            llm,
            prompt,
            llm_params={"temperature": 0, "max_tokens": 700, **(llm_params or {})},
        )
        text = str(getattr(response, "content", response) or "").strip()
        return _normalize_review(_extract_json_object(text))
    except json.JSONDecodeError:
        return _fallback(
            "invalid_json",
            "Expert model returned invalid JSON; rule-based decision used.",
        )
    except Exception as exc:  # pragma: no cover - provider exceptions vary
        log.warning("Expert domain review failed: %s", exc)
        status = "timeout" if "timeout" in type(exc).__name__.lower() else "failed"
        return _fallback(
            status,
            f"Expert model failed: {type(exc).__name__}; rule-based decision used.",
        )


def apply_expert_decision(decision: Dict[str, Any], expert_review: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade or enrich rule decision with expert semantic review.

    Expert review is treated as advisory semantic evidence: it may increase the
    enforcement level, but it must not downgrade rule-based or verification-based
    hard evidence such as an existing block decision.
    """
    if expert_review.get("status") != "success":
        return decision

    updated = dict(decision)
    risk = str(expert_review.get("semantic_risk") or "unknown").lower()
    relevance = str(expert_review.get("domain_relevance") or "unknown").lower()
    misleading = expert_review.get("is_answer_misleading") is True
    original_action = str(updated.get("action") or "pass").lower()

    if risk == "critical":
        _upgrade_action(
            updated,
            "block",
            "Expert semantic review found critical misleading domain guidance.",
        )
    elif risk == "high" and misleading:
        _upgrade_action(
            updated,
            "refine",
            "Expert semantic review found high-risk misleading domain guidance.",
        )
    elif relevance == "mismatch" and updated.get("action") in {
        "warn",
        "refine",
        "block",
    }:
        _upgrade_action(
            updated,
            "refine",
            "Expert semantic review found domain relevance mismatch.",
        )

    if original_action == "block" and updated.get("action") == "block":
        updated["expert_policy"] = "advisory_preserve_block"

    return updated


def _upgrade_action(decision: Dict[str, Any], candidate_action: str, reason: str) -> None:
    action_order = {"pass": 0, "warn": 1, "refine": 2, "block": 3}
    current_action = str(decision.get("action") or "pass").lower()
    if action_order.get(candidate_action, 0) > action_order.get(current_action, 0):
        decision["action"] = candidate_action
        decision["reason"] = reason
        decision["expert_policy"] = "advisory_upgrade_only"
    else:
        decision["expert_policy"] = "advisory_no_downgrade"


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped:
        raise json.JSONDecodeError("empty expert response", stripped, 0)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(stripped[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("expert response is not a JSON object")
    return data


def _normalize_review(data: Dict[str, Any]) -> Dict[str, Any]:
    semantic_risk = str(data.get("semantic_risk") or "unknown").strip().lower()
    if semantic_risk not in RISK_VALUES:
        semantic_risk = "unknown"

    domain_relevance = str(data.get("domain_relevance") or "unknown").strip().lower()
    if domain_relevance not in RELEVANCE_VALUES:
        domain_relevance = "unknown"

    misleading = data.get("is_answer_misleading")
    if misleading is not True and misleading is not False:
        misleading = None

    corrected = data.get("corrected_answer")
    corrected_answer = str(corrected).strip() if corrected is not None and str(corrected).strip() else None

    return {
        "enabled": True,
        "status": "success",
        "semantic_risk": semantic_risk,
        "is_answer_misleading": misleading,
        "domain_relevance": domain_relevance,
        "official_candidates": _normalize_candidates(data.get("official_candidates", [])),
        "corrected_answer": corrected_answer,
        "explanation": str(data.get("explanation") or "").strip(),
    }


def _fallback(status: str, explanation: str) -> Dict[str, Any]:
    return {
        "enabled": True,
        "status": status,
        "semantic_risk": "unknown",
        "is_answer_misleading": None,
        "domain_relevance": "unknown",
        "official_candidates": [],
        "corrected_answer": None,
        "explanation": explanation,
    }


def _compact_github_repos(items: Any) -> List[Dict[str, Any]]:
    repos: List[Dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        owner = str(item.get("owner") or "").strip()
        repo = str(item.get("repo") or "").strip()
        if owner and repo:
            repos.append({"owner": owner, "repo": repo, "url": item.get("url", "")})
    return repos


def _domain_evidence(
    domains: List[str], verification_result: Dict[str, Any], rag_result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    dns_map = _by_domain(verification_result.get("dns", []), "domain")
    http_map = _by_domain(verification_result.get("http", []), "domain")
    rag_map = rag_result.get("domain_evidence", {}) if isinstance(rag_result.get("domain_evidence"), dict) else {}
    evidence = []
    for domain in domains:
        evidence.append(
            {
                "domain": domain,
                "dns": _compact_status(
                    dns_map.get(domain, {}),
                    ["status", "resolves", "public_address_count"],
                ),
                "http": _compact_status(
                    http_map.get(domain, {}),
                    ["status", "reachable", "status_code", "final_url"],
                ),
                "kb_evidence": rag_map.get(domain, []),
            }
        )
    return evidence


def _by_domain(items: Any, key: str) -> Dict[str, Dict[str, Any]]:
    mapped: Dict[str, Dict[str, Any]] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        domain = str(item.get(key) or item.get("target") or "").strip().lower()
        if domain:
            mapped[domain] = item
    return mapped


def _compact_status(item: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return {key: item.get(key) for key in keys if key in item}


def _compact_issues(issues: Any) -> List[Dict[str, Any]]:
    compacted = []
    for issue in issues if isinstance(issues, list) else []:
        if isinstance(issue, dict):
            compacted.append(
                {
                    "type": issue.get("type"),
                    "target_type": issue.get("target_type"),
                    "target": issue.get("target"),
                    "severity": issue.get("severity"),
                    "confidence": issue.get("confidence"),
                    "signals": issue.get("signals", []),
                    "message": issue.get("message"),
                }
            )
    return compacted


def _normalize_candidates(value: Any) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in value if isinstance(value, list) else []:
        if isinstance(item, str):
            domain = item.strip().lower()
            confidence = 0.5
            source = "expert_model"
        elif isinstance(item, dict):
            domain = str(item.get("domain") or "").strip().lower()
            try:
                confidence = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            source = str(item.get("source") or "expert_model").strip()
        else:
            continue
        if not domain or domain in seen:
            continue
        seen.add(domain)
        result.append(
            {
                "domain": domain,
                "confidence": max(0.0, min(1.0, confidence)),
                "source": source,
            }
        )
    return result

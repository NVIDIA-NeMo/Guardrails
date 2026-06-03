"""NeMo Guardrails actions for domain hallucination detection."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:  # pragma: no cover - available in NeMo runtime
    from nemoguardrails.actions import action
except Exception:  # pragma: no cover - standalone import fallback
    def action(*_args: Any, **_kwargs: Any):
        def decorator(fn):
            return fn

        return decorator

from . import checkers, decision, expert_review, extractors, kb, scoring, semantic, verification

logger = logging.getLogger(__name__)


_VERIFICATION_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "dns": {},
    "http": {},
    "tls": {},
    "whois": {},
    "github": {},
}


def _cache_get(source: str, key: str) -> Optional[Dict[str, Any]]:
    cached = _VERIFICATION_CACHE.get(source, {}).get(key)
    if cached is None:
        return None
    result = dict(cached)
    result["cache_hit"] = True
    return result


def _cache_set(source: str, key: str, value: Dict[str, Any]) -> Dict[str, Any]:
    stored = dict(value)
    stored["cache_hit"] = False
    _VERIFICATION_CACHE.setdefault(source, {})[key] = stored
    return dict(stored)


def _cached_verification(source: str, key: str, fn, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    normalized_key = str(key or "").strip().lower()
    cached = _cache_get(source, normalized_key)
    if cached is not None:
        return cached
    return _cache_set(source, normalized_key, fn(*args, **kwargs))


async def analyze_answer(
    answer: str = "",
    user_query: str = "",
    verification_level: str = "dns",
    enable_semantic_check: bool = False,
    enable_advanced_verification: bool = False,
    skip_secondary_checks_on_dns_failure: bool = False,
    enable_tls_verification: bool = True,
    enable_whois_verification: bool = True,
    kb_instance: Optional[kb.KnowledgeBase] = None,
    github_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Main action for domain hallucination detection and enforcement.

    This action:
    1. Extracts URLs, domains, and GitHub repos
    2. Performs verification (DNS/HTTP/TLS/WHOIS/GitHub)
    3. Aggregates detection issues
    4. Calculates risk score
    5. Makes enforcement decision
    6. Optionally applies answer modification

    Args:
        answer: LLM-generated answer
        user_query: Original user query (for semantic checks)
        verification_level: One of "none", "dns", "http", "full"
        enable_semantic_check: Enable semantic relevance checking
        enable_advanced_verification: Enable typosquatting and protocol checks
        skip_secondary_checks_on_dns_failure: Skip HTTP/TLS/WHOIS when DNS already fails
        enable_tls_verification: Enable TLS checks in full mode
        enable_whois_verification: Enable WHOIS/RDAP checks in full mode
        kb_instance: Custom KB instance (uses global if None)
        github_token: GitHub API token for rate limit increase

    Returns:
        Dict with detection result, risk score, decision, and modified answer
    """
    if not answer:
        return {
            "status": "skipped",
            "reason": "empty_answer",
            "decision": {"action": "pass", "reason": "empty answer"},
        }

    # Step 1: Extract entities
    extracted = extractors.extract_all(answer, source="model_answer")

    if extracted.get("no_links"):
        return {
            "status": "fast_pass",
            "reason": "no_links_detected",
            "extraction": extracted,
            "decision": {"action": "pass", "reason": "no external links"},
        }

    # Step 2: Perform verification
    verification_results: Dict[str, Any] = {}
    dns_status_by_domain: Dict[str, str] = {}

    if verification_level in {"dns", "http", "full"}:
        # DNS verification
        dns_results = []
        for domain_item in extracted.get("domains", []):
            domain = domain_item.get("host", "")
            if domain:
                result = _cached_verification("dns", domain, verification.resolve_domain, domain)
                dns_results.append(result)
                dns_status_by_domain[str(domain).strip().lower()] = str(result.get("status") or "")
        verification_results["dns"] = dns_results

    if verification_level in {"http", "full"}:
        # HTTP verification
        http_results = []
        for url_item in extracted.get("urls", []):
            url = url_item.get("normalized", "")
            host = str(url_item.get("host") or "").strip().lower()
            if (
                skip_secondary_checks_on_dns_failure
                and host
                and dns_status_by_domain.get(host) == "nxdomain_or_no_data"
            ):
                continue
            if url:
                result = _cached_verification("http", url, verification.check_http_domain, url)
                http_results.append(result)
        verification_results["http"] = http_results

    if verification_level == "full":
        # TLS and WHOIS verification
        tls_results = []
        whois_results = []
        for domain_item in extracted.get("domains", []):
            domain = domain_item.get("host", "")
            if domain:
                normalized_domain = str(domain).strip().lower()
                if (
                    skip_secondary_checks_on_dns_failure
                    and dns_status_by_domain.get(normalized_domain) == "nxdomain_or_no_data"
                ):
                    continue
                if enable_tls_verification:
                    tls_results.append(_cached_verification("tls", domain, verification.check_tls, domain))
                if enable_whois_verification:
                    whois_results.append(_cached_verification("whois", domain, verification.check_whois, domain))
        verification_results["tls"] = tls_results
        verification_results["whois"] = whois_results

    # GitHub API verification (always enabled)
    github_results = []
    for repo_item in extracted.get("github_repos", []):
        repo_key = f"{repo_item.get('owner', '')}/{repo_item.get('repo', '')}"
        result = _cached_verification(
            "github",
            repo_key,
            verification.check_github_repo,
            repo_item,
            token=github_token,
        )
        github_results.append(result)
    verification_results["github"] = github_results

    # Step 3: Query KB
    if kb_instance is None:
        kb_instance = kb.get_kb()

    rag_results: Dict[str, Any] = {}
    domain_evidence = {}
    for domain_item in extracted.get("domains", []):
        domain = domain_item.get("host", "")
        if domain:
            evidence = kb_instance.query_domain_evidence(domain)
            domain_evidence[domain] = evidence
    rag_results["domain_evidence"] = domain_evidence

    # Check blacklist
    blacklist_hosts = []
    for domain_item in extracted.get("domains", []):
        domain = domain_item.get("host", "")
        if domain and kb_instance.is_blacklisted_domain(domain):
            blacklist_hosts.append(domain_item)
    rag_results["blacklist_hosts"] = blacklist_hosts

    # Step 4: Check for hallucinations
    detection_result = checkers.check_domain_hallucination(
        extracted_items=extracted,
        verification_result=verification_results,
        rag_result=rag_results,
    )

    # Step 5: Calculate risk score
    risk_score = scoring.calculate_risk_score(detection_result)

    # Step 6: Optional semantic check
    if enable_semantic_check and user_query:
        semantic_result = semantic.check_semantic_relevance(
            user_query=user_query,
            extracted_urls=extracted.get("urls", []),
            extracted_domains=extracted.get("domains", []),
        )
        if semantic_result.get("has_irrelevant_domains"):
            for domain_item in semantic_result.get("irrelevant_domains", []):
                detection_result["issues"].append({
                    "type": "semantic_mismatch",
                    "target_type": "domain",
                    "target": domain_item.get("domain", ""),
                    "severity": "low",
                    "confidence": "low",
                    "evidence_source": "semantic",
                    "evidence": domain_item,
                    "message": "Domain not semantically related to query",
                })
            detection_result = {
                "has_issues": bool(detection_result["issues"]),
                "issues": detection_result["issues"],
                "issue_summary": checkers._summarize_issues(detection_result["issues"]),
            }

    # Step 7: Optional advanced verification
    if enable_advanced_verification:
        adv_result = semantic.check_advanced_verification(
            extracted_urls=extracted.get("urls", []),
            extracted_github_repos=extracted.get("github_repos", []),
            verification_results=verification_results,
        )
        if adv_result.get("has_issues"):
            for issue_item in adv_result.get("issues", []):
                detection_result["issues"].append({
                    "type": issue_item.get("type", "advanced_verification_failed"),
                    "target_type": "url" if "url" in issue_item else "github_repo",
                    "target": issue_item.get("url", "") or issue_item.get("target", ""),
                    "severity": issue_item.get("severity", "low"),
                    "confidence": "low",
                    "evidence_source": "advanced_verification",
                    "evidence": issue_item,
                    "message": issue_item.get("message", "Advanced verification failed"),
                })
            detection_result = {
                "has_issues": bool(detection_result["issues"]),
                "issues": detection_result["issues"],
                "issue_summary": checkers._summarize_issues(detection_result["issues"]),
            }

    # Recalculate score if issues changed
    risk_score = scoring.calculate_risk_score(detection_result)

    # Step 8: Recalibrate score
    recalibrated = scoring.recalibrate_score(
        risk_score=risk_score,
        verification_results=verification_results,
        rag_results=rag_results,
    )

    # Step 9: Make decision
    enforcement_decision = decision.make_decision(
        risk_score=risk_score,
        recalibrated_score=recalibrated,
        verification_level=verification_level,
    )

    # Step 10: Apply enforcement
    enforced = decision.apply_decision(enforcement_decision, answer=answer)

    return {
        "status": "analyzed",
        "extraction": extracted,
        "detection": detection_result,
        "risk_score": risk_score,
        "recalibrated_score": recalibrated,
        "decision": enforcement_decision,
        "enforced_answer": enforced,
        "verification_results": verification_results,
        "kb_results": rag_results,
    }


LEVEL_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}


@action(output_mapping=lambda value: value)
async def self_check_domain_hallucination(
    llm: Any = None,
    context: Optional[dict] = None,
    verification_level: str = "dns",
    enable_semantic_check: bool = False,
    enable_advanced_verification: bool = False,
    skip_secondary_checks_on_dns_failure: bool = False,
    enable_tls_verification: bool = True,
    enable_whois_verification: bool = True,
    enable_expert_review: bool = False,
    expert_review_min_level: str = "L2",
    github_token: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:
    """NeMo output-rail action for domain hallucination detection."""
    context = context or {}
    answer = (
        context.get("bot_message")
        or context.get("assistant_output")
        or context.get("last_bot_message")
        or ""
    )
    user_query = (
        context.get("user_message")
        or context.get("last_user_message")
        or context.get("user_input")
        or ""
    )

    result = await analyze_answer(
        answer=str(answer),
        user_query=str(user_query),
        verification_level=verification_level,
        enable_semantic_check=enable_semantic_check,
        enable_advanced_verification=enable_advanced_verification,
        skip_secondary_checks_on_dns_failure=skip_secondary_checks_on_dns_failure,
        enable_tls_verification=enable_tls_verification,
        enable_whois_verification=enable_whois_verification,
        github_token=github_token,
    )

    risk_score = result.get("risk_score", {}) if isinstance(result, dict) else {}
    level = str(risk_score.get("level") or result.get("decision", {}).get("level") or "L0")

    if enable_expert_review and LEVEL_ORDER.get(level, 0) >= LEVEL_ORDER.get(expert_review_min_level, 2):
        review = await expert_review.review_with_nemo_llm(
            llm=llm,
            question=str(user_query),
            model_answer=str(answer),
            extracted=result.get("extraction", {}),
            verification_result=result.get("verification_results", {}),
            rag_result=result.get("kb_results", {}),
            detection_result=result.get("detection", {}),
            score=float(risk_score.get("score", 0.0)),
            level=level,
            decision=result.get("decision", {}),
        )
        updated_decision = expert_review.apply_expert_decision(result.get("decision", {}), review)
        result["expert_review"] = review
        result["decision"] = updated_decision
        enforced = decision.apply_decision(updated_decision, answer=str(answer))
        corrected = review.get("corrected_answer")
        if corrected and updated_decision.get("action") in {"refine", "warn"}:
            enforced["modified_answer"] = corrected
            enforced["enforced"] = True
        result["enforced_answer"] = enforced

    return result

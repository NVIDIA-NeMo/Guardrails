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

"""Utility functions for domain hallucination detection."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from . import extractors, verification

logger = logging.getLogger(__name__)


def verify_entities_in_answer(
    answer: str,
    verification_level: str = "dns",
    github_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Quick utility to verify all entities in an answer.

    Args:
        answer: Text to analyze
        verification_level: Level of verification (none, dns, http, full)
        github_token: GitHub API token

    Returns:
        Dict with verification results
    """
    extracted = extractors.extract_all(answer)

    results: Dict[str, Any] = {
        "extracted": extracted,
        "verification": {},
    }

    if verification_level == "none":
        return results

    # DNS verification
    if verification_level in {"dns", "http", "full"}:
        dns_results = []
        for domain_item in extracted.get("domains", []):
            domain = domain_item.get("host", "")
            if domain:
                result = verification.resolve_domain(domain)
                dns_results.append(result)
        results["verification"]["dns"] = dns_results

    # HTTP verification
    if verification_level in {"http", "full"}:
        http_results = []
        for url_item in extracted.get("urls", []):
            url = url_item.get("normalized", "")
            if url:
                result = verification.check_http_domain(url)
                http_results.append(result)
        results["verification"]["http"] = http_results

    # GitHub verification
    github_results = []
    for repo_item in extracted.get("github_repos", []):
        result = verification.check_github_repo(repo_item, token=github_token)
        github_results.append(result)
    results["verification"]["github"] = github_results

    return results


def find_unverified_domains(
    answer: str,
    verification_level: str = "dns",
) -> List[Dict[str, Any]]:
    """Find domains that failed verification.

    Args:
        answer: Text to analyze
        verification_level: Level of verification

    Returns:
        List of unverified domains
    """
    verification_result = verify_entities_in_answer(answer, verification_level)

    unverified: List[Dict[str, Any]] = []

    # Check DNS failures
    for dns_item in verification_result.get("verification", {}).get("dns", []) or []:
        if not dns_item.get("resolves"):
            unverified.append(
                {
                    "type": "domain",
                    "domain": dns_item.get("domain"),
                    "reason": dns_item.get("status", "unknown"),
                    "evidence": dns_item,
                }
            )

    # Check HTTP failures
    for http_item in verification_result.get("verification", {}).get("http", []) or []:
        if not http_item.get("reachable"):
            unverified.append(
                {
                    "type": "url",
                    "url": http_item.get("url"),
                    "reason": http_item.get("status", "unknown"),
                    "evidence": http_item,
                }
            )

    # Check GitHub failures
    for github_item in verification_result.get("verification", {}).get("github", []) or []:
        if github_item.get("exists") is False:
            unverified.append(
                {
                    "type": "github_repo",
                    "repo": github_item.get("full_name"),
                    "reason": github_item.get("status", "unknown"),
                    "evidence": github_item,
                }
            )

    return unverified


def extract_suspicious_domains(
    answer: str,
    kb_blacklist: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract domains that are suspicious or blacklisted.

    Args:
        answer: Text to analyze
        kb_blacklist: List of blacklisted domains

    Returns:
        List of suspicious domains
    """
    kb_blacklist = kb_blacklist or []
    kb_blacklist_lower = {d.lower() for d in kb_blacklist}

    extracted = extractors.extract_all(answer)
    suspicious: List[Dict[str, Any]] = []

    for domain_item in extracted.get("domains", []):
        domain = str(domain_item.get("host", "")).lower()
        if not domain:
            continue

        if domain in kb_blacklist_lower:
            suspicious.append(
                {
                    "type": "blacklisted",
                    "domain": domain,
                    "source": "kb",
                    "evidence": domain_item,
                }
            )

    return suspicious


def count_entities_by_type(answer: str) -> Dict[str, int]:
    """Count entities by type in answer.

    Args:
        answer: Text to analyze

    Returns:
        Dict with counts
    """
    extracted = extractors.extract_all(answer)

    return {
        "urls": len(extracted.get("urls", [])),
        "domains": len(extracted.get("domains", [])),
        "github_repos": len(extracted.get("github_repos", [])),
        "total_entities": (
            len(extracted.get("urls", [])) + len(extracted.get("domains", [])) + len(extracted.get("github_repos", []))
        ),
    }


def normalize_urls_in_text(answer: str) -> Dict[str, Any]:
    """Normalize all URLs in text.

    Args:
        answer: Text to process

    Returns:
        Dict with original and normalized URLs
    """
    extracted = extractors.extract_all(answer)

    urls_map: Dict[str, str] = {}
    for url_item in extracted.get("urls", []):
        raw = url_item.get("raw", "")
        normalized = url_item.get("normalized", "")
        if raw and normalized:
            urls_map[raw] = normalized

    normalized_answer = answer
    for raw, normalized in urls_map.items():
        normalized_answer = normalized_answer.replace(raw, normalized)

    return {
        "original_answer": answer,
        "normalized_answer": normalized_answer,
        "url_mappings": urls_map,
        "urls_normalized": len(urls_map),
    }


def get_domain_stats(answer: str) -> Dict[str, Any]:
    """Get domain-related statistics.

    Args:
        answer: Text to analyze

    Returns:
        Statistics dict
    """
    extracted = extractors.extract_all(answer)

    domains = extracted.get("domains", [])
    urls = extracted.get("urls", [])

    # Count unique registered domains
    unique_registered_domains = set()
    for domain_item in domains:
        registered = domain_item.get("registered_domain", "")
        if registered:
            unique_registered_domains.add(registered)

    # Count unique hosts from URLs
    url_hosts = {url_item.get("host", "") for url_item in urls if url_item.get("host")}

    # Get all TLDs
    tlds = set()
    for domain_item in domains:
        suffix = domain_item.get("suffix", "")
        if suffix:
            tlds.add(suffix)

    return {
        "total_domains": len(domains),
        "total_urls": len(urls),
        "total_github_repos": len(extracted.get("github_repos", [])),
        "unique_registered_domains": len(unique_registered_domains),
        "unique_hosts": len(url_hosts),
        "unique_tlds": len(tlds),
        "tlds": sorted(tlds),
        "registered_domains": sorted(unique_registered_domains),
        "has_github_repos": len(extracted.get("github_repos", [])) > 0,
    }


def log_analysis_result(result: Dict[str, Any], level: str = "INFO") -> None:
    """Log analysis result in human-readable format.

    Args:
        result: Analysis result from analyze_answer()
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    log_func = getattr(logger, level.lower(), logger.info)

    status = result.get("status", "unknown")
    log_func(f"Analysis Status: {status}")

    if result.get("status") == "analyzed":
        detection = result.get("detection", {})
        has_issues = detection.get("has_issues", False)
        log_func(f"  Issues Detected: {has_issues}")

        if has_issues:
            summary = detection.get("issue_summary", {})
            log_func(f"  Issue Count: {summary.get('total', 0)}")
            log_func(f"  Highest Severity: {summary.get('highest_severity', 'none')}")

        risk_score = result.get("risk_score", {})
        log_func(f"  Risk Score: {risk_score.get('score', 0.0)}")
        log_func(f"  Risk Level: {risk_score.get('level', 'L0')}")

        decision = result.get("decision", {})
        log_func(f"  Decision: {decision.get('action', 'unknown')}")

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

"""Domain hallucination checkers aggregating detection issues."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


def check_domain_hallucination(
    extracted_items: Dict[str, Any] | None = None,
    verification_result: Dict[str, Any] | None = None,
    rag_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Aggregate runtime evidence into normalized issues."""
    extracted_items = extracted_items or {}
    verification_result = verification_result or {}
    rag_result = rag_result or {}

    issues: List[Dict[str, Any]] = []

    # Check for DNS failures
    dns_result = _check_dns_failures(extracted_items, verification_result)
    issues.extend(dns_result)

    # Check for TLS certificate failures
    tls_result = _check_tls_failures(extracted_items, verification_result)
    issues.extend(tls_result)

    # Check for GitHub issues
    github_result = _check_github_repos(extracted_items, verification_result, rag_result)
    issues.extend(github_result)

    # Check for phishing-like domains
    phishing_result = _check_phishing_domains(extracted_items, verification_result, rag_result)
    issues.extend(phishing_result)

    # Check for local KB evidence
    evidence_result = _check_kb_evidence(extracted_items, rag_result)
    issues.extend(evidence_result)

    deduped_issues = _deduplicate_issues(issues)

    return {
        "has_issues": bool(deduped_issues),
        "issues": deduped_issues,
        "issue_summary": _summarize_issues(deduped_issues),
    }


def _check_dns_failures(
    extracted_items: Dict[str, Any],
    verification_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Check for DNS resolution failures."""
    issues: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    dns_map = {}
    for item in verification_result.get("dns", []) or []:
        if isinstance(item, dict):
            domain = str(item.get("domain") or "").strip().lower()
            if domain:
                dns_map[domain] = item

    for domain_item in extracted_items.get("domains", []) or []:
        if not isinstance(domain_item, dict):
            continue
        domain = str(domain_item.get("host") or "").strip().lower()
        if not domain:
            continue

        dns = dns_map.get(domain, {})
        dns_status = str(dns.get("status", "unchecked")).strip().lower()

        if dns_status == "nxdomain_or_no_data":
            key = ("non_existent_domain", domain)
            if key not in seen:
                seen.add(key)
                issues.append(
                    {
                        "type": "non_existent_domain",
                        "target_type": "domain",
                        "target": domain,
                        "severity": "high",
                        "confidence": "high",
                        "evidence_source": "dns",
                        "evidence": dns,
                        "message": "Domain cannot be resolved; likely fabricated or stale.",
                    }
                )
        elif dns_status == "no_data":
            key = ("delegated_no_address_record", domain)
            if key not in seen:
                seen.add(key)
                issues.append(
                    {
                        "type": "delegated_no_address_record",
                        "target_type": "domain",
                        "target": domain,
                        "severity": "medium",
                        "confidence": "medium",
                        "evidence_source": "dns",
                        "evidence": dns,
                        "message": "Domain exists but has no address record.",
                    }
                )

    return issues


def _check_tls_failures(
    extracted_items: Dict[str, Any],
    verification_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Check for TLS certificate verification failures."""
    issues: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    tls_map = {}
    for item in verification_result.get("tls", []) or []:
        if isinstance(item, dict):
            domain = str(item.get("domain") or "").strip().lower()
            if domain:
                tls_map[domain] = item

    status_issue_map = {
        "tls_expired": {
            "type": "tls_certificate_expired",
            "severity": "high",
            "confidence": "medium",
            "message": "TLS certificate is expired.",
        },
        "tls_hostname_mismatch": {
            "type": "tls_hostname_mismatch",
            "severity": "high",
            "confidence": "medium",
            "message": "TLS certificate hostname does not match the domain.",
        },
        "tls_untrusted_chain": {
            "type": "tls_untrusted_chain",
            "severity": "medium",
            "confidence": "medium",
            "message": "TLS certificate chain is not trusted.",
        },
        "tls_verification_failed": {
            "type": "tls_verification_failed",
            "severity": "medium",
            "confidence": "medium",
            "message": "TLS certificate verification failed.",
        },
        "tls_expiring_soon": {
            "type": "tls_certificate_expiring_soon",
            "severity": "low",
            "confidence": "low",
            "message": "TLS certificate expires soon.",
        },
    }

    for domain_item in extracted_items.get("domains", []) or []:
        if not isinstance(domain_item, dict):
            continue
        domain = str(domain_item.get("host") or "").strip().lower()
        if not domain:
            continue

        tls = tls_map.get(domain, {})
        status = str(tls.get("status", "unchecked")).strip().lower()
        issue_template = status_issue_map.get(status)
        if not issue_template:
            continue

        issue_type = issue_template["type"]
        key = (issue_type, domain)
        if key in seen:
            continue

        seen.add(key)
        issues.append(
            {
                "type": issue_type,
                "target_type": "domain",
                "target": domain,
                "severity": issue_template["severity"],
                "confidence": issue_template["confidence"],
                "evidence_source": "tls",
                "evidence": tls,
                "message": issue_template["message"],
            }
        )

    return issues


def _check_github_repos(
    extracted_items: Dict[str, Any],
    verification_result: Dict[str, Any],
    rag_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Check for GitHub repository issues."""
    issues: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    github_map = {}
    for item in verification_result.get("github", []) or []:
        if isinstance(item, dict):
            owner = str(item.get("owner") or "").strip().lower()
            repo = str(item.get("repo") or "").strip().lower()
            if owner and repo:
                key = f"{owner}/{repo}"
                github_map[key] = item

    for repo_item in extracted_items.get("github_repos", []) or []:
        if not isinstance(repo_item, dict):
            continue
        owner = str(repo_item.get("owner") or "").strip()
        repo = str(repo_item.get("repo") or "").strip()
        if not owner or not repo:
            continue

        key = f"{owner.lower()}/{repo.lower()}"
        github = github_map.get(key, {})
        status = str(github.get("status", "unchecked")).strip().lower()

        if status == "repo_not_found":
            issue_key = ("fake_github_repo", key)
            if issue_key not in seen:
                seen.add(issue_key)
                issues.append(
                    {
                        "type": "fake_github_repo",
                        "target_type": "github_repo",
                        "target": key,
                        "severity": "high",
                        "confidence": "high",
                        "evidence_source": "github",
                        "evidence": github,
                        "message": "GitHub repository does not exist; likely fabricated.",
                    }
                )

    return issues


def _check_phishing_domains(
    extracted_items: Dict[str, Any],
    verification_result: Dict[str, Any],
    rag_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Check for phishing-like domain characteristics."""
    issues: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    rag_result = rag_result or {}
    blacklist_hosts = set()
    if isinstance(rag_result.get("blacklist_hosts"), list):
        for item in rag_result["blacklist_hosts"]:
            if isinstance(item, dict):
                host = str(item.get("domain") or item.get("host") or "").strip().lower()
            else:
                host = str(item).strip().lower()
            if host:
                blacklist_hosts.add(host)

    whois_map = {}
    for item in verification_result.get("whois", []) or []:
        if isinstance(item, dict):
            domain = str(item.get("domain") or "").strip().lower()
            if domain:
                whois_map[domain] = item

    for domain_item in extracted_items.get("domains", []) or []:
        if not isinstance(domain_item, dict):
            continue
        domain = str(domain_item.get("host") or "").strip().lower()
        if not domain:
            continue

        whois = whois_map.get(domain, {})
        is_recent = whois.get("is_recent_domain") is True

        if domain in blacklist_hosts:
            key = ("blacklisted_domain", domain)
            if key not in seen:
                seen.add(key)
                issues.append(
                    {
                        "type": "blacklisted_domain",
                        "target_type": "domain",
                        "target": domain,
                        "severity": "critical",
                        "confidence": "high",
                        "evidence_source": "blacklist",
                        "evidence": {"domain": domain},
                        "message": "Domain is in blacklist.",
                    }
                )

        if is_recent:
            key = ("recent_domain", domain)
            if key not in seen:
                seen.add(key)
                issues.append(
                    {
                        "type": "recent_domain",
                        "target_type": "domain",
                        "target": domain,
                        "severity": "low",
                        "confidence": "low",
                        "evidence_source": "whois",
                        "evidence": whois,
                        "message": "Recently registered domain.",
                        "signals": ["recent_domain"],
                    }
                )

    return issues


def _check_kb_evidence(
    extracted_items: Dict[str, Any],
    rag_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Check for missing local KB evidence."""
    issues: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    rag_result = rag_result or {}
    domain_evidence = rag_result.get("domain_evidence", {}) or {}

    for domain_item in extracted_items.get("domains", []) or []:
        if not isinstance(domain_item, dict):
            continue
        domain = str(domain_item.get("host") or "").strip().lower()
        if not domain:
            continue

        evidences = domain_evidence.get(domain, []) or []
        has_evidence = bool(isinstance(evidences, list) and evidences)

        if not has_evidence:
            key = ("no_local_kb_evidence", domain)
            if key not in seen:
                seen.add(key)
                issues.append(
                    {
                        "type": "no_local_kb_evidence",
                        "target_type": "domain",
                        "target": domain,
                        "severity": "low",
                        "confidence": "low",
                        "evidence_source": "rag",
                        "evidence": [],
                        "message": "No local KB evidence for this domain.",
                    }
                )

    return issues


def _deduplicate_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate issues by type, target, and evidence source."""
    merged: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    ordered_keys: List[tuple[str, str, str]] = []

    for issue in issues:
        issue_type = str(issue.get("type") or "unknown")
        target = str(issue.get("target") or "")
        evidence_source = str(issue.get("evidence_source") or "")
        key = (issue_type, target, evidence_source)

        if key not in merged:
            merged[key] = dict(issue)
            ordered_keys.append(key)

    return [merged[key] for key in ordered_keys]


def _summarize_issues(issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize issues by type and severity."""
    by_type = Counter(issue.get("type", "unknown") for issue in issues)
    by_severity = Counter(issue.get("severity", "unknown") for issue in issues)

    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
    highest_severity = "none"
    for severity in by_severity:
        if severity_order.get(severity, 0) > severity_order.get(highest_severity, 0):
            highest_severity = severity

    return {
        "total": len(issues),
        "by_type": dict(by_type),
        "by_severity": dict(by_severity),
        "highest_severity": highest_severity,
    }

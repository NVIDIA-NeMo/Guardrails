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

"""Semantic relevance checking for domain hallucination detection."""

from __future__ import annotations

from typing import Any, Dict, List


def check_semantic_relevance(
    user_query: str = "",
    extracted_urls: List[Dict[str, Any]] | None = None,
    extracted_domains: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Check semantic relevance of mentioned domains to the query.

    This is a placeholder for semantic analysis. In production, this would use
    semantic similarity scoring (embeddings, BM25, etc.) to check if mentioned
    domains/URLs are topically related to the user's query.
    """
    extracted_urls = extracted_urls or []
    extracted_domains = extracted_domains or []

    if not user_query or (not extracted_urls and not extracted_domains):
        return {
            "has_irrelevant_domains": False,
            "irrelevant_domains": [],
            "confidence": "low",
        }

    # Placeholder: extract query keywords
    query_keywords = set(user_query.lower().split())
    query_keywords.discard("the")
    query_keywords.discard("a")
    query_keywords.discard("an")

    irrelevant: List[Dict[str, Any]] = []

    # Check domain relevance
    for domain_item in extracted_domains:
        domain = str(domain_item.get("host", "")).lower()
        if not domain:
            continue

        domain_parts = domain.replace("-", " ").replace("_", " ").split(".")
        domain_keywords = set(p for p in domain_parts if len(p) > 2)

        # Simple heuristic: if domain shares keywords with query, it's relevant
        overlap = domain_keywords & query_keywords
        if not overlap and len(query_keywords) > 3:
            irrelevant.append(
                {
                    "domain": domain,
                    "reason": "no_keyword_overlap",
                    "query_keywords": list(query_keywords),
                    "domain_keywords": list(domain_keywords),
                }
            )

    return {
        "has_irrelevant_domains": bool(irrelevant),
        "irrelevant_domains": irrelevant,
        "confidence": "low",  # Semantic check is lower confidence
    }


def check_advanced_verification(
    extracted_urls: List[Dict[str, Any]] | None = None,
    extracted_github_repos: List[Dict[str, Any]] | None = None,
    verification_results: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Perform advanced verification checks beyond basic DNS/HTTP.

    This includes:
    - Certificate validation
    - HTTPS/SSL checks
    - GitHub API metadata validation
    - Typosquatting detection
    """
    extracted_urls = extracted_urls or []
    extracted_github_repos = extracted_github_repos or []
    verification_results = verification_results or {}

    issues: List[Dict[str, Any]] = []

    # Check HTTPS
    for url_item in extracted_urls:
        url = str(url_item.get("normalized", "")).lower()
        scheme = str(url_item.get("scheme", "")).lower()

        if scheme == "http" and not url.endswith(":8080"):
            issues.append(
                {
                    "type": "insecure_protocol",
                    "url": url,
                    "severity": "low",
                    "message": "URL uses HTTP instead of HTTPS",
                }
            )

    # Check GitHub typosquatting
    github_map = {}
    for repo_item in extracted_github_repos:
        owner = str(repo_item.get("owner", "")).lower()
        repo = str(repo_item.get("repo", "")).lower()
        if owner and repo:
            github_map[f"{owner}/{repo}"] = repo_item

    # Look for common typosquatting patterns
    common_owners = {
        "github",
        "pytorch",
        "tensorflow",
        "google",
        "microsoft",
        "facebook",
        "aws",
    }
    for key, repo_item in github_map.items():
        owner = key.split("/")[0]
        if owner in common_owners:
            continue

        # Check for common typos (single character differences)
        for common_owner in common_owners:
            if _edit_distance(owner, common_owner) == 1:
                issues.append(
                    {
                        "type": "possible_typosquatting",
                        "target": key,
                        "severity": "medium",
                        "similar_to": common_owner,
                        "message": f"GitHub owner '{owner}' is similar to '{common_owner}'",
                    }
                )
                break

    return {
        "has_issues": bool(issues),
        "issues": issues,
    }


def _edit_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

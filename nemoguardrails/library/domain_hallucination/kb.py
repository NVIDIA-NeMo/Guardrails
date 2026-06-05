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

"""Knowledge base management for trusted domains and GitHub repos."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class KnowledgeBase:
    """In-memory knowledge base for domain hallucination detection."""

    def __init__(self):
        self.trusted_domains: Dict[str, Dict[str, Any]] = {}
        self.trusted_github_repos: Dict[str, Dict[str, Any]] = {}
        self.blacklisted_domains: Dict[str, Dict[str, Any]] = {}
        self.seed_kb: Dict[str, Any] = {}
        self.external_kb_root: Optional[str] = None

    def add_trusted_domain(
        self,
        domain: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a trusted domain."""
        domain = (domain or "").strip().lower()
        if not domain:
            return
        self.trusted_domains[domain] = metadata or {"domain": domain}

    def add_trusted_github_repo(
        self,
        owner: str,
        repo: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a trusted GitHub repository."""
        owner = (owner or "").strip()
        repo = (repo or "").strip()
        if not owner or not repo:
            return
        key = f"{owner.lower()}/{repo.lower()}"
        self.trusted_github_repos[key] = metadata or {"owner": owner, "repo": repo}

    def add_blacklisted_domain(
        self,
        domain: str,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a blacklisted domain."""
        domain = (domain or "").strip().lower()
        if not domain:
            return
        entry = metadata or {"domain": domain}
        entry["reason"] = reason
        self.blacklisted_domains[domain] = entry

    def is_trusted_domain(self, domain: str) -> bool:
        """Check if domain is trusted."""
        domain = (domain or "").strip().lower()
        return domain in self.trusted_domains

    def is_trusted_github_repo(self, owner: str, repo: str) -> bool:
        """Check if GitHub repo is trusted."""
        owner = (owner or "").strip()
        repo = (repo or "").strip()
        if not owner or not repo:
            return False
        key = f"{owner.lower()}/{repo.lower()}"
        return key in self.trusted_github_repos

    def is_blacklisted_domain(self, domain: str) -> bool:
        """Check if domain is blacklisted."""
        domain = (domain or "").strip().lower()
        return domain in self.blacklisted_domains

    def get_blacklist_reason(self, domain: str) -> Optional[str]:
        """Get blacklist reason for a domain."""
        domain = (domain or "").strip().lower()
        entry = self.blacklisted_domains.get(domain, {})
        return entry.get("reason")

    def load_seed_kb(self, kb_data: Dict[str, Any]) -> None:
        """Load seed KB data."""
        self.seed_kb = kb_data or {}

        # Load trusted domains
        if "trusted_domains" in kb_data:
            for domain_item in kb_data["trusted_domains"]:
                if isinstance(domain_item, str):
                    self.add_trusted_domain(domain_item)
                elif isinstance(domain_item, dict):
                    domain = domain_item.get("domain", "")
                    self.add_trusted_domain(domain, metadata=domain_item)

        # Load trusted GitHub repos
        if "trusted_github_repos" in kb_data:
            for repo_item in kb_data["trusted_github_repos"]:
                if isinstance(repo_item, str):
                    parts = repo_item.split("/")
                    if len(parts) == 2:
                        self.add_trusted_github_repo(parts[0], parts[1])
                elif isinstance(repo_item, dict):
                    owner = repo_item.get("owner", "")
                    repo = repo_item.get("repo", "")
                    if owner and repo:
                        self.add_trusted_github_repo(owner, repo, metadata=repo_item)

        # Load blacklisted domains
        if "blacklisted_domains" in kb_data:
            for domain_item in kb_data["blacklisted_domains"]:
                if isinstance(domain_item, str):
                    self.add_blacklisted_domain(domain_item)
                elif isinstance(domain_item, dict):
                    domain = domain_item.get("domain", "")
                    reason = domain_item.get("reason", "")
                    self.add_blacklisted_domain(domain, reason=reason, metadata=domain_item)

    def query_domain_evidence(self, domain: str) -> List[Dict[str, Any]]:
        """Query evidence for a domain from KB."""
        domain = (domain or "").strip().lower()
        if not domain:
            return []

        evidence: List[Dict[str, Any]] = []

        # Check if domain is trusted
        if domain in self.trusted_domains:
            evidence.append(
                {
                    "type": "trusted_domain",
                    "domain": domain,
                    "metadata": self.trusted_domains[domain],
                }
            )

        # Check if domain is blacklisted
        if domain in self.blacklisted_domains:
            evidence.append(
                {
                    "type": "blacklisted_domain",
                    "domain": domain,
                    "metadata": self.blacklisted_domains[domain],
                }
            )

        # Query external KB if configured
        if self.external_kb_root and os.path.isdir(self.external_kb_root):
            evidence.extend(self._query_external_kb(domain))

        return evidence

    def _query_external_kb(self, domain: str) -> List[Dict[str, Any]]:
        """Query external KB root directory."""
        domain = (domain or "").strip().lower()
        if not domain or not self.external_kb_root:
            return []
        if not re.fullmatch(r"[a-z0-9._-]+", domain):
            return []

        kb_root = Path(self.external_kb_root).resolve()
        domains_root = (kb_root / "domains").resolve()
        evidence: List[Dict[str, Any]] = []

        # Look for domain-specific files
        domain_file = (domains_root / f"{domain}.json").resolve()
        if not domain_file.is_relative_to(kb_root):
            return []
        if domain_file.is_file():
            try:
                with open(domain_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    evidence.append(
                        {
                            "type": "external_kb",
                            "source": str(domain_file),
                            "domain": domain,
                            "data": data,
                        }
                    )
            except Exception:
                pass

        # Look for wildcard files
        if domains_root.is_dir():
            for f in domains_root.glob(f"*.{domain}.json"):
                resolved = f.resolve()
                if not resolved.is_relative_to(kb_root):
                    continue
                try:
                    with open(resolved, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        evidence.append(
                            {
                                "type": "external_kb",
                                "source": str(resolved),
                                "domain": domain,
                                "data": data,
                            }
                        )
                except Exception:
                    pass

        return evidence

    def get_stats(self) -> Dict[str, Any]:
        """Get KB statistics."""
        return {
            "trusted_domains_count": len(self.trusted_domains),
            "trusted_github_repos_count": len(self.trusted_github_repos),
            "blacklisted_domains_count": len(self.blacklisted_domains),
            "external_kb_root": self.external_kb_root,
        }


# Global KB instance
_kb_instance: Optional[KnowledgeBase] = None


def get_kb() -> KnowledgeBase:
    """Get or create global KB instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance


def initialize_kb(
    seed_kb_path: Optional[str] = None,
    external_kb_root: Optional[str] = None,
) -> KnowledgeBase:
    """Initialize KB with seed data and external root."""
    kb = get_kb()

    if seed_kb_path and os.path.isfile(seed_kb_path):
        try:
            with open(seed_kb_path, "r", encoding="utf-8") as f:
                seed_data = json.load(f)
                kb.load_seed_kb(seed_data)
        except Exception:
            pass

    if external_kb_root:
        kb.external_kb_root = external_kb_root

    return kb

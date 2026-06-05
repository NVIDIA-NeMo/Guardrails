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

"""Tests for domain hallucination checkers."""

import unittest

from nemoguardrails.library.domain_hallucination import checkers


class TestCheckers(unittest.TestCase):
    """Test issue aggregation logic."""

    def test_aggregate_mixed_issues(self):
        """Test aggregating DNS, GitHub, blacklist, and KB signals."""
        extracted = {
            "domains": [
                {"host": "fake.example"},
                {"host": "phishing.com"},
                {"host": "known.example"},
            ],
            "github_repos": [{"owner": "fake", "repo": "repo"}],
        }
        verification = {
            "dns": [{"domain": "fake.example", "status": "nxdomain_or_no_data"}],
            "github": [{"owner": "fake", "repo": "repo", "status": "repo_not_found"}],
            "whois": [{"domain": "phishing.com", "is_recent_domain": True}],
            "tls": [{"domain": "known.example", "status": "tls_expiring_soon"}],
        }
        rag = {
            "blacklist_hosts": ["phishing.com"],
            "domain_evidence": {"known.example": [{"type": "trusted_domain"}]},
        }

        result = checkers.check_domain_hallucination(extracted, verification, rag)

        assert result["has_issues"] is True
        issue_types = {issue["type"] for issue in result["issues"]}
        assert "non_existent_domain" in issue_types
        assert "fake_github_repo" in issue_types
        assert "blacklisted_domain" in issue_types
        assert "recent_domain" in issue_types
        assert "tls_certificate_expiring_soon" in issue_types
        assert "no_local_kb_evidence" in issue_types
        assert result["issue_summary"]["highest_severity"] == "critical"

    def test_deduplicate_repeated_issues(self):
        """Test repeated issue records are deduplicated."""
        issues = [
            {
                "type": "fake_github_repo",
                "target": "fake/repo",
                "evidence_source": "github",
            },
            {
                "type": "fake_github_repo",
                "target": "fake/repo",
                "evidence_source": "github",
            },
            {
                "type": "fake_github_repo",
                "target": "fake/repo",
                "evidence_source": "rag",
            },
        ]

        deduped = checkers._deduplicate_issues(issues)

        assert len(deduped) == 2

    def test_summarize_issues(self):
        """Test issue summary counts and severity ranking."""
        issues = [
            {"type": "blacklisted_domain", "severity": "critical"},
            {"type": "recent_domain", "severity": "low"},
            {"type": "recent_domain", "severity": "low"},
        ]

        summary = checkers._summarize_issues(issues)

        assert summary["total"] == 3
        assert summary["by_type"]["recent_domain"] == 2
        assert summary["by_severity"]["low"] == 2
        assert summary["highest_severity"] == "critical"

    def test_check_domain_hallucination_empty_extraction(self):
        """Test empty extraction produces no issues."""
        result = checkers.check_domain_hallucination(
            extracted_items={"domains": [], "urls": [], "github_repos": []},
            verification_result={},
            rag_result={},
        )
        assert result["has_issues"] is False
        assert result["issues"] == []

    def test_deduplicate_identical_issues(self):
        """Test identical issues are deduplicated."""
        issues = [
            {
                "type": "nxdomain",
                "target": "fake.com",
                "severity": "high",
                "evidence_source": "dns",
            },
            {
                "type": "nxdomain",
                "target": "fake.com",
                "severity": "high",
                "evidence_source": "dns",
            },
            {
                "type": "nxdomain",
                "target": "fake.com",
                "severity": "high",
                "evidence_source": "dns",
            },
        ]
        dedupe = checkers._deduplicate_issues(issues)
        assert len(dedupe) == 1

    def test_summarize_issues_multiple_types(self):
        """Test summary with multiple issue types."""
        issues = [
            {"type": "nxdomain", "target": "a.com", "severity": "high"},
            {"type": "fake_github_repo", "target": "x/y", "severity": "high"},
            {"type": "tls_certificate_expired", "target": "b.com", "severity": "high"},
        ]
        summary = checkers._summarize_issues(issues)
        assert summary["total"] == 3
        assert summary["by_type"]["fake_github_repo"] == 1

    def test_check_domain_hallucination_with_github_issues(self):
        """Test hallucination check with GitHub repo issues."""
        result = checkers.check_domain_hallucination(
            extracted_items={
                "domains": [],
                "urls": [],
                "github_repos": [
                    {
                        "owner": "fake",
                        "repo": "nonexistent",
                        "url": "https://github.com/fake/nonexistent",
                    }
                ],
            },
            verification_result={
                "github": [
                    {
                        "owner": "fake",
                        "repo": "nonexistent",
                        "exists": False,
                        "status": "repo_not_found",
                    }
                ]
            },
            rag_result={},
        )
        assert result["has_issues"] is True
        assert result["issues"][0]["type"] == "fake_github_repo"

    def test_check_domain_hallucination_mixed_results(self):
        """Test with both real and hallucinated domains."""
        result = checkers.check_domain_hallucination(
            extracted_items={
                "domains": [
                    {"host": "python.org"},
                    {"host": "fakesite.xyz"},
                ],
                "urls": [],
                "github_repos": [],
            },
            verification_result={
                "dns": [
                    {"domain": "python.org", "status": "resolved"},
                    {"domain": "fakesite.xyz", "status": "nxdomain_or_no_data"},
                ]
            },
            rag_result={"domain_evidence": {"python.org": [{"type": "trusted_domain"}]}},
        )
        assert result["has_issues"] is True
        assert any(issue["target"] == "fakesite.xyz" for issue in result["issues"])


class TestCheckersEdgeCases(unittest.TestCase):
    """Edge cases: non-dict items and empty-host guards in checkers."""

    def test_non_dict_domain_item_is_skipped(self):
        """Non-dict entries in domains list are silently skipped."""
        from nemoguardrails.library.domain_hallucination import checkers

        extracted = {
            "domains": ["not-a-dict", None, 42, {"host": ""}],
            "urls": [],
            "github_repos": [],
        }
        result = checkers.check_domain_hallucination(
            extracted_items=extracted,
            verification_result={},
            rag_result={},
        )
        # No crash; the bad items are simply ignored
        assert isinstance(result, dict)
        assert "issues" in result

    def test_empty_host_domain_item_is_skipped(self):
        """Domain items with empty host are skipped without raising."""
        from nemoguardrails.library.domain_hallucination import checkers

        extracted = {
            "domains": [{"host": ""}, {"host": None}],
            "urls": [],
            "github_repos": [],
        }
        result = checkers.check_domain_hallucination(
            extracted_items=extracted,
            verification_result={},
            rag_result={},
        )
        assert result["has_issues"] is False

    def test_http_checker_skips_non_dict_and_empty_host(self):
        """HTTP checker handles non-dict URL items and items without host."""
        from nemoguardrails.library.domain_hallucination import checkers

        extracted = {
            "domains": [],
            "urls": ["not-a-dict", {"host": ""}, {"normalized": ""}],
            "github_repos": [],
        }
        result = checkers.check_domain_hallucination(
            extracted_items=extracted,
            verification_result={"http": []},
            rag_result={},
        )
        assert isinstance(result, dict)


if __name__ == "__main__":
    unittest.main()

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

"""Tests for domain hallucination actions module."""

import asyncio
import unittest
from unittest.mock import patch

from nemoguardrails.library.domain_hallucination import actions
from nemoguardrails.library.domain_hallucination import kb as kb_module


class TestVerificationCache(unittest.TestCase):
    """Test cache eviction and TTL behavior."""

    def setUp(self):
        for bucket in actions._VERIFICATION_CACHE.values():
            bucket.clear()

    def test_cache_entry_expires_after_ttl(self):
        """Test expired entries are evicted on read."""
        with patch(
            "nemoguardrails.library.domain_hallucination.actions.time.time",
            return_value=1000.0,
        ):
            actions._cache_set("dns", "example.com", {"status": "resolved"})

        with patch(
            "nemoguardrails.library.domain_hallucination.actions.time.time",
            return_value=1000.0 + actions._CACHE_TTL_SECONDS + 1,
        ):
            assert actions._cache_get("dns", "example.com") is None

        assert "example.com" not in actions._VERIFICATION_CACHE["dns"]

    def test_cache_hit_on_second_call(self):
        """Test cached verification reuses the first result."""
        calls = {"count": 0}

        def fake_check(value):
            calls["count"] += 1
            return {"status": "ok", "value": value}

        first = actions._cached_verification("dns", "example.com", fake_check, "example.com")
        second = actions._cached_verification("dns", "example.com", fake_check, "example.com")

        assert first["cache_hit"] is False
        assert second["cache_hit"] is True
        assert second["value"] == "example.com"
        assert calls["count"] == 1

    def test_cache_caps_bucket_size(self):
        """Test cache size is bounded per verification source."""
        for index in range(actions._CACHE_MAX_ITEMS_PER_SOURCE + 1):
            actions._cache_set("dns", f"example-{index}.com", {"status": "resolved"})

        bucket = actions._VERIFICATION_CACHE["dns"]
        assert len(bucket) == actions._CACHE_MAX_ITEMS_PER_SOURCE
        assert "example-0.com" not in bucket


class TestAnalyzeAnswer(unittest.IsolatedAsyncioTestCase):
    """Test verification-level behavior in analyze_answer."""

    def setUp(self):
        for bucket in actions._VERIFICATION_CACHE.values():
            bucket.clear()

    def _base_patches(self, extracted):
        class DummyKB:
            def query_domain_evidence(self, _domain):
                return []

            def is_blacklisted_domain(self, _domain):
                return False

        return (
            patch(
                "nemoguardrails.library.domain_hallucination.actions.extractors.extract_all",
                return_value=extracted,
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.kb.get_kb",
                return_value=DummyKB(),
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.checkers.check_domain_hallucination",
                return_value={"has_issues": False, "issues": [], "issue_summary": {}},
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.scoring.calculate_risk_score",
                return_value={"score": 0.0, "level": "L0"},
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.scoring.recalibrate_score",
                return_value={},
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.decision.make_decision",
                return_value={
                    "action": "pass",
                    "reason": "safe",
                    "level": "L0",
                    "score": 0.0,
                },
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.decision.apply_decision",
                return_value={
                    "action": "pass",
                    "modified_answer": "hello",
                    "enforced": False,
                },
            ),
        )

    async def test_analyze_empty_answer(self):
        """Test empty answers are skipped early."""
        result = await actions.analyze_answer(answer="")
        assert result["status"] == "skipped"
        assert result["reason"] == "empty_answer"

    async def test_analyze_rejects_invalid_verification_level(self):
        """Test misspelled verification levels fail closed."""
        with self.assertRaises(ValueError):
            await actions.analyze_answer(answer="hello", verification_level="ful")

    async def test_self_check_rejects_invalid_verification_level(self):
        """Test rail action rejects invalid verification levels."""
        with self.assertRaises(ValueError):
            await actions.self_check_domain_hallucination(
                context={"bot_message": "hello"},
                verification_level="ful",
            )

    async def test_cached_verification_async_uses_executor(self):
        """Test blocking verification is delegated to an executor."""
        real_loop = asyncio.get_running_loop()

        class DummyLoop:
            def __init__(self):
                self.executor = "unset"

            def run_in_executor(self, executor, func):
                self.executor = executor
                future = real_loop.create_future()
                future.set_result(func())
                return future

        dummy_loop = DummyLoop()

        with patch(
            "nemoguardrails.library.domain_hallucination.actions.asyncio.get_running_loop",
            return_value=dummy_loop,
        ):
            result = await actions._cached_verification_async(
                "dns",
                "example.com",
                lambda value: {"status": "ok", "value": value},
                "example.com",
            )

        assert dummy_loop.executor is None
        assert result["status"] == "ok"
        assert result["cache_hit"] is False

    async def test_analyze_no_links(self):
        """Test link-free answers fast-pass early."""
        with patch(
            "nemoguardrails.library.domain_hallucination.actions.extractors.extract_all",
            return_value={
                "urls": [],
                "domains": [],
                "github_repos": [],
                "no_links": True,
            },
        ):
            result = await actions.analyze_answer(answer="plain text")
        assert result["status"] == "fast_pass"
        assert result["reason"] == "no_links_detected"

    async def test_github_verification_runs_only_in_full_mode(self):
        """Test GitHub API checks are skipped outside full verification."""

        extracted = {
            "urls": [],
            "domains": [],
            "github_repos": [{"owner": "octo", "repo": "hello-world"}],
            "no_links": False,
        }

        class DummyKB:
            def query_domain_evidence(self, _domain):
                return {}

            def is_blacklisted_domain(self, _domain):
                return False

        with (
            patch(
                "nemoguardrails.library.domain_hallucination.actions.extractors.extract_all",
                return_value=extracted,
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.kb.get_kb",
                return_value=DummyKB(),
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.checkers.check_domain_hallucination",
                return_value={"has_issues": False, "issues": [], "issue_summary": {}},
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.scoring.calculate_risk_score",
                return_value={"score": 0.0, "level": "L0"},
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.scoring.recalibrate_score",
                return_value={},
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.decision.make_decision",
                return_value={
                    "action": "pass",
                    "reason": "safe",
                    "level": "L0",
                    "score": 0.0,
                },
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.decision.apply_decision",
                return_value={
                    "action": "pass",
                    "modified_answer": "hello",
                    "enforced": False,
                },
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.verification.check_github_repo",
                return_value={"status": "repo_exists", "exists": True},
            ) as mock_check_github,
        ):
            dns_result = await actions.analyze_answer(answer="hello", verification_level="dns")
            assert dns_result["verification_results"]["github"] == []
            mock_check_github.assert_not_called()

            full_result = await actions.analyze_answer(answer="hello", verification_level="full")
            assert full_result["verification_results"]["github"] == [
                {"status": "repo_exists", "exists": True, "cache_hit": False}
            ]
            mock_check_github.assert_called_once()

    async def test_analyze_answer_with_dns_verification(self):
        """Test DNS verification in analyze_answer."""
        extracted = {
            "urls": [{"normalized": "https://example.com", "host": "example.com"}],
            "domains": [{"host": "example.com"}],
            "github_repos": [],
            "no_links": False,
        }
        patches = self._base_patches(extracted)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patch(
                "nemoguardrails.library.domain_hallucination.actions.verification.resolve_domain",
                return_value={
                    "status": "resolved",
                    "domain": "example.com",
                    "resolves": True,
                },
            ) as mock_dns,
        ):
            result = await actions.analyze_answer("Check https://example.com", verification_level="dns")
        assert result["status"] == "analyzed"
        assert result["verification_results"]["dns"][0]["status"] == "resolved"
        mock_dns.assert_called_once_with("example.com")

    async def test_analyze_answer_http_verification(self):
        """Test HTTP verification level."""
        extracted = {
            "urls": [{"normalized": "https://example.com", "host": "example.com"}],
            "domains": [{"host": "example.com"}],
            "github_repos": [],
            "no_links": False,
        }
        patches = self._base_patches(extracted)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patch(
                "nemoguardrails.library.domain_hallucination.actions.verification.resolve_domain",
                return_value={
                    "status": "resolved",
                    "domain": "example.com",
                    "resolves": True,
                },
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.verification.check_http_domain",
                return_value={"status": "http_ok", "reachable": True},
            ) as mock_http,
        ):
            result = await actions.analyze_answer("Visit https://example.com", verification_level="http")
        assert result["verification_results"]["http"][0]["status"] == "http_ok"
        mock_http.assert_called_once_with("https://example.com")

    async def test_analyze_answer_full_verification(self):
        """Test full verification with TLS and WHOIS."""
        extracted = {
            "urls": [],
            "domains": [{"host": "example.com"}],
            "github_repos": [],
            "no_links": False,
        }
        patches = self._base_patches(extracted)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patch(
                "nemoguardrails.library.domain_hallucination.actions.verification.resolve_domain",
                return_value={
                    "status": "resolved",
                    "domain": "example.com",
                    "resolves": True,
                },
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.verification.check_tls",
                return_value={"status": "tls_ok"},
            ) as mock_tls,
            patch(
                "nemoguardrails.library.domain_hallucination.actions.verification.check_whois",
                return_value={"status": "ok"},
            ) as mock_whois,
        ):
            result = await actions.analyze_answer("Check domain", verification_level="full")
        assert result["verification_results"]["tls"][0]["status"] == "tls_ok"
        assert result["verification_results"]["whois"][0]["status"] == "ok"
        mock_tls.assert_called_once_with("example.com")
        mock_whois.assert_called_once_with("example.com")

    async def test_analyze_answer_with_semantic_check(self):
        """Test semantic check enabled."""
        extracted = {
            "urls": [{"normalized": "https://example.com", "host": "example.com"}],
            "domains": [{"host": "example.com"}],
            "github_repos": [],
            "no_links": False,
        }
        patches = self._base_patches(extracted)
        semantic_result = {
            "has_irrelevant_domains": True,
            "irrelevant_domains": [{"domain": "example.com"}],
        }
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patch(
                "nemoguardrails.library.domain_hallucination.actions.verification.resolve_domain",
                return_value={
                    "status": "resolved",
                    "domain": "example.com",
                    "resolves": True,
                },
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.semantic.check_semantic_relevance",
                return_value=semantic_result,
            ) as mock_semantic,
        ):
            result = await actions.analyze_answer(
                "PyTorch docs at https://example.com",
                user_query="What is PyTorch?",
                enable_semantic_check=True,
            )
        assert result["status"] == "analyzed"
        mock_semantic.assert_called_once()

    async def test_analyze_answer_semantic_check_no_irrelevant_domains(self):
        """Test semantic check false result leaves detection unchanged."""
        extracted = {
            "urls": [{"normalized": "https://example.com", "host": "example.com"}],
            "domains": [{"host": "example.com"}],
            "github_repos": [],
            "no_links": False,
        }
        patches = self._base_patches(extracted)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patch(
                "nemoguardrails.library.domain_hallucination.actions.verification.resolve_domain",
                return_value={
                    "status": "resolved",
                    "domain": "example.com",
                    "resolves": True,
                },
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.semantic.check_semantic_relevance",
                return_value={
                    "has_irrelevant_domains": False,
                    "irrelevant_domains": [],
                },
            ) as mock_semantic,
        ):
            result = await actions.analyze_answer(
                "Docs at https://example.com",
                user_query="Where are the docs?",
                enable_semantic_check=True,
            )
        assert result["detection"]["issues"] == []
        mock_semantic.assert_called_once()

    async def test_analyze_answer_with_advanced_verification(self):
        """Test advanced verification enabled."""
        extracted = {
            "urls": [{"normalized": "https://typo-example.com", "host": "typo-example.com"}],
            "domains": [{"host": "typo-example.com"}],
            "github_repos": [],
            "no_links": False,
        }
        patches = self._base_patches(extracted)
        adv_result = {
            "has_issues": True,
            "issues": [
                {
                    "type": "advanced_verification_failed",
                    "url": "https://typo-example.com",
                }
            ],
        }
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patch(
                "nemoguardrails.library.domain_hallucination.actions.verification.resolve_domain",
                return_value={
                    "status": "resolved",
                    "domain": "typo-example.com",
                    "resolves": True,
                },
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.semantic.check_advanced_verification",
                return_value=adv_result,
            ) as mock_advanced,
        ):
            result = await actions.analyze_answer(
                "Visit https://typo-example.com",
                enable_advanced_verification=True,
            )
        assert result["status"] == "analyzed"
        mock_advanced.assert_called_once()

    async def test_analyze_answer_advanced_verification_no_issues(self):
        """Test advanced verification false result leaves detection unchanged."""
        extracted = {
            "urls": [{"normalized": "https://example.com", "host": "example.com"}],
            "domains": [{"host": "example.com"}],
            "github_repos": [],
            "no_links": False,
        }
        patches = self._base_patches(extracted)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patch(
                "nemoguardrails.library.domain_hallucination.actions.verification.resolve_domain",
                return_value={
                    "status": "resolved",
                    "domain": "example.com",
                    "resolves": True,
                },
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.semantic.check_advanced_verification",
                return_value={"has_issues": False, "issues": []},
            ) as mock_advanced,
        ):
            result = await actions.analyze_answer(
                "Visit https://example.com",
                enable_advanced_verification=True,
            )
        assert result["detection"]["issues"] == []
        mock_advanced.assert_called_once()

    async def test_analyze_answer_skip_secondary_checks(self):
        """Test skip secondary checks on DNS failure."""
        extracted = {
            "urls": [{"normalized": "https://fake.xyz", "host": "fake.xyz"}],
            "domains": [{"host": "fake.xyz"}],
            "github_repos": [],
            "no_links": False,
        }
        patches = self._base_patches(extracted)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patch(
                "nemoguardrails.library.domain_hallucination.actions.verification.resolve_domain",
                return_value={
                    "status": "nxdomain_or_no_data",
                    "domain": "fake.xyz",
                    "resolves": False,
                },
            ),
            patch("nemoguardrails.library.domain_hallucination.actions.verification.check_http_domain") as mock_http,
        ):
            result = await actions.analyze_answer(
                "Site at https://fake.xyz",
                skip_secondary_checks_on_dns_failure=True,
                verification_level="http",
            )
        assert result["verification_results"]["http"] == []
        mock_http.assert_not_called()

    async def test_analyze_answer_custom_kb(self):
        """Test with custom KB instance."""
        custom_kb = kb_module.KnowledgeBase()
        custom_kb.add_trusted_domain("trusted.com")
        extracted = {
            "urls": [{"normalized": "https://trusted.com", "host": "trusted.com"}],
            "domains": [{"host": "trusted.com"}],
            "github_repos": [],
            "no_links": False,
        }
        patches = self._base_patches(extracted)
        with (
            patches[0],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patch(
                "nemoguardrails.library.domain_hallucination.actions.verification.resolve_domain",
                return_value={
                    "status": "resolved",
                    "domain": "trusted.com",
                    "resolves": True,
                },
            ),
        ):
            result = await actions.analyze_answer(
                "Visit https://trusted.com",
                kb_instance=custom_kb,
            )
        assert result["kb_results"]["domain_evidence"]["trusted.com"]

    async def test_analyze_answer_complete_return_structure(self):
        """Test analyzed result includes the public response structure."""
        extracted = {
            "urls": [{"normalized": "https://example.com", "host": "example.com"}],
            "domains": [{"host": "example.com"}],
            "github_repos": [],
            "no_links": False,
        }
        detection = {"has_issues": False, "issues": [], "issue_summary": {"total": 0}}
        risk_score = {"score": 0.0, "level": "L0"}
        decision = {"action": "pass", "reason": "safe", "level": "L0", "score": 0.0}
        enforced = {"action": "pass", "modified_answer": "hello", "enforced": False}
        with (
            patch(
                "nemoguardrails.library.domain_hallucination.actions.extractors.extract_all",
                return_value=extracted,
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.kb.get_kb",
                return_value=kb_module.KnowledgeBase(),
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.verification.resolve_domain",
                return_value={
                    "status": "resolved",
                    "domain": "example.com",
                    "resolves": True,
                },
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.checkers.check_domain_hallucination",
                return_value=detection,
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.scoring.calculate_risk_score",
                return_value=risk_score,
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.scoring.recalibrate_score",
                return_value={"recalibrated_score": 0.0, "recalibrated_level": "L0"},
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.decision.make_decision",
                return_value=decision,
            ),
            patch(
                "nemoguardrails.library.domain_hallucination.actions.decision.apply_decision",
                return_value=enforced,
            ),
        ):
            result = await actions.analyze_answer(answer="hello", verification_level="dns")

        assert result["status"] == "analyzed"
        assert result["extraction"] == extracted
        assert result["detection"] == detection
        assert result["risk_score"] == risk_score
        assert result["decision"] == decision
        assert result["enforced_answer"] == enforced
        assert "verification_results" in result
        assert "kb_results" in result


if __name__ == "__main__":
    unittest.main()

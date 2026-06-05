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

"""Tests for domain hallucination expert review helpers."""

import json
import unittest
from unittest.mock import patch

from nemoguardrails.library.domain_hallucination import expert_review


class TestExpertReview(unittest.TestCase):
    """Test expert review parsing and decision upgrade logic."""

    def test_disabled_review_default(self):
        """Test disabled review payload."""
        result = expert_review.build_disabled_review()
        assert result["enabled"] is False
        assert result["status"] == "disabled"
        assert result["semantic_risk"] == "unknown"

    def test_extract_json_object_clean(self):
        """Test direct JSON extraction."""
        data = expert_review._extract_json_object('{"semantic_risk":"low"}')
        assert data["semantic_risk"] == "low"

    def test_extract_json_object_markdown_wrapped(self):
        """Test JSON extraction from markdown-wrapped response."""
        wrapped = '```json\n{"semantic_risk":"medium","domain_relevance":"match"}\n```'
        data = expert_review._extract_json_object(wrapped)
        assert data["semantic_risk"] == "medium"
        assert data["domain_relevance"] == "match"

    def test_extract_json_object_empty_raises(self):
        """Test empty input raises decode error."""
        with self.assertRaises(json.JSONDecodeError):
            expert_review._extract_json_object("")

    def test_normalize_review_valid(self):
        """Test valid review normalization."""
        result = expert_review._normalize_review(
            {
                "semantic_risk": "high",
                "is_answer_misleading": True,
                "domain_relevance": "match",
                "official_candidates": [{"domain": "docs.python.org", "confidence": 1.4, "source": "kb"}],
                "corrected_answer": "Use the official docs.",
                "explanation": "Official docs match the question.",
            }
        )
        assert result["status"] == "success"
        assert result["semantic_risk"] == "high"
        assert result["official_candidates"] == [{"domain": "docs.python.org", "confidence": 1.0, "source": "kb"}]

    def test_normalize_review_invalid_risk(self):
        """Test invalid enum values fall back to unknown."""
        result = expert_review._normalize_review(
            {
                "semantic_risk": "super_high",
                "domain_relevance": "not_sure",
                "is_answer_misleading": "maybe",
                "official_candidates": ["python.org"],
            }
        )
        assert result["semantic_risk"] == "unknown"
        assert result["domain_relevance"] == "unknown"
        assert result["is_answer_misleading"] is None
        assert result["official_candidates"] == [{"domain": "python.org", "confidence": 0.5, "source": "expert_model"}]

    def test_apply_expert_upgrades_to_block(self):
        """Test critical expert review upgrades action to block."""
        updated = expert_review.apply_expert_decision(
            {"action": "warn", "reason": "rule"},
            {
                "status": "success",
                "semantic_risk": "critical",
                "domain_relevance": "unknown",
                "is_answer_misleading": True,
            },
        )
        assert updated["action"] == "block"
        assert updated["expert_policy"] == "advisory_upgrade_only"

    def test_apply_expert_does_not_downgrade_block(self):
        """Test low-risk expert review does not weaken a block."""
        updated = expert_review.apply_expert_decision(
            {"action": "block", "reason": "hard evidence"},
            {
                "status": "success",
                "semantic_risk": "low",
                "domain_relevance": "match",
                "is_answer_misleading": False,
            },
        )
        assert updated["action"] == "block"

    def test_build_expert_review_payload(self):
        """Test payload construction with extracted evidence."""
        extracted = {
            "urls": [{"normalized": "https://example.com", "host": "example.com"}],
            "domains": [{"host": "example.com"}],
            "github_repos": [{"owner": "pytorch", "repo": "pytorch"}],
        }
        payload = expert_review.build_expert_review_payload(
            question="What is PyTorch?",
            model_answer="PyTorch is...",
            extracted=extracted,
            verification_result={},
            rag_result={},
            detection_result={},
            score=50.0,
            level="L2",
            decision={"action": "warn"},
        )
        assert payload["question"] == "What is PyTorch?"
        assert payload["rule_score"] == 50.0
        assert payload["extracted_domains"] == ["example.com"]
        assert payload["github_repos"] == [{"owner": "pytorch", "repo": "pytorch", "url": ""}]

    def test_fallback_invalid_json(self):
        """Test fallback handler."""
        result = expert_review._fallback("invalid_json", "Bad JSON")
        assert result["status"] == "invalid_json"
        assert result["enabled"] is True
        assert result["explanation"] == "Bad JSON"

    def test_compact_github_repos_with_data(self):
        """Test GitHub repo compaction."""
        repos = [
            {
                "owner": "pytorch",
                "repo": "pytorch",
                "url": "https://github.com/pytorch/pytorch",
            },
            {"owner": "tensorflow", "repo": "tensorflow"},
            {"owner": "", "repo": "ignored"},
            "not-a-repo",
        ]
        compacted = expert_review._compact_github_repos(repos)
        assert len(compacted) == 2
        assert compacted[0]["owner"] == "pytorch"
        assert compacted[1]["repo"] == "tensorflow"

    def test_domain_evidence_extraction(self):
        """Test domain evidence collection from verification maps."""
        verification_result = {
            "dns": [{"domain": "example.com", "status": "resolved"}],
            "http": [{"domain": "example.com", "status": "http_ok", "status_code": 200}],
        }
        rag_result = {"domain_evidence": {"example.com": [{"type": "trusted_domain"}]}}
        evidence = expert_review._domain_evidence(["example.com"], verification_result, rag_result)
        assert len(evidence) == 1
        assert evidence[0]["domain"] == "example.com"
        assert evidence[0]["dns"]["status"] == "resolved"
        assert evidence[0]["http"]["status_code"] == 200
        assert evidence[0]["kb_evidence"] == [{"type": "trusted_domain"}]

    def test_normalize_candidates_mixed_format(self):
        """Test candidate normalization with mixed input formats."""
        result = expert_review._normalize_review(
            {
                "semantic_risk": "high",
                "official_candidates": [
                    "python.org",
                    {"domain": "docs.python.org", "confidence": 0.8},
                    {"domain": "python.org", "confidence": 0.9},
                ],
            }
        )
        assert result["official_candidates"] == [
            {"domain": "python.org", "confidence": 0.5, "source": "expert_model"},
            {"domain": "docs.python.org", "confidence": 0.8, "source": "expert_model"},
        ]

    def test_build_payload_with_empty_lists(self):
        """Test payload building with empty extraction."""
        payload = expert_review.build_expert_review_payload(
            question="Q",
            model_answer="A",
            extracted={"urls": [], "domains": [], "github_repos": []},
            verification_result={},
            rag_result={},
            detection_result={"issues": []},
            score=0.0,
            level="L0",
            decision={"action": "pass"},
        )
        assert payload["extracted_domains"] == []
        assert payload["github_repos"] == []

    def test_extract_json_with_trailing_comma(self):
        """Test strict JSON extraction rejects trailing comma safely."""
        with self.assertRaises(json.JSONDecodeError):
            expert_review._extract_json_object('{"semantic_risk": "high",}')

    def test_compact_issues_filtering(self):
        """Test issue compaction keeps normalized issue fields."""
        compacted = expert_review._compact_issues(
            [
                {
                    "type": "nxdomain",
                    "target": "fake.com",
                    "severity": "high",
                    "message": "Not found",
                },
                {
                    "type": "tls_error",
                    "target": "bad.com",
                    "severity": "high",
                    "message": "Cert error",
                },
            ]
        )
        assert len(compacted) == 2
        assert all("type" in item for item in compacted)


class TestExpertReviewAsync(unittest.IsolatedAsyncioTestCase):
    """Test async expert review entrypoint without real LLM calls."""

    async def test_review_with_nemo_llm_disabled_when_llm_missing(self):
        """Test expert review safely disables itself without an LLM."""
        result = await expert_review.review_with_nemo_llm(
            llm=None,
            question="Q",
            model_answer="A",
            extracted={},
            verification_result={},
            rag_result={},
            detection_result={},
            score=0.0,
            level="L0",
            decision={"action": "pass"},
        )
        assert result["enabled"] is False
        assert result["status"] == "disabled"

    async def test_review_with_nemo_llm_normalizes_valid_json(self):
        """Test valid mocked LLM JSON is normalized."""

        async def fake_llm_call(*_args, **_kwargs):
            return '{"semantic_risk":"high","is_answer_misleading":true,"domain_relevance":"mismatch"}'

        with patch(
            "nemoguardrails.library.domain_hallucination.expert_review.llm_call",
            fake_llm_call,
        ):
            result = await expert_review.review_with_nemo_llm(
                llm=object(),
                question="Q",
                model_answer="A",
                extracted={},
                verification_result={},
                rag_result={},
                detection_result={},
                score=42.0,
                level="L2",
                decision={"action": "warn"},
            )

        assert result["enabled"] is True
        assert result["status"] == "success"
        assert result["semantic_risk"] == "high"
        assert result["domain_relevance"] == "mismatch"

    async def test_review_with_nemo_llm_invalid_json_falls_back(self):
        """Test invalid mocked LLM JSON returns fallback."""

        async def fake_llm_call(*_args, **_kwargs):
            return "not json"

        with patch(
            "nemoguardrails.library.domain_hallucination.expert_review.llm_call",
            fake_llm_call,
        ):
            result = await expert_review.review_with_nemo_llm(
                llm=object(),
                question="Q",
                model_answer="A",
                extracted={},
                verification_result={},
                rag_result={},
                detection_result={},
                score=42.0,
                level="L2",
                decision={"action": "warn"},
            )

        assert result["enabled"] is True
        assert result["status"] == "invalid_json"


if __name__ == "__main__":
    unittest.main()

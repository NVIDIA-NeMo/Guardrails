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

"""Tests for utility helpers."""

from unittest.mock import patch

from nemoguardrails.library.domain_hallucination import utils

EXTRACTED = {
    "urls": [
        {
            "raw": "example.com/path",
            "normalized": "https://example.com/path",
            "host": "example.com",
        }
    ],
    "domains": [
        {
            "host": "example.com",
            "registered_domain": "example.com",
            "suffix": "com",
        }
    ],
    "github_repos": [{"owner": "pytorch", "repo": "pytorch"}],
}


@patch("nemoguardrails.library.domain_hallucination.utils.extractors.extract_all")
def test_verify_entities_in_answer_none_skips_verification(mock_extract_all):
    mock_extract_all.return_value = EXTRACTED

    result = utils.verify_entities_in_answer("See example.com", "none")

    assert result["extracted"] == EXTRACTED
    assert result["verification"] == {}


@patch("nemoguardrails.library.domain_hallucination.utils.verification.check_github_repo")
@patch("nemoguardrails.library.domain_hallucination.utils.verification.resolve_domain")
@patch("nemoguardrails.library.domain_hallucination.utils.extractors.extract_all")
def test_verify_entities_in_answer_dns(mock_extract_all, mock_resolve, mock_github):
    mock_extract_all.return_value = EXTRACTED
    mock_resolve.return_value = {"domain": "example.com", "resolves": True}
    mock_github.return_value = {"full_name": "pytorch/pytorch", "exists": True}

    result = utils.verify_entities_in_answer("See example.com", "dns", github_token="t")

    assert result["verification"]["dns"] == [{"domain": "example.com", "resolves": True}]
    assert "github" not in result["verification"]
    mock_github.assert_not_called()


@patch("nemoguardrails.library.domain_hallucination.utils.verification.check_github_repo")
@patch("nemoguardrails.library.domain_hallucination.utils.verification.check_http_domain")
@patch("nemoguardrails.library.domain_hallucination.utils.verification.resolve_domain")
@patch("nemoguardrails.library.domain_hallucination.utils.extractors.extract_all")
def test_verify_entities_in_answer_http_and_full(mock_extract_all, mock_resolve, mock_http, mock_github):
    mock_extract_all.return_value = EXTRACTED
    mock_resolve.return_value = {"domain": "example.com", "resolves": True}
    mock_http.return_value = {"url": "https://example.com/path", "reachable": True}
    mock_github.return_value = {"full_name": "pytorch/pytorch", "exists": True}

    http_result = utils.verify_entities_in_answer("See example.com", "http")
    full_result = utils.verify_entities_in_answer("See example.com", "full")

    assert http_result["verification"]["http"][0]["reachable"] is True
    assert "github" not in http_result["verification"]
    assert full_result["verification"]["http"][0]["reachable"] is True
    assert full_result["verification"]["github"][0]["exists"] is True
    assert mock_http.call_count == 2
    mock_github.assert_called_once_with({"owner": "pytorch", "repo": "pytorch"}, token=None)


@patch("nemoguardrails.library.domain_hallucination.utils.verify_entities_in_answer")
def test_find_unverified_domains_none(mock_verify):
    mock_verify.return_value = {"verification": {"dns": [], "http": [], "github": []}}

    assert utils.find_unverified_domains("clean") == []


@patch("nemoguardrails.library.domain_hallucination.utils.verify_entities_in_answer")
def test_find_unverified_domains_mixed_failures(mock_verify):
    mock_verify.return_value = {
        "verification": {
            "dns": [
                {"domain": "bad.test", "resolves": False, "status": "nxdomain"},
                {"domain": "good.test", "resolves": True, "status": "resolved"},
            ],
            "http": [{"url": "https://bad.test", "reachable": False, "status": "http_error"}],
            "github": [
                {
                    "full_name": "missing/repo",
                    "exists": False,
                    "status": "repo_not_found",
                }
            ],
        }
    }

    result = utils.find_unverified_domains("mixed", "full")

    assert [item["type"] for item in result] == ["domain", "url", "github_repo"]
    assert result[0]["reason"] == "nxdomain"
    assert result[1]["reason"] == "http_error"
    assert result[2]["repo"] == "missing/repo"


@patch("nemoguardrails.library.domain_hallucination.utils.extractors.extract_all")
def test_extract_suspicious_domains_blacklist_match(mock_extract_all):
    mock_extract_all.return_value = {"domains": [{"host": "Bad.Example"}, {"host": "safe.example"}, {"host": ""}]}

    result = utils.extract_suspicious_domains("text", kb_blacklist=["bad.example"])

    assert result == [
        {
            "type": "blacklisted",
            "domain": "bad.example",
            "source": "kb",
            "evidence": {"host": "Bad.Example"},
        }
    ]


@patch("nemoguardrails.library.domain_hallucination.utils.extractors.extract_all")
def test_extract_suspicious_domains_no_match(mock_extract_all):
    mock_extract_all.return_value = {"domains": [{"host": "safe.example"}]}

    assert utils.extract_suspicious_domains("text") == []


@patch("nemoguardrails.library.domain_hallucination.utils.extractors.extract_all")
def test_count_entities_by_type_empty_and_populated(mock_extract_all):
    mock_extract_all.return_value = {"urls": [], "domains": [], "github_repos": []}
    assert utils.count_entities_by_type("")["total_entities"] == 0

    mock_extract_all.return_value = EXTRACTED
    result = utils.count_entities_by_type("populated")
    assert result == {"urls": 1, "domains": 1, "github_repos": 1, "total_entities": 3}


@patch("nemoguardrails.library.domain_hallucination.utils.extractors.extract_all")
def test_normalize_urls_in_text_no_urls(mock_extract_all):
    mock_extract_all.return_value = {"urls": []}

    result = utils.normalize_urls_in_text("no urls here")

    assert result["normalized_answer"] == "no urls here"
    assert result["url_mappings"] == {}
    assert result["urls_normalized"] == 0


@patch("nemoguardrails.library.domain_hallucination.utils.extractors.extract_all")
def test_normalize_urls_in_text_replaces_multiple_urls(mock_extract_all):
    mock_extract_all.return_value = {
        "urls": [
            {"raw": "example.com", "normalized": "https://example.com/"},
            {"raw": "http://test.com", "normalized": "http://test.com/"},
        ]
    }

    result = utils.normalize_urls_in_text("Visit example.com and http://test.com")

    assert result["urls_normalized"] == 2
    assert "https://example.com/" in result["normalized_answer"]
    assert result["url_mappings"]["example.com"] == "https://example.com/"


@patch("nemoguardrails.library.domain_hallucination.utils.extractors.extract_all")
def test_get_domain_stats_empty_and_populated(mock_extract_all):
    mock_extract_all.return_value = {"urls": [], "domains": [], "github_repos": []}
    empty = utils.get_domain_stats("")

    mock_extract_all.return_value = {
        "urls": [{"host": "example.com"}, {"host": "docs.example.com"}],
        "domains": [
            {"host": "example.com", "registered_domain": "example.com", "suffix": "com"},
            {
                "host": "docs.example.com",
                "registered_domain": "example.com",
                "suffix": "com",
            },
        ],
        "github_repos": [{"owner": "pytorch", "repo": "pytorch"}],
    }
    populated = utils.get_domain_stats("text")

    assert empty["total_domains"] == 0
    assert populated["unique_registered_domains"] == 1
    assert populated["unique_hosts"] == 2
    assert populated["tlds"] == ["com"]
    assert populated["has_github_repos"] is True


@patch("nemoguardrails.library.domain_hallucination.utils.logger")
def test_log_analysis_result_analyzed_with_issues(mock_logger):
    utils.log_analysis_result(
        {
            "status": "analyzed",
            "detection": {
                "has_issues": True,
                "issue_summary": {"total": 2, "highest_severity": "high"},
            },
            "risk_score": {"score": 80.0, "level": "L3"},
            "decision": {"action": "block"},
        },
        level="WARNING",
    )

    messages = [call.args[0] for call in mock_logger.warning.call_args_list]
    assert "Analysis Status: analyzed" in messages
    assert "  Issue Count: 2" in messages
    assert "  Decision: block" in messages


@patch("nemoguardrails.library.domain_hallucination.utils.logger")
def test_log_analysis_result_skipped_and_unknown_level(mock_logger):
    utils.log_analysis_result({"status": "skipped"}, level="NOT_A_LEVEL")

    mock_logger.not_a_level.assert_called_once_with("Analysis Status: skipped")

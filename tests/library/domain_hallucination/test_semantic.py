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

"""Tests for semantic domain hallucination checks."""

from nemoguardrails.library.domain_hallucination import semantic


def test_check_semantic_relevance_empty_inputs():
    result = semantic.check_semantic_relevance()

    assert result["has_irrelevant_domains"] is False
    assert result["irrelevant_domains"] == []
    assert result["confidence"] == "low"


def test_check_semantic_relevance_no_domains():
    result = semantic.check_semantic_relevance(user_query="find pytorch docs")

    assert result["has_irrelevant_domains"] is False


def test_check_semantic_relevance_with_keyword_overlap():
    result = semantic.check_semantic_relevance(
        user_query="find official pytorch installation documentation",
        extracted_domains=[{"host": "pytorch.org"}],
    )

    assert result["has_irrelevant_domains"] is False


def test_check_semantic_relevance_flags_no_overlap_for_long_query():
    result = semantic.check_semantic_relevance(
        user_query="find official pytorch installation documentation",
        extracted_domains=[{"host": "totally-unrelated-example.com"}],
    )

    assert result["has_irrelevant_domains"] is True
    assert result["irrelevant_domains"][0]["domain"] == "totally-unrelated-example.com"
    assert result["irrelevant_domains"][0]["reason"] == "no_keyword_overlap"


def test_check_semantic_relevance_short_query_does_not_flag():
    result = semantic.check_semantic_relevance(
        user_query="pytorch docs",
        extracted_domains=[{"host": "unrelated.example"}],
    )

    assert result["has_irrelevant_domains"] is False


def test_check_semantic_relevance_skips_blank_domain():
    result = semantic.check_semantic_relevance(
        user_query="find official pytorch installation documentation",
        extracted_domains=[{"host": ""}],
    )

    assert result["has_irrelevant_domains"] is False


def test_check_advanced_verification_no_issues():
    result = semantic.check_advanced_verification(
        extracted_urls=[{"normalized": "https://example.com", "scheme": "https"}],
        extracted_github_repos=[{"owner": "pytorch", "repo": "pytorch"}],
    )

    assert result == {"has_issues": False, "issues": []}


def test_check_advanced_verification_http_url_triggers_issue():
    result = semantic.check_advanced_verification(
        extracted_urls=[{"normalized": "http://example.com", "scheme": "http"}]
    )

    assert result["has_issues"] is True
    assert result["issues"][0]["type"] == "insecure_protocol"


def test_check_advanced_verification_http_local_port_is_allowed():
    result = semantic.check_advanced_verification(
        extracted_urls=[{"normalized": "http://localhost:8080", "scheme": "http"}]
    )

    assert result["has_issues"] is False


def test_check_advanced_verification_typosquatting_triggers_issue():
    result = semantic.check_advanced_verification(extracted_github_repos=[{"owner": "githab", "repo": "security-tool"}])

    assert result["has_issues"] is True
    assert result["issues"][0]["type"] == "possible_typosquatting"
    assert result["issues"][0]["similar_to"] == "github"


def test_check_advanced_verification_ignores_blank_repo_parts():
    result = semantic.check_advanced_verification(extracted_github_repos=[{"owner": "", "repo": "repo"}])

    assert result["has_issues"] is False


def test_edit_distance_boundaries():
    assert semantic._edit_distance("github", "github") == 0
    assert semantic._edit_distance("github", "githab") == 1
    assert semantic._edit_distance("kitten", "sitting") == 3
    assert semantic._edit_distance("", "abc") == 3
    assert semantic._edit_distance("abc", "") == 3

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

"""Tests for extractors module."""

import unittest

from nemoguardrails.library.domain_hallucination import extractors


class TestExtractors(unittest.TestCase):
    """Test extraction functions."""

    def test_extract_urls_basic(self):
        """Test basic URL extraction."""
        text = "Visit https://github.com/pytorch/pytorch for more."
        urls = extractors.extract_urls(text)
        assert len(urls) == 1
        assert urls[0]["host"] == "github.com"
        assert urls[0]["normalized"] == "https://github.com/pytorch/pytorch"

    def test_extract_urls_multiple(self):
        """Test extracting multiple URLs."""
        text = """
        Check out:
        - https://pytorch.org
        - https://tensorflow.org
        - www.github.com
        """
        urls = extractors.extract_urls(text)
        assert len(urls) >= 3
        hosts = {u["host"] for u in urls}
        assert "pytorch.org" in hosts
        assert "tensorflow.org" in hosts
        assert "github.com" in hosts

    def test_extract_domains(self):
        """Test domain extraction."""
        text = "Learn about python.org and numpy.org"
        domains = extractors.extract_domains(text)
        domain_hosts = {d["host"] for d in domains}
        assert "python.org" in domain_hosts
        assert "numpy.org" in domain_hosts

    def test_normalize_domain(self):
        """Test domain normalization."""
        assert extractors.normalize_domain("GitHub.COM") == "github.com"
        assert extractors.normalize_domain("www.example.com") == "example.com"
        assert extractors.normalize_domain("https://example.com") == "example.com"
        assert extractors.normalize_domain("example.com:443") == "example.com"

    def test_extract_github_repos(self):
        """Test GitHub repo extraction."""
        urls = [
            {
                "normalized": "https://github.com/pytorch/pytorch",
                "host": "github.com",
            }
        ]
        repos = extractors.extract_github_repos(urls)
        assert len(repos) == 1
        assert repos[0]["owner"] == "pytorch"
        assert repos[0]["repo"] == "pytorch"

    def test_parse_github_url(self):
        """Test GitHub URL parsing."""
        url = "https://github.com/pytorch/pytorch/tree/main/torch"
        parsed = extractors.parse_github_url(url)
        assert parsed["owner"] == "pytorch"
        assert parsed["repo"] == "pytorch"
        assert parsed["link_type"] == "tree"
        assert parsed["branch"] == "main"

    def test_parse_github_url_issue_path(self):
        """Test GitHub issue URLs are parsed as repo references."""
        parsed = extractors.parse_github_url("https://github.com/pytorch/pytorch/issues/123")
        assert parsed["owner"] == "pytorch"
        assert parsed["repo"] == "pytorch"
        assert parsed["link_type"] == "issues"
        assert parsed["path"] == "123"

    def test_parse_github_url_pull_release_and_wiki_paths(self):
        """Test additional GitHub repo subpaths."""
        for url, link_type in [
            ("https://github.com/pytorch/pytorch/pull/1", "pull"),
            ("https://github.com/pytorch/pytorch/releases/tag/v1", "releases"),
            ("https://github.com/pytorch/pytorch/wiki/Home", "wiki"),
        ]:
            parsed = extractors.parse_github_url(url)
            assert parsed["link_type"] == link_type
            assert parsed["owner"] == "pytorch"

    def test_extract_github_repos_deduplicates_paths(self):
        """Test GitHub repo extraction deduplicates identical repo paths."""
        urls = [
            {"normalized": "https://github.com/pytorch/pytorch/issues/1", "host": "github.com"},
            {"normalized": "https://github.com/pytorch/pytorch/issues/1", "host": "github.com"},
        ]
        repos = extractors.extract_github_repos(urls)
        assert len(repos) == 1

    def test_parse_github_url_skips_reserved_paths(self):
        """Test reserved GitHub paths are not mistaken for repos."""
        for path in [
            "https://github.com/orgs/openai",
            "https://github.com/users/octocat",
            "https://github.com/apps/copilot",
            "https://github.com/settings/profile",
            "https://github.com/notifications",
            "https://github.com/codespaces",
            "https://github.com/site",
            "https://github.com/contact",
        ]:
            assert extractors.parse_github_url(path) is None

    def test_parse_github_url_too_short(self):
        """Test owner-only GitHub URL does not parse as repo."""
        assert extractors.parse_github_url("https://github.com/pytorch") is None

    def test_parse_github_url_non_github(self):
        """Test non-GitHub hosts are ignored."""
        assert extractors.parse_github_url("https://gitlab.com/pytorch/pytorch") is None

    def test_extract_all(self):
        """Test extracting all entity types."""
        text = """
        Visit https://pytorch.org for PyTorch.
        Check the repo: https://github.com/pytorch/pytorch
        See also tensorflow.org
        """
        result = extractors.extract_all(text)
        assert result["no_links"] is False
        assert len(result["urls"]) >= 2
        assert len(result["domains"]) >= 2
        assert len(result["github_repos"]) >= 1

    def test_fast_pass_no_links(self):
        """Test fast pass when no links."""
        text = "Machine learning is a subset of AI."
        result = extractors.extract_all(text)
        assert result["no_links"] is True
        assert len(result["urls"]) == 0


class TestURLNormalization(unittest.TestCase):
    """Test URL normalization."""

    def test_clean_url(self):
        """Test URL cleaning."""
        assert extractors.clean_url("https://example.com.") == "https://example.com"
        assert extractors.clean_url("`https://example.com`") == "https://example.com"

    def test_normalize_url(self):
        """Test URL normalization."""
        normalized = extractors.normalize_url("example.com")
        assert normalized.startswith("https://")
        assert "example.com" in normalized

    def test_url_with_trailing_punctuation(self):
        """Test URLs with trailing punctuation."""
        urls = extractors.extract_urls("Check https://example.com.")
        assert len(urls) > 0
        assert urls[0]["normalized"] == "https://example.com"

    def test_extract_urls_edge_cases(self):
        """Test URL extraction with unsupported and unusual inputs."""
        text = """
        ftp://files.example.com
        https://sub.domain.co.uk
        www.nodomain
        """
        urls = extractors.extract_urls(text)
        normalized = {item["normalized"] for item in urls}
        assert "https://sub.domain.co.uk" in normalized
        assert "https://www.nodomain" in normalized

    def test_normalize_url_idempotent(self):
        """Test URL normalization is idempotent."""
        url = "https://EXAMPLE.COM:443/path?query=1"
        norm1 = extractors.normalize_url(url)
        norm2 = extractors.normalize_url(norm1)
        assert norm1 == norm2

    def test_extract_domains_with_ports(self):
        """Test domain extraction ignores ports."""
        text = "Visit example.com:8080 or api.service:3000"
        domains = extractors.extract_domains(text)
        extracted_hosts = {d.get("host", "") for d in domains}
        assert "example.com" in extracted_hosts

    def test_normalize_domain_idn(self):
        """Test internationalized domains are normalized to IDNA."""
        assert extractors.normalize_domain("münchen.de") == "xn--mnchen-3ya.de"

    def test_parse_url_ipv6_literal(self):
        """Test IPv6 URL parsing does not crash."""
        parsed = extractors.parse_url("http://[::1]:8080/path")
        assert parsed["host"] == "::1"

    def test_clean_url_multiple_markers(self):
        """Test URL cleaning with nested markers."""
        cleaned = extractors.clean_url("``https://example.com``")
        assert cleaned == "https://example.com"

    def test_extract_urls_with_fragments(self):
        """Test URL extraction preserves fragments."""
        urls = extractors.extract_urls("Visit https://example.com#section1 and https://docs.org#api")
        assert len(urls) >= 2
        assert any("#" in item.get("normalized", "") for item in urls)

    def test_clean_url_with_backslashes(self):
        """Test URL cleaning handles backslash-wrapped input without crashing."""
        cleaned = extractors.clean_url(r"\\https://example.com\\")
        assert "example.com" in cleaned


class TestDomainParsing(unittest.TestCase):
    """Test domain parsing."""

    def test_split_domain(self):
        """Test splitting domain into parts."""
        parts = extractors.split_domain("subdomain.example.co.uk")
        assert parts["domain"] is not None or parts["subdomain"] is not None

    def test_split_domain_valid(self):
        """Test splitting a valid multi-label domain."""
        parts = extractors.split_domain("sub.example.co.uk")
        assert parts is not None
        assert "example" in str(parts).lower() or "domain" in str(parts).lower()

    def test_normalize_domain_with_underscore(self):
        """Test underscore host labels are rejected cleanly."""
        assert extractors.normalize_domain("_api.example.com") is None

    def test_extract_domains_ipv4_addresses(self):
        """Test IPv4-like text does not crash domain extraction."""
        domains = extractors.extract_domains("Server at 192.168.1.1 and 10.0.0.1")
        assert isinstance(domains, list)

    def test_parse_url_with_credentials(self):
        """Test URL parsing with embedded credentials."""
        parsed = extractors.parse_url("https://user:pass@example.com/path")
        assert parsed["host"] == "example.com"


if __name__ == "__main__":
    unittest.main()

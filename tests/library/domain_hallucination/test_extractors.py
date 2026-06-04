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


class TestDomainParsing(unittest.TestCase):
    """Test domain parsing."""

    def test_split_domain(self):
        """Test splitting domain into parts."""
        parts = extractors.split_domain("subdomain.example.co.uk")
        assert parts["domain"] is not None or parts["subdomain"] is not None


if __name__ == "__main__":
    unittest.main()

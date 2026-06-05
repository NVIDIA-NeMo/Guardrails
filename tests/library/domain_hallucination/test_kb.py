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

"""Tests for knowledge base module."""

import json
import os
import tempfile
import unittest

from nemoguardrails.library.domain_hallucination import kb


class TestKnowledgeBase(unittest.TestCase):
    """Test knowledge base."""

    def setUp(self):
        """Set up test KB."""
        self.kb = kb.KnowledgeBase()

    def test_add_trusted_domain(self):
        """Test adding trusted domain."""
        self.kb.add_trusted_domain("github.com")
        assert self.kb.is_trusted_domain("github.com")
        assert self.kb.is_trusted_domain("GITHUB.COM")  # Case insensitive

    def test_add_trusted_github_repo(self):
        """Test adding trusted GitHub repo."""
        self.kb.add_trusted_github_repo("pytorch", "pytorch")
        assert self.kb.is_trusted_github_repo("pytorch", "pytorch")
        assert self.kb.is_trusted_github_repo("PYTORCH", "PYTORCH")  # Case insensitive

    def test_add_blacklisted_domain(self):
        """Test adding blacklisted domain."""
        self.kb.add_blacklisted_domain("phishing.com", reason="Known phishing")
        assert self.kb.is_blacklisted_domain("phishing.com")
        assert self.kb.get_blacklist_reason("phishing.com") == "Known phishing"

    def test_load_seed_kb(self):
        """Test loading seed KB."""
        seed_data = {
            "trusted_domains": [
                {"domain": "github.com", "category": "vcs"},
                "pytorch.org",
            ],
            "trusted_github_repos": [
                {"owner": "pytorch", "repo": "pytorch"},
                "tensorflow/tensorflow",
            ],
            "blacklisted_domains": [
                {"domain": "phishing.com", "reason": "Phishing site"},
            ],
        }

        self.kb.load_seed_kb(seed_data)

        assert self.kb.is_trusted_domain("github.com")
        assert self.kb.is_trusted_domain("pytorch.org")
        assert self.kb.is_trusted_github_repo("pytorch", "pytorch")
        assert self.kb.is_trusted_github_repo("tensorflow", "tensorflow")
        assert self.kb.is_blacklisted_domain("phishing.com")

    def test_get_stats(self):
        """Test getting KB statistics."""
        self.kb.add_trusted_domain("github.com")
        self.kb.add_trusted_github_repo("pytorch", "pytorch")
        self.kb.add_blacklisted_domain("phishing.com")

        stats = self.kb.get_stats()
        assert stats["trusted_domains_count"] == 1
        assert stats["trusted_github_repos_count"] == 1
        assert stats["blacklisted_domains_count"] == 1

    def test_query_domain_evidence_empty_kb(self):
        """Test querying an empty KB returns a list."""
        evidence = self.kb.query_domain_evidence("example.com")
        assert isinstance(evidence, list)
        assert evidence == []

    def test_is_blacklisted_domain_exact_match(self):
        """Test blacklist exact matching."""
        self.kb.add_blacklisted_domain("evil.com")
        assert self.kb.is_blacklisted_domain("evil.com") is True
        assert self.kb.is_blacklisted_domain("safe.com") is False

    def test_add_trusted_domain_duplicate(self):
        """Test adding duplicate trusted domains does not duplicate stats."""
        self.kb.add_trusted_domain("example.com")
        self.kb.add_trusted_domain("example.com")
        stats = self.kb.get_stats()
        assert stats["trusted_domains_count"] == 1

    def test_get_stats_populated_kb(self):
        """Test stats on populated KB."""
        self.kb.add_trusted_domain("safe.com")
        self.kb.add_blacklisted_domain("bad.com")
        stats = self.kb.get_stats()
        assert stats["trusted_domains_count"] >= 1
        assert stats["blacklisted_domains_count"] >= 1

    def test_kb_add_and_query_multiple_domains(self):
        """Test adding and querying multiple trusted domains."""
        domains = ["python.org", "docs.python.org", "github.com", "pytorch.org"]
        for domain in domains:
            self.kb.add_trusted_domain(domain)
        stats = self.kb.get_stats()
        assert stats["trusted_domains_count"] == len(domains)

    def test_kb_add_and_check_blacklist(self):
        """Test blacklist functionality across several domains."""
        bad_domains = ["malware.com", "phishing.net", "scam.org"]
        for domain in bad_domains:
            self.kb.add_blacklisted_domain(domain)
        for domain in bad_domains:
            assert self.kb.is_blacklisted_domain(domain) is True
        assert self.kb.is_blacklisted_domain("safe.com") is False

    def test_kb_domain_evidence_persistence(self):
        """Test domain evidence can be stored and retrieved."""
        self.kb.add_trusted_domain("example.com", metadata={"source": "unit_test"})
        evidence = self.kb.query_domain_evidence("example.com")
        assert isinstance(evidence, list)
        assert evidence[0]["metadata"]["source"] == "unit_test"

    def test_kb_stats_comprehensive(self):
        """Test comprehensive stats reporting."""
        self.kb.add_trusted_domain("safe1.com")
        self.kb.add_trusted_domain("safe2.com")
        self.kb.add_blacklisted_domain("bad1.com")
        stats = self.kb.get_stats()
        assert "trusted_domains_count" in stats
        assert "blacklisted_domains_count" in stats
        assert stats["trusted_domains_count"] >= 2
        assert stats["blacklisted_domains_count"] >= 1

    def test_initialize_kb_loads_seed_kb_from_file(self):
        """Test initializing KB from a seed file."""
        original = kb._kb_instance
        kb._kb_instance = None
        path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
                json.dump({"trusted_domains": ["safe.com"], "blacklisted_domains": ["bad.com"]}, handle)
                path = handle.name
            kb_inst = kb.initialize_kb(seed_kb_path=path)
            assert kb_inst.is_trusted_domain("safe.com") is True
            assert kb_inst.is_blacklisted_domain("bad.com") is True
        finally:
            if path and os.path.exists(path):
                os.unlink(path)
            kb._kb_instance = original

    def test_query_external_kb_domain_files(self):
        """Test external KB JSON files are queried."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            domains_dir = os.path.join(tmp_dir, "domains")
            os.mkdir(domains_dir)
            with open(os.path.join(domains_dir, "example.com.json"), "w", encoding="utf-8") as handle:
                json.dump({"owner": "unit-test"}, handle)
            self.kb.external_kb_root = tmp_dir
            evidence = self.kb.query_domain_evidence("example.com")
        assert evidence[0]["type"] == "external_kb"
        assert evidence[0]["data"] == {"owner": "unit-test"}

    def test_initialize_kb_sets_external_root(self):
        """Test initialize_kb stores external KB root."""
        original = kb._kb_instance
        kb._kb_instance = None
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                kb_inst = kb.initialize_kb(external_kb_root=tmp_dir)
                assert kb_inst.external_kb_root == tmp_dir
        finally:
            kb._kb_instance = original

    def test_kb_add_trusted_github_repo_stats(self):
        """Test trusted GitHub repos are counted in stats."""
        self.kb.add_trusted_github_repo("pytorch", "pytorch")
        stats = self.kb.get_stats()
        assert stats["trusted_github_repos_count"] == 1
        assert self.kb.is_trusted_github_repo("pytorch", "pytorch") is True

    def test_kb_multiple_operations(self):
        """Test multiple KB operations in sequence."""
        self.kb.add_trusted_domain("python.org")
        self.kb.add_trusted_domain("github.com")
        self.kb.add_blacklisted_domain("malware.com")
        evidence = self.kb.query_domain_evidence("python.org")
        stats = self.kb.get_stats()
        assert isinstance(evidence, list)
        assert self.kb.is_blacklisted_domain("malware.com") is True
        assert stats["trusted_domains_count"] >= 2

    def test_kb_stats_all_categories(self):
        """Test stats include all categories."""
        self.kb.add_trusted_domain("d1.com")
        self.kb.add_blacklisted_domain("bad.com")
        stats = self.kb.get_stats()
        assert "trusted_domains_count" in stats
        assert "trusted_github_repos_count" in stats
        assert "blacklisted_domains_count" in stats


if __name__ == "__main__":
    unittest.main()

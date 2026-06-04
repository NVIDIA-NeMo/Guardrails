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


if __name__ == "__main__":
    unittest.main()

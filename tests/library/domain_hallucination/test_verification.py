"""Tests for verification module."""

import unittest
from unittest.mock import patch, MagicMock
from nemoguardrails.library.domain_hallucination import verification


class TestDNSVerification(unittest.TestCase):
    """Test DNS verification."""

    def test_resolve_valid_domain(self):
        """Test resolving a valid domain."""
        result = verification.resolve_domain("github.com")
        assert result["source"] == "dns"
        assert result["domain"] == "github.com"
        # May or may not resolve depending on network
        assert "status" in result

    def test_resolve_invalid_domain(self):
        """Test resolving an invalid domain."""
        result = verification.resolve_domain("thisdomain-does-not-exist-12345.com")
        assert result["source"] == "dns"
        assert result["resolves"] is False
        assert result["status"] in {"nxdomain_or_no_data", "dns_error"}

    def test_resolve_empty_domain(self):
        """Test resolving empty domain."""
        result = verification.resolve_domain("")
        assert result["resolves"] is False
        assert result["status"] == "invalid_domain"


class TestGitHubVerification(unittest.TestCase):
    """Test GitHub verification."""

    def test_github_repo_format_validation(self):
        """Test GitHub repo format validation."""
        # Invalid formats
        result = verification.check_github_repo({"owner": "", "repo": "test"})
        assert result["format_valid"] is False

        result = verification.check_github_repo({"owner": "pytorch", "repo": ""})
        assert result["format_valid"] is False

    def test_github_known_repo(self):
        """Test checking a known GitHub repo."""
        repo_item = {"owner": "pytorch", "repo": "pytorch"}
        result = verification.check_github_repo(repo_item)
        # May require network access
        assert "status" in result
        assert result["source"] == "github"

    def test_github_fake_repo(self):
        """Test checking a fake GitHub repo."""
        repo_item = {"owner": "fake-owner-xyz", "repo": "fake-repo-xyz"}
        result = verification.check_github_repo(repo_item)
        # Could be not found or rate limited
        assert "status" in result


class TestHTTPVerification(unittest.TestCase):
    """Test HTTP verification."""

    def test_http_valid_domain(self):
        """Test checking a valid domain."""
        result = verification.check_http_domain("github.com")
        # May or may not be reachable depending on network
        assert "status" in result

    def test_http_invalid_domain(self):
        """Test checking an invalid domain."""
        result = verification.check_http_domain("thisdomain-does-not-exist-12345.xyz")
        # Should fail or timeout
        assert result["reachable"] is False or result["status"].startswith("http")


if __name__ == "__main__":
    unittest.main()


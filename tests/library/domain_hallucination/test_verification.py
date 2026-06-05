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

"""Tests for verification module."""

import socket
import ssl
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from nemoguardrails.library.domain_hallucination import verification


class TestDNSVerification(unittest.TestCase):
    """Test DNS verification."""

    @patch("nemoguardrails.library.domain_hallucination.verification.socket.getaddrinfo")
    def test_resolve_valid_domain(self, mock_getaddrinfo):
        """Test resolving a valid domain without real DNS."""
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("140.82.121.4", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("140.82.121.4", 443)),
        ]

        result = verification.resolve_domain("github.com")

        assert result["source"] == "dns"
        assert result["domain"] == "github.com"
        assert result["status"] == "resolved"
        assert result["resolves"] is True
        assert result["public_address_count"] == 1

    @patch(
        "nemoguardrails.library.domain_hallucination.verification.socket.getaddrinfo",
        side_effect=socket.gaierror("not found"),
    )
    def test_resolve_invalid_domain(self, _mock_getaddrinfo):
        """Test resolving an invalid domain without real DNS."""
        result = verification.resolve_domain("thisdomain-does-not-exist-12345.com")
        assert result["source"] == "dns"
        assert result["resolves"] is False
        assert result["status"] == "nxdomain_or_no_data"

    def test_resolve_empty_domain(self):
        """Test resolving empty domain."""
        result = verification.resolve_domain("")
        assert result["resolves"] is False
        assert result["status"] == "invalid_domain"


class TestGitHubVerification(unittest.TestCase):
    """Test GitHub verification."""

    def test_github_repo_format_validation(self):
        """Test GitHub repo format validation."""
        result = verification.check_github_repo({"owner": "", "repo": "test"})
        assert result["format_valid"] is False

        result = verification.check_github_repo({"owner": "pytorch", "repo": ""})
        assert result["format_valid"] is False

    @patch("nemoguardrails.library.domain_hallucination.verification.urlopen")
    def test_github_known_repo(self, mock_urlopen):
        """Test checking a known GitHub repo without a real request."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.status = 200
        mock_response.geturl.return_value = "https://api.github.com/repos/pytorch/pytorch"
        mock_response.read.return_value = b'{"full_name":"pytorch/pytorch","private":false}'
        mock_urlopen.return_value = mock_response

        repo_item = {"owner": "pytorch", "repo": "pytorch"}
        result = verification.check_github_repo(repo_item)

        assert result["source"] == "github"
        assert result["status"] == "repo_exists"
        assert result["exists"] is True

    @patch(
        "nemoguardrails.library.domain_hallucination.verification.urlopen",
        side_effect=HTTPError(
            url="https://api.github.com/repos/fake-owner-xyz/fake-repo-xyz",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        ),
    )
    def test_github_fake_repo(self, _mock_urlopen):
        """Test checking a fake GitHub repo without a real request."""
        repo_item = {"owner": "fake-owner-xyz", "repo": "fake-repo-xyz"}
        result = verification.check_github_repo(repo_item)
        assert result["source"] == "github"
        assert result["status"] == "repo_not_found"
        assert result["exists"] is False

    @patch("nemoguardrails.library.domain_hallucination.verification.urlopen", side_effect=TimeoutError("API timeout"))
    def test_github_timeout(self, _mock_urlopen):
        """Test checking a GitHub repo handles provider errors."""
        result = verification.check_github_repo({"owner": "pytorch", "repo": "pytorch"})
        assert result["source"] == "github"
        assert result["status"] == "github_error"
        assert result["exists"] is None

    @patch(
        "nemoguardrails.library.domain_hallucination.verification.urlopen",
        side_effect=HTTPError(
            url="https://api.github.com/repos/pytorch/pytorch",
            code=500,
            msg="Server Error",
            hdrs=None,
            fp=None,
        ),
    )
    def test_github_http_error(self, _mock_urlopen):
        """Test GitHub non-404 HTTP errors are structured."""
        result = verification.check_github_repo({"owner": "pytorch", "repo": "pytorch"})
        assert result["status"] == "github_http_error"
        assert result["exists"] is None


class TestHTTPVerification(unittest.TestCase):
    """Test HTTP verification."""

    @patch("nemoguardrails.library.domain_hallucination.verification.urlopen")
    def test_http_valid_domain(self, mock_urlopen):
        """Test checking a valid domain without a real request."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.status = 200
        mock_response.geturl.return_value = "https://github.com/"
        mock_urlopen.return_value = mock_response

        result = verification.check_http_domain("github.com")

        assert result["source"] == "http"
        assert result["status"] == "http_ok"
        assert result["reachable"] is True
        assert result["domain"] == "github.com"

    @patch(
        "nemoguardrails.library.domain_hallucination.verification.urlopen",
        side_effect=TimeoutError("timed out"),
    )
    def test_http_invalid_domain(self, _mock_urlopen):
        """Test checking an invalid domain without a real request."""
        result = verification.check_http_domain("thisdomain-does-not-exist-12345.xyz")
        assert result["source"] == "http"
        assert result["status"] == "http_timeout"
        assert result["reachable"] is False

    @patch("nemoguardrails.library.domain_hallucination.verification.urlopen", side_effect=OSError("connection failed"))
    def test_http_connection_error(self, _mock_urlopen):
        """Test HTTP connection errors are reported."""
        result = verification.check_http_domain("example.com")
        assert result["source"] == "http"
        assert result["status"] == "http_error"
        assert result["reachable"] is False

    @patch("nemoguardrails.library.domain_hallucination.verification.urlopen", side_effect=TimeoutError("timed out"))
    def test_http_timeout_handling(self, _mock_urlopen):
        """Test HTTP timeout is handled gracefully."""
        result = verification.check_http_domain("slow.example.com", timeout=0.001)
        assert result["source"] == "http"
        assert result["status"] == "http_timeout"
        assert result["reachable"] is False

    def test_http_empty_target(self):
        """Test empty HTTP target is invalid."""
        result = verification.check_http_domain("")
        assert result["status"] == "invalid_url"
        assert result["reachable"] is False


class TestTLSVerification(unittest.TestCase):
    """Test TLS verification."""

    @patch("nemoguardrails.library.domain_hallucination.verification.ssl.create_default_context")
    @patch("nemoguardrails.library.domain_hallucination.verification.socket.create_connection")
    def test_tls_ok(self, mock_create_connection, mock_create_default_context):
        """Test TLS verification without a real socket."""
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        mock_secure_sock = MagicMock()
        mock_secure_sock.__enter__.return_value = mock_secure_sock
        mock_secure_sock.getpeercert.return_value = {"notAfter": "Dec 31 23:59:59 2099 GMT"}

        mock_context = MagicMock()
        mock_context.wrap_socket.return_value = mock_secure_sock
        mock_create_default_context.return_value = mock_context
        mock_create_connection.return_value = mock_sock

        result = verification.check_tls("github.com")

        assert result["source"] == "tls"
        assert result["status"] == "tls_ok"
        assert result["confidence"] == "medium"

    @patch(
        "nemoguardrails.library.domain_hallucination.verification.socket.create_connection",
        side_effect=socket.timeout("timed out"),
    )
    def test_tls_timeout(self, _mock_create_connection):
        """Test TLS timeout without a real socket."""
        result = verification.check_tls("github.com")
        assert result["source"] == "tls"
        assert result["status"] == "tls_timeout"

    @patch(
        "nemoguardrails.library.domain_hallucination.verification.socket.create_connection",
        side_effect=ssl.SSLCertVerificationError("hostname mismatch"),
    )
    def test_tls_verification_failure(self, _mock_create_connection):
        """Test TLS verification failure without a real socket."""
        result = verification.check_tls("github.com")
        assert result["source"] == "tls"
        assert result["status"] in {"tls_hostname_mismatch", "tls_untrusted_chain", "tls_verification_failed"}

    @patch(
        "nemoguardrails.library.domain_hallucination.verification.socket.create_connection",
        side_effect=OSError("socket failed"),
    )
    def test_tls_generic_error(self, _mock_create_connection):
        """Test TLS generic errors return a structured result."""
        result = verification.check_tls("github.com")
        assert result["source"] == "tls"
        assert result["status"] == "tls_error"


class TestWhoisVerification(unittest.TestCase):
    """Test WHOIS/RDAP verification."""

    def test_whois_empty_domain(self):
        """Test WHOIS empty input."""
        result = verification.check_whois("")
        assert result["status"] == "invalid_domain"
        assert result["enabled"] is False

    @patch("nemoguardrails.library.domain_hallucination.verification._check_rdap", side_effect=ValueError("bad rdap"))
    def test_whois_parse_error(self, _mock_check_rdap):
        """Test WHOIS parsing errors are handled."""
        result = verification.check_whois("example.com")
        assert result["status"] == "whois_error"
        assert result["enabled"] is False

    @patch(
        "nemoguardrails.library.domain_hallucination.verification._check_rdap", side_effect=TimeoutError("slow rdap")
    )
    def test_whois_timeout(self, _mock_check_rdap):
        """Test WHOIS timeout is handled."""
        result = verification.check_whois("example.com", timeout=0.001)
        assert result["status"] == "whois_timeout"
        assert result["enabled"] is False


if __name__ == "__main__":
    unittest.main()

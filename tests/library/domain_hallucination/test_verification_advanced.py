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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advanced verification tests for error handling and edge cases."""

import socket
import unittest
from unittest.mock import patch

from nemoguardrails.library.domain_hallucination import verification


class TestDNSVerification(unittest.TestCase):
    """DNS verification error paths."""

    @patch("nemoguardrails.library.domain_hallucination.verification.socket.getaddrinfo")
    def test_resolve_domain_nxdomain(self, mock_getaddrinfo):
        """NXDOMAIN response."""
        mock_getaddrinfo.side_effect = socket.gaierror("not found")
        result = verification.resolve_domain("nonexistent.test")
        assert result["status"] == "nxdomain_or_no_data"

    @patch("nemoguardrails.library.domain_hallucination.verification.socket.getaddrinfo")
    def test_resolve_domain_no_a_records(self, mock_getaddrinfo):
        """Domain with no A records."""
        mock_getaddrinfo.return_value = []
        result = verification.resolve_domain("norecords.test")
        assert result["status"] == "no_data"

    @patch("nemoguardrails.library.domain_hallucination.verification.socket.getaddrinfo")
    def test_resolve_domain_exception(self, mock_getaddrinfo):
        """Generic DNS exception."""
        mock_getaddrinfo.side_effect = Exception("DNS error")
        result = verification.resolve_domain("error.test")
        assert isinstance(result, dict)
        assert result["status"] == "dns_error"


class TestHTTPVerification(unittest.TestCase):
    """HTTP verification error paths."""

    def test_check_http_empty_target(self):
        """Empty HTTP target."""
        result = verification.check_http_domain("")
        assert result["status"] == "invalid_url"

    @patch("nemoguardrails.library.domain_hallucination.verification.urlopen")
    def test_check_http_domain_connection_error(self, mock_urlopen):
        """Connection error."""
        mock_urlopen.side_effect = OSError("Connection failed")
        result = verification.check_http_domain("http://refused.test")
        assert result["reachable"] is False
        assert result["status"] == "http_error"

    @patch("nemoguardrails.library.domain_hallucination.verification.urlopen")
    def test_check_http_domain_timeout(self, mock_urlopen):
        """HTTP timeout."""
        mock_urlopen.side_effect = TimeoutError("Timeout")
        result = verification.check_http_domain("http://slow.test")
        assert result["reachable"] is False
        assert result["status"] == "http_timeout"


class TestTLSVerification(unittest.TestCase):
    """TLS verification error paths."""

    @patch("nemoguardrails.library.domain_hallucination.verification.socket.create_connection")
    def test_check_tls_connection_error(self, mock_socket):
        """TLS connection error."""
        mock_socket.side_effect = OSError("Connection refused")
        result = verification.check_tls("refused.test")
        assert result["status"] in ["error", "connection_failed", "tls_error"]

    def test_check_tls_empty_domain(self):
        """Empty TLS domain."""
        result = verification.check_tls("")
        assert isinstance(result, dict)


class TestGitHubVerification(unittest.TestCase):
    """GitHub verification error paths."""

    @patch("nemoguardrails.library.domain_hallucination.verification.urlopen")
    def test_check_github_repo_timeout(self, mock_urlopen):
        """GitHub API timeout."""
        mock_urlopen.side_effect = TimeoutError("Timeout")
        result = verification.check_github_repo({"owner": "pytorch", "repo": "pytorch"})
        assert result["status"] == "github_error"

    def test_check_github_repo_empty_input(self):
        """Empty repo info."""
        result = verification.check_github_repo({"owner": "", "repo": ""})
        assert isinstance(result, dict)


class TestEdgeCases(unittest.TestCase):
    """Edge cases."""

    def test_check_http_domain_invalid_url(self):
        """Invalid URL format."""
        result = verification.check_http_domain("not a url")
        assert isinstance(result, dict)

    def test_resolve_domain_empty(self):
        """Empty domain."""
        result = verification.resolve_domain("")
        assert isinstance(result, dict)

    @patch("nemoguardrails.library.domain_hallucination.verification.urlopen")
    def test_check_http_exception_handling(self, mock_urlopen):
        """Generic exception in HTTP."""
        mock_urlopen.side_effect = Exception("Unexpected")
        result = verification.check_http_domain("http://error.test")
        assert isinstance(result, dict)
        assert result["status"] == "http_error"


if __name__ == "__main__":
    unittest.main()

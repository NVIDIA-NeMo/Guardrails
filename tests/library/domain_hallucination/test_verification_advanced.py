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

import json
import socket
import ssl
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

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

    @patch("nemoguardrails.library.domain_hallucination.verification.socket.setdefaulttimeout")
    @patch("nemoguardrails.library.domain_hallucination.verification.socket.getdefaulttimeout")
    @patch("nemoguardrails.library.domain_hallucination.verification.socket.getaddrinfo")
    def test_resolve_domain_applies_and_restores_timeout(
        self, mock_getaddrinfo, mock_getdefaulttimeout, mock_setdefaulttimeout
    ):
        """DNS timeout argument is applied and restored around getaddrinfo."""
        mock_getdefaulttimeout.return_value = 12.0
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 443))
        ]

        result = verification.resolve_domain("example.com", timeout=1.5)

        assert result["status"] == "resolved"
        mock_setdefaulttimeout.assert_any_call(1.5)
        mock_setdefaulttimeout.assert_called_with(12.0)


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

    @patch("nemoguardrails.library.domain_hallucination.verification.urlopen")
    def test_check_http_403_https_falls_back_to_http(self, mock_urlopen):
        """403 on HTTPS should continue to the HTTP fallback URL."""
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.status = 200
        mock_response.geturl.return_value = "http://fallback.test/"
        mock_urlopen.side_effect = [
            HTTPError(
                url="https://fallback.test/",
                code=403,
                msg="Forbidden",
                hdrs=None,
                fp=None,
            ),
            mock_response,
        ]

        result = verification.check_http_domain("fallback.test")

        assert result["status"] == "http_ok"
        assert result["url"] == "http://fallback.test/"
        assert mock_urlopen.call_count == 2

    @patch(
        "nemoguardrails.library.domain_hallucination.verification.urlopen",
        side_effect=HTTPError(
            url="https://forbidden.test/",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=None,
        ),
    )
    def test_check_http_403_single_url_returns_structured_fallback(self, _mock_urlopen):
        """A single HTTPS URL 403 should never return an empty dict."""
        result = verification.check_http_domain("https://forbidden.test/")

        assert result["source"] == "http"
        assert result["status"] == "http_error"
        assert result["reachable"] is False
        assert result["error"] == "no_successful_url"

    @patch("nemoguardrails.library.domain_hallucination.verification.urlopen")
    def test_check_http_blocks_private_ip_literal(self, mock_urlopen):
        """Private IP literals are blocked before urlopen to prevent SSRF."""
        result = verification.check_http_domain("http://127.0.0.1/admin")

        assert result["status"] == "ssrf_blocked"
        assert result["reachable"] is False
        mock_urlopen.assert_not_called()


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

    def _mock_tls_result(self, not_after):
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        mock_secure_sock = MagicMock()
        mock_secure_sock.__enter__.return_value = mock_secure_sock
        mock_secure_sock.getpeercert.return_value = {"notAfter": not_after}
        mock_context = MagicMock()
        mock_context.wrap_socket.return_value = mock_secure_sock
        return mock_sock, mock_context

    @patch("nemoguardrails.library.domain_hallucination.verification.ssl.create_default_context")
    @patch("nemoguardrails.library.domain_hallucination.verification.socket.create_connection")
    def test_check_tls_expired(self, mock_socket, mock_context_factory):
        """Expired certificates are flagged."""
        not_after = "Jan 01 00:00:00 2000 GMT"
        mock_socket.return_value, mock_context_factory.return_value = self._mock_tls_result(not_after)

        result = verification.check_tls("expired.test")

        assert result["status"] == "tls_expired"
        assert result["days_until_expiry"] < 0

    @patch("nemoguardrails.library.domain_hallucination.verification.ssl.create_default_context")
    @patch("nemoguardrails.library.domain_hallucination.verification.socket.create_connection")
    def test_check_tls_expiring_soon(self, mock_socket, mock_context_factory):
        """Certificates expiring within 30 days are flagged."""
        not_after = (datetime.now(timezone.utc) + timedelta(days=10)).strftime("%b %d %H:%M:%S %Y GMT")
        mock_socket.return_value, mock_context_factory.return_value = self._mock_tls_result(not_after)

        result = verification.check_tls("soon.test")

        assert result["status"] == "tls_expiring_soon"
        assert 0 <= result["days_until_expiry"] <= 30

    @patch(
        "nemoguardrails.library.domain_hallucination.verification.socket.create_connection",
        side_effect=ssl.SSLCertVerificationError("hostname mismatch"),
    )
    def test_check_tls_hostname_mismatch(self, _mock_socket):
        """Hostname mismatch errors get a specific status."""
        result = verification.check_tls("wrong-host.test")

        assert result["status"] == "tls_hostname_mismatch"

    @patch(
        "nemoguardrails.library.domain_hallucination.verification.socket.create_connection",
        side_effect=ssl.SSLCertVerificationError("self-signed certificate"),
    )
    def test_check_tls_untrusted_chain(self, _mock_socket):
        """Self-signed certificate errors get a specific status."""
        result = verification.check_tls("self-signed.test")

        assert result["status"] == "tls_untrusted_chain"


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

    def test_check_github_repo_invalid_format(self):
        """Invalid owner/repo characters are rejected before network calls."""
        result = verification.check_github_repo({"owner": "-bad", "repo": "repo name"})

        assert result["status"] == "invalid_repo_item"
        assert result["format_valid"] is False
        assert result["exists"] is False

    @patch(
        "nemoguardrails.library.domain_hallucination.verification.urlopen",
        side_effect=HTTPError(
            url="https://api.github.com/repos/pytorch/pytorch",
            code=403,
            msg="rate limited",
            hdrs=None,
            fp=None,
        ),
    )
    def test_check_github_repo_rate_limited_403(self, _mock_urlopen):
        """403 responses are treated as rate limits."""
        result = verification.check_github_repo({"owner": "pytorch", "repo": "pytorch"})

        assert result["status"] == "github_rate_limited"
        assert result["exists"] is None
        assert result["use_in_scoring"] is False

    @patch(
        "nemoguardrails.library.domain_hallucination.verification.urlopen",
        side_effect=HTTPError(
            url="https://api.github.com/repos/pytorch/pytorch",
            code=429,
            msg="too many requests",
            hdrs=None,
            fp=None,
        ),
    )
    def test_check_github_repo_rate_limited_429(self, _mock_urlopen):
        """429 responses are treated as rate limits."""
        result = verification.check_github_repo({"owner": "pytorch", "repo": "pytorch"})

        assert result["status"] == "github_rate_limited"
        assert result["status_code"] == 429

    @patch(
        "nemoguardrails.library.domain_hallucination.verification.urlopen",
        side_effect=HTTPError(
            url="https://api.github.com/repos/pytorch/pytorch",
            code=500,
            msg="server error",
            hdrs=None,
            fp=None,
        ),
    )
    def test_check_github_repo_non_rate_limit_http_error(self, _mock_urlopen):
        """Non-404/rate-limit errors are preserved as HTTP errors."""
        result = verification.check_github_repo({"owner": "pytorch", "repo": "pytorch"})

        assert result["status"] == "github_http_error"
        assert result["status_code"] == 500


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


class TestWhoisHelpers(unittest.TestCase):
    """WHOIS/RDAP helper paths."""

    def test_parse_whois_date_list_datetime_iso_and_invalid(self):
        """WHOIS date parser accepts common shapes."""
        naive = datetime(2024, 1, 1, 12, 0, 0)
        aware = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)

        assert verification._parse_whois_date([naive]).tzinfo == timezone.utc
        assert verification._parse_whois_date(aware) == aware
        assert verification._parse_whois_date("2024-01-03T12:00:00Z").tzinfo == timezone.utc
        assert verification._parse_whois_date(None) is None
        assert verification._parse_whois_date("not-a-date") is None

    def test_calculate_age_and_days_until(self):
        """Age/expiry helpers handle dates and None."""
        past = datetime.now(timezone.utc) - timedelta(days=5)
        future = datetime.now(timezone.utc) + timedelta(days=10)

        assert verification._calculate_age_days(past) >= 4
        assert verification._calculate_age_days(None) is None
        assert verification._calculate_days_until(future) >= 9
        assert verification._calculate_days_until(None) is None

    def test_build_whois_payload_recent_and_error(self):
        """Payload construction includes recent-domain hints and errors."""
        started = verification.time.perf_counter()
        recent = datetime.now(timezone.utc) - timedelta(days=3)
        expires = datetime.now(timezone.utc) + timedelta(days=300)

        payload = verification._build_whois_payload(
            domain="new.test",
            status="ok",
            enabled=True,
            registrar="Registrar",
            created=recent,
            expires=expires,
            source_method="rdap",
            started=started,
            error="sample",
        )
        no_dates = verification._build_whois_payload(
            domain="unknown.test",
            status="whois_error",
            enabled=False,
            source_method="rdap",
            started=started,
            use_in_scoring=False,
        )

        assert payload["is_recent_domain"] is True
        assert payload["risk_hint"] == "recent_domain"
        assert payload["error"] == "sample"
        assert payload["use_in_scoring"] is True
        assert no_dates["is_recent_domain"] is None
        assert no_dates["use_in_scoring"] is False

    def test_extract_rdap_event_date(self):
        """RDAP event extraction matches requested actions."""
        data = {
            "events": [
                {"eventAction": "last changed", "eventDate": "2024-01-01T00:00:00Z"},
                {"eventAction": "registration", "eventDate": "2024-02-01T00:00:00Z"},
            ]
        }

        parsed = verification._extract_rdap_event_date(data, {"registration"})

        assert parsed.month == 2
        assert verification._extract_rdap_event_date(data, {"expiration"}) is None
        assert verification._extract_rdap_event_date({"events": "bad"}, {"registration"}) is None

    def test_extract_rdap_registrar(self):
        """Registrar extraction reads RDAP vCard data."""
        data = {
            "entities": [
                {"roles": ["registrant"]},
                {
                    "roles": ["registrar"],
                    "vcardArray": ["vcard", [["fn", {}, "text", "Example Registrar"]]],
                },
            ]
        }

        assert verification._extract_rdap_registrar(data) == "Example Registrar"
        assert verification._extract_rdap_registrar({"entities": []}) is None
        assert verification._extract_rdap_registrar({"entities": [{"roles": ["registrar"]}]}) is None

    def test_registered_domain_candidates(self):
        """Registered-domain candidates include ccTLD forms."""
        assert verification._registered_domain_candidates("docs.example.com") == [
            "docs.example.com",
            "example.com",
        ]
        assert verification._registered_domain_candidates("a.b.co.uk") == [
            "a.b.co.uk",
            "b.co.uk",
        ]

    @patch("nemoguardrails.library.domain_hallucination.verification.urllib.request.urlopen")
    def test_check_rdap_success(self, mock_urlopen):
        """RDAP success response produces a WHOIS payload."""
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps(
            {
                "events": [
                    {"eventAction": "registration", "eventDate": "2024-01-01T00:00:00Z"},
                    {"eventAction": "expiration", "eventDate": "2030-01-01T00:00:00Z"},
                ],
                "entities": [
                    {
                        "roles": ["registrar"],
                        "vcardArray": ["vcard", [["fn", {}, "text", "Registrar Inc."]]],
                    }
                ],
            }
        ).encode()
        mock_urlopen.return_value = response

        result = verification._check_rdap("example.com", timeout=1.0, started=0.0)

        assert result["status"] == "ok"
        assert result["enabled"] is True
        assert result["registrar"] == "Registrar Inc."
        assert result["queried_domain"] == "example.com"

    @patch(
        "nemoguardrails.library.domain_hallucination.verification.urllib.request.urlopen",
        side_effect=URLError("rdap down"),
    )
    def test_check_rdap_all_candidates_fail(self, _mock_urlopen):
        """RDAP raises the last lookup error when all candidates fail."""
        with self.assertRaises(URLError):
            verification._check_rdap("docs.example.com", timeout=1.0, started=0.0)


if __name__ == "__main__":
    unittest.main()

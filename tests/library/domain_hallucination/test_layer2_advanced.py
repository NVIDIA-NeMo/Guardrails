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

# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Layer 2 advanced network verification."""

import asyncio
import json
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemoguardrails.library.domain_hallucination import layer2_advanced


class TestVerifyDomainNetwork:
    """Test network verification function."""

    @pytest.mark.asyncio
    async def test_verify_domain_network_success(self):
        """Test successful domain resolution."""
        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            # Mock successful DNS resolution
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            ]

            result = await layer2_advanced.verify_domain_network("example.com")

            assert result["domain"] == "example.com"
            assert result["status"] == "resolved"
            assert result["address_count"] == 1

    @pytest.mark.asyncio
    async def test_verify_domain_network_nxdomain(self):
        """Test NXDOMAIN (domain not found) error."""
        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            # Mock DNS resolution failure (NXDOMAIN)
            mock_getaddrinfo.side_effect = socket.gaierror(
                socket.EAI_NONAME, "Name or service not known"
            )

            result = await layer2_advanced.verify_domain_network("nonexistent-domain-12345.com")

            assert result["domain"] == "nonexistent-domain-12345.com"
            assert result["status"] == "nxdomain"
            assert result["error"] == "DNS resolution failed"

    @pytest.mark.asyncio
    async def test_verify_domain_network_timeout(self):
        """Test DNS timeout."""
        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            # Mock DNS timeout
            mock_getaddrinfo.side_effect = socket.timeout("DNS timeout")

            result = await layer2_advanced.verify_domain_network("example.com", timeout=1.0)

            assert result["domain"] == "example.com"
            assert result["status"] == "timeout"
            assert result["error"] == "DNS timeout"

    @pytest.mark.asyncio
    async def test_verify_domain_network_generic_error(self):
        """Test generic socket error."""
        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            # Mock generic error
            mock_getaddrinfo.side_effect = Exception("Connection refused")

            result = await layer2_advanced.verify_domain_network("example.com")

            assert result["domain"] == "example.com"
            assert result["status"] == "error"
            assert "error" in result

    @pytest.mark.asyncio
    async def test_verify_domain_network_multiple_addresses(self):
        """Test domain with multiple IP addresses."""
        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            # Mock multiple DNS results
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
                (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:2800:220:1:248:1893:25c8:1946", 443)),
            ]

            result = await layer2_advanced.verify_domain_network("example.com")

            assert result["status"] == "resolved"
            assert result["address_count"] == 2

    @pytest.mark.asyncio
    async def test_verify_domain_network_uses_executor(self):
        """Test that DNS resolution runs in executor (non-blocking)."""
        with patch("socket.getaddrinfo") as mock_getaddrinfo, patch(
            "asyncio.get_event_loop"
        ) as mock_get_loop:
            mock_getaddrinfo.return_value = []
            mock_executor = MagicMock()
            mock_loop = MagicMock()
            mock_loop.run_in_executor = AsyncMock(
                return_value=[
                    (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
                ]
            )
            mock_get_loop.return_value = mock_loop

            result = await layer2_advanced.verify_domain_network("example.com")

            # Verify run_in_executor was called (avoiding event loop blocking)
            assert mock_loop.run_in_executor.called


class TestLayer2CheckWithVerification:
    """Test Layer 2 check function with verification results."""

    @pytest.mark.asyncio
    async def test_layer2_check_no_domains(self):
        """Test Layer 2 with no domains to verify."""
        mock_llm_task_manager = MagicMock()
        mock_llm_call_func = AsyncMock()
        mock_config = MagicMock()

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "no"
        mock_llm_call_func.return_value = mock_response

        with patch(
            "nemoguardrails.library.domain_hallucination.layer2_advanced.verify_domain_network"
        ) as mock_verify:
            # No domains to verify
            result = await layer2_advanced.layer2_check_with_verification(
                bot_response="Talk about AI",
                user_message="What is AI?",
                urls=[],
                domains=[],
                github_repos=[],
                llm_call_func=mock_llm_call_func,
                llm_task_manager=mock_llm_task_manager,
                config=mock_config,
            )

            assert result["is_hallucinated"] is False
            assert result["status"] == "verified"
            assert result["layer"] == "layer2"

    @pytest.mark.asyncio
    async def test_layer2_check_json_formatting(self):
        """Test that verification results are properly formatted as JSON."""
        mock_llm_task_manager = MagicMock()
        mock_llm_call_func = AsyncMock()
        mock_config = MagicMock()

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "no"
        mock_llm_call_func.return_value = mock_response

        with patch(
            "nemoguardrails.library.domain_hallucination.layer2_advanced.verify_domain_network"
        ) as mock_verify:
            mock_verify.return_value = {"domain": "example.com", "status": "resolved"}

            result = await layer2_advanced.layer2_check_with_verification(
                bot_response="Visit https://example.com",
                user_message="Where to visit?",
                urls=["https://example.com"],
                domains=["example.com"],
                github_repos=[],
                llm_call_func=mock_llm_call_func,
                llm_task_manager=mock_llm_task_manager,
                config=mock_config,
            )

            # Verify verification_results is a dict (not a string)
            assert isinstance(result["verification_results"], dict)
            assert "example.com" in result["verification_results"]

    @pytest.mark.asyncio
    async def test_layer2_check_llm_yes_response(self):
        """Test Layer 2 when LLM responds 'yes' (hallucinated)."""
        mock_llm_task_manager = MagicMock()
        mock_llm_call_func = AsyncMock()
        mock_config = MagicMock()

        # Mock LLM response indicating hallucination
        mock_response = MagicMock()
        mock_response.content = "yes, this looks fabricated"
        mock_llm_call_func.return_value = mock_response

        with patch(
            "nemoguardrails.library.domain_hallucination.layer2_advanced.verify_domain_network"
        ) as mock_verify:
            mock_verify.return_value = {"domain": "fake.com", "status": "nxdomain"}

            result = await layer2_advanced.layer2_check_with_verification(
                bot_response="Visit https://fake.com",
                user_message="Where?",
                urls=["https://fake.com"],
                domains=["fake.com"],
                github_repos=[],
                llm_call_func=mock_llm_call_func,
                llm_task_manager=mock_llm_task_manager,
                config=mock_config,
            )

            assert result["is_hallucinated"] is True
            assert result["status"] == "suspicious"
            assert "yes" in result["llm_response"]

    @pytest.mark.asyncio
    async def test_layer2_check_llm_error(self):
        """Test Layer 2 error handling when LLM call fails."""
        mock_llm_task_manager = MagicMock()
        mock_llm_call_func = AsyncMock(side_effect=Exception("LLM error"))
        mock_config = MagicMock()

        with patch(
            "nemoguardrails.library.domain_hallucination.layer2_advanced.verify_domain_network"
        ) as mock_verify:
            mock_verify.return_value = {"domain": "example.com", "status": "resolved"}

            result = await layer2_advanced.layer2_check_with_verification(
                bot_response="Visit https://example.com",
                user_message="Where?",
                urls=["https://example.com"],
                domains=["example.com"],
                github_repos=[],
                llm_call_func=mock_llm_call_func,
                llm_task_manager=mock_llm_task_manager,
                config=mock_config,
            )

            # Should return gracefully with error status
            assert result["status"] == "error"
            assert result["is_hallucinated"] is False

    @pytest.mark.asyncio
    async def test_layer2_check_multiple_domains(self):
        """Test Layer 2 with multiple domains."""
        mock_llm_task_manager = MagicMock()
        mock_llm_call_func = AsyncMock()
        mock_config = MagicMock()

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "no"
        mock_llm_call_func.return_value = mock_response

        with patch(
            "nemoguardrails.library.domain_hallucination.layer2_advanced.verify_domain_network"
        ) as mock_verify:
            # Different results for each domain
            def verify_side_effect(domain, timeout=5.0):
                if domain == "example.com":
                    return {"domain": domain, "status": "resolved"}
                else:
                    return {"domain": domain, "status": "nxdomain"}

            mock_verify.side_effect = verify_side_effect

            result = await layer2_advanced.layer2_check_with_verification(
                bot_response="Visit https://example.com and https://fake.com",
                user_message="Where?",
                urls=["https://example.com", "https://fake.com"],
                domains=["example.com", "fake.com"],
                github_repos=[],
                llm_call_func=mock_llm_call_func,
                llm_task_manager=mock_llm_task_manager,
                config=mock_config,
            )

            # Should have verification results for both domains
            assert "example.com" in result["verification_results"]
            assert "fake.com" in result["verification_results"]

    @pytest.mark.asyncio
    async def test_layer2_check_case_insensitive_yes(self):
        """Test Layer 2 response parsing (case insensitive)."""
        mock_llm_task_manager = MagicMock()
        mock_llm_call_func = AsyncMock()
        mock_config = MagicMock()

        # Mock LLM response with different cases
        test_cases = ["YES", "Yes", "YeS"]
        for response_text in test_cases:
            mock_response = MagicMock()
            mock_response.content = response_text
            mock_llm_call_func.return_value = mock_response

            with patch(
                "nemoguardrails.library.domain_hallucination.layer2_advanced.verify_domain_network"
            ) as mock_verify:
                mock_verify.return_value = {"domain": "test.com", "status": "resolved"}

                result = await layer2_advanced.layer2_check_with_verification(
                    bot_response="Test",
                    user_message="Test",
                    urls=[],
                    domains=["test.com"],
                    github_repos=[],
                    llm_call_func=mock_llm_call_func,
                    llm_task_manager=mock_llm_task_manager,
                    config=mock_config,
                )

                # All should be detected as "yes"
                assert result["is_hallucinated"] is True

    @pytest.mark.asyncio
    async def test_layer2_check_with_github_repos(self):
        """Test Layer 2 with GitHub repositories."""
        mock_llm_task_manager = MagicMock()
        mock_llm_call_func = AsyncMock()
        mock_config = MagicMock()

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "no"
        mock_llm_call_func.return_value = mock_response

        with patch(
            "nemoguardrails.library.domain_hallucination.layer2_advanced.verify_domain_network"
        ) as mock_verify:
            mock_verify.return_value = {"domain": "github.com", "status": "resolved"}

            result = await layer2_advanced.layer2_check_with_verification(
                bot_response="Check pytorch/pytorch repo",
                user_message="What's pytorch?",
                urls=[],
                domains=["github.com"],
                github_repos=["pytorch/pytorch"],
                llm_call_func=mock_llm_call_func,
                llm_task_manager=mock_llm_task_manager,
                config=mock_config,
            )

            assert result["is_hallucinated"] is False

    @pytest.mark.asyncio
    async def test_layer2_check_render_task_prompt(self):
        """Test that render_task_prompt is called with correct context."""
        mock_llm_task_manager = MagicMock()
        mock_llm_task_manager.render_task_prompt = MagicMock(return_value="test prompt")
        mock_llm_task_manager.get_stop_tokens = MagicMock(return_value=None)
        mock_llm_task_manager.get_max_tokens = MagicMock(return_value=1024)
        mock_llm_call_func = AsyncMock()

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "no"
        mock_llm_call_func.return_value = mock_response
        mock_config = MagicMock()
        mock_config.model = "gpt-3.5"
        mock_config.lowest_temperature = 0.1

        with patch(
            "nemoguardrails.library.domain_hallucination.layer2_advanced.verify_domain_network"
        ) as mock_verify:
            mock_verify.return_value = {"domain": "example.com", "status": "resolved"}

            await layer2_advanced.layer2_check_with_verification(
                bot_response="Visit example.com",
                user_message="Where?",
                urls=["https://example.com"],
                domains=["example.com"],
                github_repos=[],
                llm_call_func=mock_llm_call_func,
                llm_task_manager=mock_llm_task_manager,
                config=mock_config,
            )

            # Verify render_task_prompt was called with verification_results as JSON
            assert mock_llm_task_manager.render_task_prompt.called
            call_args = mock_llm_task_manager.render_task_prompt.call_args
            context = call_args.kwargs["context"]

            # Verification results should be JSON string, not a raw dict
            verification_results_str = context["verification_results"]
            # Should be parseable as JSON
            parsed = json.loads(verification_results_str)
            assert "example.com" in parsed


class TestLayer2Async:
    """Test Layer 2 async behavior."""

    @pytest.mark.asyncio
    async def test_verify_domain_network_async_executor(self):
        """Test that verify_domain_network uses async executor to prevent blocking."""
        call_count = 0

        def slow_getaddrinfo(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Simulate a slow DNS lookup
            import time
            time.sleep(0.01)  # 10ms delay
            return []

        with patch("socket.getaddrinfo", side_effect=slow_getaddrinfo):
            # Should not block the event loop (uses executor)
            result = await layer2_advanced.verify_domain_network("example.com")
            assert result["domain"] == "example.com"
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_layer2_check_concurrent_domains(self):
        """Test Layer 2 can verify multiple domains (potential concurrency)."""
        mock_llm_task_manager = MagicMock()
        mock_llm_call_func = AsyncMock()
        mock_config = MagicMock()

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "no"
        mock_llm_call_func.return_value = mock_response

        domains_verified = []

        async def track_verify(domain, timeout=5.0):
            domains_verified.append(domain)
            return {"domain": domain, "status": "resolved"}

        with patch(
            "nemoguardrails.library.domain_hallucination.layer2_advanced.verify_domain_network",
            side_effect=track_verify,
        ):
            await layer2_advanced.layer2_check_with_verification(
                bot_response="Multi domain check",
                user_message="Test",
                urls=[],
                domains=["a.com", "b.com", "c.com"],
                github_repos=[],
                llm_call_func=mock_llm_call_func,
                llm_task_manager=mock_llm_task_manager,
                config=mock_config,
            )

            # All domains should be verified
            assert "a.com" in domains_verified
            assert "b.com" in domains_verified
            assert "c.com" in domains_verified

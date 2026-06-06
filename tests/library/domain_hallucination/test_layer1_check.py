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

"""Unit tests for Layer 1 (fast LLM-based) domain hallucination detection."""

from unittest.mock import MagicMock

import pytest

from nemoguardrails.library.domain_hallucination import layer1_check


class TestEntityExtraction:
    """Test zero-dependency entity extraction (覆盖 extract_urls, extract_domains, extract_github_repos)."""

    def test_extract_urls_https(self):
        """Extract HTTPS URLs."""
        text = "Check https://github.com/pytorch/pytorch for details."
        urls = layer1_check.extract_urls(text)
        assert len(urls) == 1
        assert urls[0] == "https://github.com/pytorch/pytorch"

    def test_extract_urls_http(self):
        """Extract HTTP URLs."""
        text = "Visit http://example.com for more."
        urls = layer1_check.extract_urls(text)
        assert len(urls) == 1
        assert urls[0] == "http://example.com"

    def test_extract_urls_www(self):
        """Extract www URLs and normalize to https."""
        text = "See www.pytorch.org"
        urls = layer1_check.extract_urls(text)
        assert len(urls) == 1
        assert urls[0] == "https://www.pytorch.org"

    def test_extract_urls_deduplicate(self):
        """Deduplicate repeated URLs."""
        text = "Visit https://github.com and https://github.com again."
        urls = layer1_check.extract_urls(text)
        assert len(urls) == 1
        assert urls[0] == "https://github.com"

    def test_extract_urls_multiple(self):
        """Extract multiple distinct URLs."""
        text = "Check https://github.com and https://pytorch.org"
        urls = layer1_check.extract_urls(text)
        assert len(urls) == 2
        assert "https://github.com" in urls
        assert "https://pytorch.org" in urls

    def test_extract_urls_empty(self):
        """Handle text with no URLs."""
        text = "This is plain text with no links."
        urls = layer1_check.extract_urls(text)
        assert len(urls) == 0

    def test_extract_urls_with_trailing_punctuation(self):
        """Extract URLs with trailing punctuation (should be cleaned)."""
        text = "See https://github.com."
        urls = layer1_check.extract_urls(text)
        assert len(urls) == 1
        assert urls[0] == "https://github.com"

    def test_extract_urls_with_parenthesis(self):
        """Extract URLs with parenthesis context."""
        text = "(check https://pytorch.org)"
        urls = layer1_check.extract_urls(text)
        assert len(urls) == 1
        assert urls[0] == "https://pytorch.org"

    def test_extract_domains_simple(self):
        """Extract domains from URLs."""
        urls = ["https://github.com/pytorch/pytorch"]
        domains = layer1_check.extract_domains(urls)
        assert len(domains) == 1
        assert domains[0] == "github.com"

    def test_extract_domains_remove_www(self):
        """Remove www prefix from domains."""
        urls = ["https://www.pytorch.org"]
        domains = layer1_check.extract_domains(urls)
        assert len(domains) == 1
        assert domains[0] == "pytorch.org"

    def test_extract_domains_deduplicate(self):
        """Deduplicate domains from multiple URLs."""
        urls = ["https://github.com/pytorch/pytorch", "https://github.com/tensorflow/tensorflow"]
        domains = layer1_check.extract_domains(urls)
        assert len(domains) == 1
        assert domains[0] == "github.com"

    def test_extract_domains_case_insensitive(self):
        """Handle case-insensitive domain names."""
        urls = ["https://GitHub.com", "https://GITHUB.COM"]
        domains = layer1_check.extract_domains(urls)
        assert len(domains) == 1
        assert domains[0] == "github.com"

    def test_extract_github_repos_valid(self):
        """Extract valid GitHub repositories."""
        urls = ["https://github.com/pytorch/pytorch"]
        repos = layer1_check.extract_github_repos(urls)
        assert len(repos) == 1
        assert repos[0] == "pytorch/pytorch"

    def test_extract_github_repos_with_git_suffix(self):
        """Extract GitHub repos with .git suffix."""
        urls = ["https://github.com/pytorch/pytorch.git"]
        repos = layer1_check.extract_github_repos(urls)
        assert len(repos) == 1
        assert repos[0] == "pytorch/pytorch"

    def test_extract_github_repos_reserved_owner(self):
        """Skip reserved GitHub path segments as owners."""
        urls = ["https://github.com/features/something"]
        repos = layer1_check.extract_github_repos(urls)
        assert len(repos) == 0  # "features" is reserved

    def test_extract_github_repos_reserved_settings(self):
        """Skip 'settings' reserved path."""
        urls = ["https://github.com/settings/profile"]
        repos = layer1_check.extract_github_repos(urls)
        assert len(repos) == 0

    def test_extract_github_repos_deduplicate(self):
        """Deduplicate GitHub repositories."""
        urls = ["https://github.com/pytorch/pytorch", "https://github.com/pytorch/pytorch.git"]
        repos = layer1_check.extract_github_repos(urls)
        assert len(repos) == 1
        assert repos[0] == "pytorch/pytorch"

    def test_extract_github_repos_non_github_urls(self):
        """Ignore non-GitHub URLs."""
        urls = ["https://pytorch.org/docs"]
        repos = layer1_check.extract_github_repos(urls)
        assert len(repos) == 0

    def test_extract_github_repos_www_github(self):
        """Extract from www.github.com URLs."""
        urls = ["https://www.github.com/pytorch/pytorch"]
        repos = layer1_check.extract_github_repos(urls)
        assert len(repos) == 1
        assert repos[0] == "pytorch/pytorch"

    def test_extract_github_repos_incomplete_path(self):
        """Skip incomplete GitHub paths."""
        urls = ["https://github.com/pytorch"]
        repos = layer1_check.extract_github_repos(urls)
        assert len(repos) == 0

    def test_extract_entities_comprehensive(self):
        """Extract all entity types from mixed text."""
        text = "Check pytorch.org and https://github.com/pytorch/pytorch for info."
        entities = layer1_check.extract_entities(text)

        assert len(entities["urls"]) >= 1
        assert len(entities["domains"]) >= 1
        assert len(entities["github_repos"]) >= 1
        assert entities["has_entities"] is True

    def test_extract_entities_empty(self):
        """Handle text with no external entities."""
        text = "This is plain text about machine learning in general."
        entities = layer1_check.extract_entities(text)

        assert len(entities["urls"]) == 0
        assert len(entities["domains"]) == 0
        assert len(entities["github_repos"]) == 0
        assert entities["has_entities"] is False

    def test_extract_entities_mixed_cases(self):
        """Extract entities with mixed case domains."""
        text = "See https://github.com/PyTorch/PyTorch"
        entities = layer1_check.extract_entities(text)
        # Should normalize to lowercase
        assert entities["has_entities"] is True
        if entities["domains"]:
            assert "github.com" in entities["domains"]


class TestLayer1Integration:
    """Integration tests for Layer 1 functionality."""

    @pytest.mark.asyncio
    async def test_layer1_no_entities_fast_pass(self):
        """Layer 1 should pass through when no entities found."""
        result = await layer1_check.layer1_check_domain_hallucination(
            bot_response="Machine learning is a branch of AI.",
            user_message="What is machine learning?",
            llm_call_func=None,
            llm_task_manager=None,
            config=None,
        )

        assert result["layer"] == "layer1"
        assert result["is_hallucinated"] is False
        assert result["status"] == "clean"
        assert result["entities"]["has_entities"] is False

    @pytest.mark.asyncio
    async def test_layer1_with_llm_no_hallucination(self):
        """Test Layer 1 with LLM call - no hallucination found."""
        from unittest.mock import MagicMock

        # Mock LLM response
        mock_llm_response = MagicMock()
        mock_llm_response.content = "no"

        async def mock_llm_call(*args, **kwargs):
            return mock_llm_response

        # Mock task manager
        mock_task_manager = MagicMock()
        mock_task_manager.render_task_prompt.return_value = "test prompt"
        mock_task_manager.get_stop_tokens.return_value = None
        mock_task_manager.get_max_tokens.return_value = 1024

        # Mock config
        mock_config = MagicMock()
        mock_config.model = MagicMock()
        mock_config.lowest_temperature = 0.1

        result = await layer1_check.layer1_check_domain_hallucination(
            bot_response="Check https://pytorch.org",
            user_message="Tell me about PyTorch",
            llm_call_func=mock_llm_call,
            llm_task_manager=mock_task_manager,
            config=mock_config,
        )

        assert result["is_hallucinated"] is False
        assert result["status"] == "clean"
        assert result["llm_response"] == "no"

    @pytest.mark.asyncio
    async def test_layer1_with_llm_hallucination_detected(self):
        """Test Layer 1 with LLM call - hallucination detected."""
        from unittest.mock import MagicMock

        # Mock LLM response indicating hallucination
        mock_llm_response = MagicMock()
        mock_llm_response.content = "yes"

        async def mock_llm_call(*args, **kwargs):
            return mock_llm_response

        # Mock task manager
        mock_task_manager = MagicMock()
        mock_task_manager.render_task_prompt.return_value = "test prompt"
        mock_task_manager.get_stop_tokens.return_value = None
        mock_task_manager.get_max_tokens.return_value = 1024

        # Mock config
        mock_config = MagicMock()
        mock_config.model = MagicMock()
        mock_config.lowest_temperature = 0.1

        result = await layer1_check.layer1_check_domain_hallucination(
            bot_response="Check https://fake-pytorch.com",
            user_message="Tell me about PyTorch",
            llm_call_func=mock_llm_call,
            llm_task_manager=mock_task_manager,
            config=mock_config,
        )

        assert result["is_hallucinated"] is True
        assert result["status"] == "suspicious"
        assert result["llm_response"] == "yes"

    @pytest.mark.asyncio
    async def test_layer1_llm_call_error(self):
        """Test Layer 1 with LLM call error - should default to safe."""

        async def mock_llm_call_error(*args, **kwargs):
            raise ValueError("LLM call failed")

        # Mock task manager
        mock_task_manager = MagicMock()
        mock_task_manager.render_task_prompt.return_value = "test prompt"
        mock_task_manager.get_stop_tokens.return_value = None
        mock_task_manager.get_max_tokens.return_value = 1024

        # Mock config
        mock_config = MagicMock()
        mock_config.model = MagicMock()
        mock_config.lowest_temperature = 0.1

        result = await layer1_check.layer1_check_domain_hallucination(
            bot_response="Check https://pytorch.org",
            user_message="Tell me about PyTorch",
            llm_call_func=mock_llm_call_error,
            llm_task_manager=mock_task_manager,
            config=mock_config,
        )

        # Should default to safe (no hallucination) on error
        assert result["is_hallucinated"] is False
        assert result["status"] == "error"
        assert result["llm_response"] == "error"

    def test_layer1_entity_extraction(self):
        """Verify entity extraction in layer1 context."""
        bot_response = "See https://github.com/pytorch/pytorch for PyTorch source."
        entities = layer1_check.extract_entities(bot_response)

        assert entities["has_entities"] is True
        assert "https://github.com/pytorch/pytorch" in entities["urls"]
        assert "pytorch/pytorch" in entities["github_repos"]

    def test_layer1_multi_entity_extraction(self):
        """Extract multiple entities from complex text."""
        bot_response = """
        Check PyTorch at https://pytorch.org and tensorflow.org.
        Also see https://github.com/pytorch/pytorch and https://github.com/tensorflow/tensorflow
        for more details.
        """
        entities = layer1_check.extract_entities(bot_response)

        assert len(entities["urls"]) >= 3
        assert len(entities["domains"]) >= 2
        assert len(entities["github_repos"]) >= 2
        assert entities["has_entities"] is True

    def test_entity_extraction_no_false_positives(self):
        """Ensure no false positives in entity extraction."""
        text = "The variable www is used here. http means HyperText Transfer Protocol."
        entities = layer1_check.extract_entities(text)
        # Should not extract "www" or "http" as URLs
        assert entities["has_entities"] is False

    @pytest.mark.asyncio
    async def test_layer1_result_structure(self):
        """Verify Layer 1 result structure."""
        result = await layer1_check.layer1_check_domain_hallucination(
            bot_response="No links here.",
            user_message="Tell me about AI.",
            llm_call_func=None,
            llm_task_manager=None,
            config=None,
        )

        # Verify all required fields
        assert "layer" in result
        assert "entities" in result
        assert "llm_response" in result
        assert "is_hallucinated" in result
        assert "status" in result

        # Verify nested entities structure
        assert "urls" in result["entities"]
        assert "domains" in result["entities"]
        assert "github_repos" in result["entities"]
        assert "has_entities" in result["entities"]


class TestURLCleaning:
    """Test URL cleaning and normalization."""

    def test_clean_function(self):
        """Test _clean helper function."""
        # Test with trailing punctuation
        assert layer1_check._clean("https://github.com.") == "https://github.com"
        assert layer1_check._clean("https://github.com,") == "https://github.com"

        # Test with quotes
        assert layer1_check._clean('"https://github.com"') == "https://github.com"
        assert layer1_check._clean("'https://github.com'") == "https://github.com"

    def test_extract_urls_complex_punctuation(self):
        """Extract URLs surrounded by complex punctuation."""
        text = "Visit (https://pytorch.org) for details!"
        urls = layer1_check.extract_urls(text)
        assert len(urls) == 1
        assert urls[0] == "https://pytorch.org"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extract_urls_empty_string(self):
        """Handle empty strings."""
        assert layer1_check.extract_urls("") == []
        assert layer1_check.extract_domains([]) == []
        assert layer1_check.extract_github_repos([]) == []

    def test_extract_urls_only_whitespace(self):
        """Handle strings with only whitespace."""
        assert layer1_check.extract_urls("   \n\t  ") == []

    def test_extract_github_repos_multiple_slashes(self):
        """Handle GitHub URLs with paths."""
        urls = ["https://github.com/pytorch/pytorch/issues/123"]
        repos = layer1_check.extract_github_repos(urls)
        assert len(repos) == 1
        assert repos[0] == "pytorch/pytorch"

    def test_extract_urls_special_characters(self):
        """Handle URLs with special characters in query strings."""
        text = "Check https://pytorch.org/docs?lang=en&version=1.0"
        urls = layer1_check.extract_urls(text)
        assert len(urls) == 1
        assert "pytorch.org" in urls[0]

    def test_entities_with_newlines(self):
        """Handle entity extraction across newlines."""
        text = """First URL: https://github.com/pytorch/pytorch
        Second URL: https://pytorch.org
        And domain: www.tensorflow.org"""
        entities = layer1_check.extract_entities(text)
        assert entities["has_entities"] is True
        assert len(entities["urls"]) >= 2


class TestDomainNormalization:
    """Test domain normalization."""

    def test_domain_lowercase_normalization(self):
        """Domains should be normalized to lowercase."""
        urls = ["https://GitHub.COM/PyTorch/PyTorch"]
        domains = layer1_check.extract_domains(urls)
        assert domains[0] == "github.com"

    def test_domain_www_removal(self):
        """Verify www prefix removal."""
        urls = [
            "https://www.pytorch.org",
            "https://WWW.PYTORCH.ORG",
        ]
        domains = layer1_check.extract_domains(urls)
        # Should have 1 domain (deduplicated)
        assert len(domains) == 1
        assert domains[0] == "pytorch.org"

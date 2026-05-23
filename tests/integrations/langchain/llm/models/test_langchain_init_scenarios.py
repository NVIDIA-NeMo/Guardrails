# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Comprehensive tests for chat-model initialization scenarios.

This module tests all possible paths through the initialization chain:

    INITIALIZATION ORDER:
    ┌─────────────────────────────────────────────────────────────────┐
    │  #1  _handle_model_special_cases     (nvidia / nim)             │
    │  #2  _init_chat_completion_model     (langchain init_chat_model)│
    │  #3  _init_community_chat_models     (community chat models)    │
    └─────────────────────────────────────────────────────────────────┘

EXCEPTION PRIORITY RULES:
    1. ImportError (first one seen) - helps users know which package to install
    2. Last exception (if no ImportError) - later initializers are more specific
    3. Generic error (if no exceptions) - "Failed to initialize model..."

OUTCOME TYPES:
    Success  - Returns valid model, chain stops
    None     - Returns None, chain continues
    Error    - Raises exception, caught & stored, chain continues
"""

from dataclasses import dataclass
from typing import Callable, Optional, Type
from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails.integrations.langchain.langchain_initializer import (
    _PROVIDER_INITIALIZERS,
    ModelInitializationError,
    init_langchain_model,
)
from nemoguardrails.integrations.langchain.providers.providers import (
    _chat_providers,
    register_chat_provider,
)


@dataclass
class MockProvider:
    """Factory for creating mock provider classes with configurable behavior."""

    behavior: str
    error_type: Optional[Type[Exception]] = None
    error_msg: str = ""

    def create_class(self):
        """
        Create a provider class with the specified behavior.

        Behaviors:
        - "success": Provider initializes successfully
        - "error": Provider raises error_type with error_msg during __init__
        """
        behavior = self.behavior
        error_type = self.error_type
        error_msg = self.error_msg

        class _Provider:
            model_fields = {"model": None}

            def __init__(self, **kwargs):
                if behavior == "success":
                    self.model = kwargs.get("model")
                elif behavior == "error":
                    raise error_type(error_msg)

        return _Provider


class ProviderRegistry:
    """
    Helper to register providers and automatically clean them up after tests.

    Usage:
        with registry fixture:
            registry.register_chat("name", provider_class)
            # provider is automatically removed after test
    """

    def __init__(self):
        self._originals = {}

    def register_chat(self, name: str, provider_cls):
        self._originals[("chat", name)] = _chat_providers.get(name)
        if provider_cls is None:
            _chat_providers[name] = None
        else:
            register_chat_provider(name, provider_cls)

    def register_special_provider(self, provider_name: str, handler: Callable):
        self._originals[("special", provider_name)] = _PROVIDER_INITIALIZERS.get(provider_name)
        _PROVIDER_INITIALIZERS[provider_name] = handler

    def cleanup(self):
        for (ptype, name), original in self._originals.items():
            registry = {
                "chat": _chat_providers,
                "special": _PROVIDER_INITIALIZERS,
            }[ptype]

            if original is not None:
                registry[name] = original
            elif name in registry:
                del registry[name]


@pytest.fixture
def registry():
    reg = ProviderRegistry()
    yield reg
    reg.cleanup()


class TestSuccessScenarios:
    """Tests where initialization succeeds at various points in the chain."""

    def test_chat_completion_success(self, registry):
        """The langchain ``init_chat_model`` path returns a model."""
        mock_model = MagicMock()
        with patch(
            "nemoguardrails.integrations.langchain.langchain_initializer.init_chat_model", return_value=mock_model
        ):
            result = init_langchain_model("test-model", "openai", {})
            assert result == mock_model

    def test_community_chat_success(self, registry):
        """A community chat provider initializes successfully."""
        registry.register_chat("_test_community", MockProvider("success").create_class())
        with patch("nemoguardrails.integrations.langchain.langchain_initializer.init_chat_model", return_value=None):
            result = init_langchain_model("test-model", "_test_community", {})
            assert result is not None


class TestSingleErrorScenarios:
    """Tests where exactly one initializer raises an error."""

    def test_chat_error_preserved(self):
        """When the chat-completion initializer fails, the error is preserved."""
        with patch(
            "nemoguardrails.integrations.langchain.langchain_initializer.init_chat_model",
            side_effect=ValueError("Invalid API key"),
        ):
            with pytest.raises(ModelInitializationError) as exc_info:
                init_langchain_model("test-model", "_test_err", {})
            assert "Invalid API key" in str(exc_info.value)

    def test_community_error_preserved(self, registry):
        """When the community initializer fails, the error is preserved."""
        registry.register_chat("_test_err", MockProvider("error", ValueError, "Rate limit exceeded").create_class())
        with patch("nemoguardrails.integrations.langchain.langchain_initializer.init_chat_model", return_value=None):
            with pytest.raises(ModelInitializationError) as exc_info:
                init_langchain_model("test-model", "_test_err", {})
            assert "Rate limit exceeded" in str(exc_info.value)


class TestMultipleErrorPriority:
    """Tests for exception priority when multiple initializers fail."""

    @pytest.mark.parametrize(
        ("chat_exc", "community_exc", "expected_winner"),
        [
            ((ValueError, "Error A"), (ValueError, "Error B"), "Error B"),
            ((RuntimeError, "Error A"), (ValueError, "Error B"), "Error B"),
            ((ImportError, "Import A"), (ValueError, "Error B"), "Import A"),
            ((ValueError, "Error A"), (ImportError, "Import B"), "Import B"),
            ((ImportError, "Import A"), (ImportError, "Import B"), "Import A"),
        ],
    )
    def test_exception_priority(self, registry, chat_exc, community_exc, expected_winner):
        """Verify exception priority rules are correctly applied."""
        provider = "_test_priority"
        registry.register_chat(provider, MockProvider("error", community_exc[0], community_exc[1]).create_class())

        with patch(
            "nemoguardrails.integrations.langchain.langchain_initializer.init_chat_model",
            side_effect=chat_exc[0](chat_exc[1]),
        ):
            with pytest.raises(ModelInitializationError) as exc_info:
                init_langchain_model("test-model", provider, {})

        assert expected_winner in str(exc_info.value)


class TestErrorRecovery:
    """Tests where early errors are recovered by later successful initialization."""

    def test_chat_fails_community_succeeds(self, registry):
        """Chat-completion fails but a community chat provider succeeds."""
        provider = "_test_recovery"
        registry.register_chat(provider, MockProvider("success").create_class())

        with patch(
            "nemoguardrails.integrations.langchain.langchain_initializer.init_chat_model",
            side_effect=ValueError("Chat failed"),
        ):
            result = init_langchain_model("test-recovery-model", provider, {})
            assert result is not None


class TestSpecialCaseHandling:
    """Tests for the provider-specific special case handlers."""

    def test_nvidia_provider_import_error(self, registry):
        """NVIDIA provider surfaces ImportError when package missing."""

        def nvidia_import_error(*args, **kwargs):
            raise ImportError("langchain_nvidia_ai_endpoints not installed")

        registry.register_special_provider("nvidia_ai_endpoints", nvidia_import_error)

        with pytest.raises(ModelInitializationError) as exc_info:
            init_langchain_model("test-model", "nvidia_ai_endpoints", {})

        assert "langchain_nvidia_ai_endpoints" in str(exc_info.value)


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_none_provider_handled(self, registry):
        """Provider registered as None doesn't crash."""
        provider = "_test_none"
        registry.register_chat(provider, None)
        _chat_providers[provider] = None

        with pytest.raises((ModelInitializationError, TypeError, AttributeError)):
            init_langchain_model("test-model", provider, {})

    def test_empty_model_name_rejected(self):
        """Empty model name raises clear error."""
        with pytest.raises(ModelInitializationError, match="Model name is required"):
            init_langchain_model("", "openai", {})

    def test_all_return_none_generic_error(self):
        """When all initializers return None, generic error is raised."""
        with patch("nemoguardrails.integrations.langchain.langchain_initializer.init_chat_model", return_value=None):
            with pytest.raises(ModelInitializationError) as exc_info:
                init_langchain_model("test-model", "nonexistent_xyz", {})

            error_msg = str(exc_info.value)
            assert "Failed to initialize model" in error_msg
            assert "nonexistent_xyz" in error_msg


class TestE2EIntegration:
    """End-to-end tests through RailsConfig/LLMRails."""

    def test_e2e_meaningful_error_from_config(self, registry):
        """When a provider is found but initialization fails, the error should bubble up."""
        from nemoguardrails import LLMRails, RailsConfig

        provider = "_e2e_test"
        registry.register_chat(provider, MockProvider("error", ValueError, "Invalid API key: sk-xxx").create_class())

        config = RailsConfig.from_content(
            config={"models": [{"type": "main", "engine": provider, "model": "test-model"}]}
        )

        with patch("nemoguardrails.integrations.langchain.langchain_initializer.init_chat_model", return_value=None):
            with pytest.raises(ModelInitializationError) as exc_info:
                LLMRails(config=config)

        assert "Invalid API key" in str(exc_info.value)

    def test_e2e_successful_initialization(self, registry):
        """When the provider initializes, the full stack should complete without errors."""
        from nemoguardrails import LLMRails, RailsConfig

        provider = "_e2e_success"
        registry.register_chat(provider, MockProvider("success").create_class())

        config = RailsConfig.from_content(
            config={"models": [{"type": "main", "engine": provider, "model": "test-model"}]}
        )

        with patch("nemoguardrails.integrations.langchain.langchain_initializer.init_chat_model", return_value=None):
            rails = LLMRails(config=config)
        assert rails.llm is not None

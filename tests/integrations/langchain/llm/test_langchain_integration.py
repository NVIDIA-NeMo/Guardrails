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

import os
import warnings
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel

from nemoguardrails.integrations.langchain.langchain_initializer import init_langchain_model
from nemoguardrails.integrations.langchain.providers.providers import (
    _chat_providers,
    _discover_langchain_community_chat_providers,
)


class MockLangChainChatModel:
    def _call(self, *args, **kwargs):
        return "Mock LangChain Chat Model response"


@pytest.fixture
def mock_langchain_chat_models():
    with patch("nemoguardrails.integrations.langchain.providers.providers._module_lookup") as mock_lookup:
        # mock the items method to return a list of tuples
        mock_lookup.items.return_value = [
            (
                "MockLangChainChatModel",
                "langchain_community.chat_models.mock_provider",
            )
        ]
        with patch("nemoguardrails.integrations.langchain.providers.providers.importlib.import_module") as mock_import:
            # mock the import_module function
            mock_module = MagicMock()
            mock_module.MockLangChainChatModel = MockLangChainChatModel
            mock_import.return_value = mock_module
            yield mock_lookup


def test_discover_mocked_langchain_community_chat_providers(mock_langchain_chat_models):
    providers = _discover_langchain_community_chat_providers()
    assert "mock_provider" in providers
    assert providers["mock_provider"] == MockLangChainChatModel


def test_langchain_providers_in_registry():
    """Test that LangChain chat providers are included in the registry."""
    assert len(_chat_providers) > 0, "No chat providers found in the registry"


def test_langchain_provider_imports():
    """Test that LangChain chat providers can be imported without errors."""
    chat_provider_names = list(_chat_providers.keys())

    for provider_name in chat_provider_names:
        try:
            provider_cls = _chat_providers[provider_name]
            assert provider_cls is not None, f"Provider class for '{provider_name}' is None"
        except Exception as e:
            warnings.warn(f"Failed to import chat provider '{provider_name}': {str(e)}")


def _is_langchain_installed():
    """Check if LangChain is installed."""
    from nemoguardrails.imports import check_optional_dependency

    return check_optional_dependency("langchain")


def _is_langchain_community_installed():
    """Check if LangChain Community is installed."""
    from nemoguardrails.imports import check_optional_dependency

    return check_optional_dependency("langchain_community")


def _has_openai():
    """Check if OpenAI package is installed."""
    from nemoguardrails.imports import check_optional_dependency

    return check_optional_dependency("langchain_openai")


class TestLangChainIntegration:
    """Integration tests for LangChain model initialization."""

    @pytest.mark.skipif(not _is_langchain_installed(), reason="LangChain is not installed")
    def test_init_openai_chat_model(self):
        """Test initializing an OpenAI chat model with real implementation."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not set")

        model = init_langchain_model("gpt-3.5-turbo", "openai", {"temperature": 0.1})
        assert model is not None
        assert hasattr(model, "invoke")
        assert isinstance(model, BaseChatModel)

        # Test that the model can be used
        from langchain_core.messages import HumanMessage

        response = model.invoke([HumanMessage(content="Hello, world!")])
        assert response is not None
        assert hasattr(response, "content")

    @pytest.mark.skipif(
        not _is_langchain_installed() or not _has_openai(),
        reason="LangChain is not installed",
    )
    def test_init_with_kwargs(self):
        """Test initializing a model with additional kwargs."""

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not set")

        # Initialize with additional kwargs
        model = init_langchain_model(
            "gpt-4o",
            "openai",
            {
                "temperature": 0.1,
                "max_tokens": 100,
                "top_p": 0.9,
            },
        )
        assert model is not None
        assert hasattr(model, "invoke")

        from langchain_core.messages import HumanMessage

        response = model.invoke([HumanMessage(content="Hello, world!")])
        assert response is not None
        assert hasattr(response, "content")

    @pytest.mark.skipif(
        not _is_langchain_installed() or not _has_openai(),
        reason="LangChain is not installed",
    )
    def test_init_with_api_key_env_var_chat_completion_model(self):
        """Test initializing a chat model with api_key_env_var."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not set")

        original_api_key = os.environ["OPENAI_API_KEY"]
        custom_env_var = "NG_OPENAI_API_KEY"
        os.environ[custom_env_var] = original_api_key
        del os.environ["OPENAI_API_KEY"]

        try:
            model = init_langchain_model(
                "gpt-4o",
                "openai",
                {"api_key": os.environ.get(custom_env_var)},
            )
            assert model is not None
            assert hasattr(model, "invoke")
            assert isinstance(model, BaseChatModel)

            from langchain_core.messages import HumanMessage

            response = model.invoke([HumanMessage(content="Hello, world!")])
            assert response is not None
            assert hasattr(response, "content")
        finally:
            os.environ["OPENAI_API_KEY"] = original_api_key
            del os.environ[custom_env_var]

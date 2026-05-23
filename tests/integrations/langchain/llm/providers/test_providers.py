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

import warnings
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel

from nemoguardrails.integrations.langchain.providers.providers import (
    _chat_providers,
    _get_chat_completion_provider,
    _parse_version,
    get_community_chat_provider_names,
    register_chat_provider,
)


class MockChatModel(BaseChatModel):
    def _call(self, *args, **kwargs):
        return "Mock chat response"


@pytest.fixture
def mock_langchain_chat_models():
    with patch("nemoguardrails.integrations.langchain.providers.providers._module_lookup") as mock_lookup:
        mock_lookup.items.return_value = [("mock_provider", "langchain_community.chat_models.mock_provider")]
        with patch("nemoguardrails.integrations.langchain.providers.providers.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.mock_provider = MockChatModel
            mock_import.return_value = mock_module
            yield mock_lookup


@pytest.fixture
def test_chat_provider_fixture():
    register_chat_provider("test_chat_provider", MockChatModel)
    yield
    # unregister the provider after the test so that it doesn't affect other tests
    _chat_providers.pop("test_chat_provider", None)


def test_register_chat_provider(test_chat_provider_fixture):
    assert "test_chat_provider" in _chat_providers
    assert _chat_providers["test_chat_provider"] == MockChatModel


def test_get_chat_provider_names():
    provider_names = get_community_chat_provider_names()
    assert isinstance(provider_names, list)

    # check for common providers that should be available
    common_providers = ["openai", "anthropic", "huggingface"]
    for provider in common_providers:
        if provider in provider_names:
            pass
        else:
            warnings.warn(
                f"Common chat provider '{provider}' is not available. "
                "This might be due to a version mismatch with LangChain."
            )


def test_get_chat_completion_provider():
    # test with a registered provider
    with patch(
        "nemoguardrails.integrations.langchain.providers.providers._chat_providers",
        {"test_provider": MockChatModel},
    ):
        provider = _get_chat_completion_provider("test_provider")
        assert provider == MockChatModel

    # test with a non-existent provider
    with pytest.raises(RuntimeError):
        _get_chat_completion_provider("non_existent_provider")


def test_parse_version():
    assert _parse_version("1.2.3") == (1, 2, 3)
    assert _parse_version("0.1.0") == (0, 1, 0)
    assert _parse_version("10.20.30") == (10, 20, 30)

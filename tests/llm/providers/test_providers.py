# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import warnings
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel, BaseLLM

from nemoguardrails.llm.providers.providers import (
    _acall,
    _chat_providers,
    _get_chat_completion_provider,
    _get_text_completion_provider,
    _llm_providers,
    _parse_version,
    _patch_acall_method_to,
    get_community_chat_provider_names,
    get_llm_provider_names,
    register_chat_provider,
    register_llm_provider,
)


class MockLLM(BaseLLM):
    def _call(self, *args, **kwargs):
        return "Mock response"

    def _generate(self, *args, **kwargs):
        return "Mock generation"

    def _llm_type(self):
        return "mock_llm_type"


class MockChatModel(BaseChatModel):
    def _call(self, *args, **kwargs):
        return "Mock chat response"


@pytest.fixture
def mock_langchain_llms():
    with patch("nemoguardrails.llm.providers.providers.llms") as mock_llms:
        mock_dict = {"mock_provider": MockLLM}
        mock_llms.get_type_to_cls_dict.return_value = mock_dict
        mock_llms.type_to_cls_dict = mock_dict
        yield mock_llms


@pytest.fixture
def mock_langchain_chat_models():
    with patch("nemoguardrails.llm.providers.providers._module_lookup") as mock_lookup:
        mock_lookup.items.return_value = [("mock_provider", "langchain_community.chat_models.mock_provider")]
        with patch("nemoguardrails.llm.providers.providers.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.mock_provider = MockChatModel
            mock_import.return_value = mock_module
            yield mock_lookup


def test_patch_acall_method_to():
    # create a new class for testing to avoid affecting other tests
    class TestLLM(MockLLM):
        pass

    providers = {"mock_provider": TestLLM}

    # Mock asyncio.to_thread to return a future with the result
    with patch("asyncio.to_thread") as mock_to_thread:
        mock_to_thread.return_value = "Mock response"

        # Patch the _acall method
        _patch_acall_method_to(providers)

        assert hasattr(TestLLM, "_acall")
        assert callable(getattr(TestLLM, "_acall"))

        # test that the patched _acall method works correctly
        mock_llm = TestLLM()

        # create an event loop to run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(mock_llm._acall("test prompt"))
            assert result == "Mock response"
            mock_to_thread.assert_called_once()
        finally:
            loop.close()


@pytest.mark.asyncio
async def test_acall():
    mock_llm = MockLLM()
    result = await _acall(mock_llm, "test prompt")
    assert result == "Mock response"


@pytest.fixture
def test_llm_provider_fixture():
    _patch_acall_method_to({"test_llm_provider": MockLLM})
    register_llm_provider("test_llm_provider", MockLLM)
    yield

    # unregister the provider after the test so that it doesn't affect other tests
    _llm_providers.pop("test_llm_provider", None)


def test_register_llm_provider(test_llm_provider_fixture):
    assert "test_llm_provider" in _llm_providers
    assert _llm_providers["test_llm_provider"] == MockLLM


@pytest.fixture
def test_chat_provider_fixture():
    register_chat_provider("test_chat_provider", MockChatModel)
    yield
    # unregister the provider after the test so that it doesn't affect other tests
    _chat_providers.pop("test_chat_provider", None)


def test_register_chat_provider(test_chat_provider_fixture):
    assert "test_chat_provider" in _chat_providers
    assert _chat_providers["test_chat_provider"] == MockChatModel


def test_get_llm_provider_names():
    provider_names = get_llm_provider_names()
    assert isinstance(provider_names, list)

    # the default providers
    assert "trt_llm" in provider_names, "Default provider 'trt_llm' is not in the list of providers"

    common_providers = ["openai", "anthropic", "huggingface"]
    for provider in common_providers:
        if provider in provider_names:
            # provider is available, this is good
            pass
        else:
            # provider is not available, but we don't fail the test
            # instead, we issue a warning
            warnings.warn(
                f"Common LLM provider '{provider}' is not available. "
                "This might be due to a version mismatch with LangChain."
            )


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


def test_get_text_completion_provider():
    # test with a registered provider
    with patch(
        "nemoguardrails.llm.providers.providers._llm_providers",
        {"test_provider": MockLLM},
    ):
        provider = _get_text_completion_provider("test_provider")
        assert provider == MockLLM

    # test with a non-existent provider
    with pytest.raises(RuntimeError):
        _get_text_completion_provider("non_existent_provider")


def test_get_chat_completion_provider():
    # test with a registered provider
    with patch(
        "nemoguardrails.llm.providers.providers._chat_providers",
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

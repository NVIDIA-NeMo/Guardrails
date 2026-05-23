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
from importlib.metadata import PackageNotFoundError, version

import pytest

from nemoguardrails.integrations.langchain.providers.providers import (
    _chat_providers,
    _discover_langchain_community_chat_providers,
    _discover_langchain_partner_chat_providers,
    get_chat_provider_names,
    get_community_chat_provider_names,
)

# valid for 0.3.13 till 0.3.21
# previous 0.3 versions miss   -     'openllm_client',
# 0.2 versions have more and less
# name         : langchain-community
# version      : 0.3.16
# description  : Community contributed LangChain integrations.

_COMMUNITY_CHAT_PROVIDERS_NAMES = [
    "azure_openai",
    "bedrock",
    "anthropic",
    "anyscale",
    "baichuan",
    "naver",
    "cohere",
    "coze",
    "databricks",
    "deepinfra",
    "everlyai",
    "edenai",
    "fireworks",
    "friendli",
    "google_palm",
    "huggingface",
    "hunyuan",
    "javelin_ai_gateway",
    "kinetica",
    "konko",
    "litellm",
    "litellm_router",
    "mlflow_ai_gateway",
    "mlx",
    "maritalk",
    "mlflow",
    "symblai_nebula",
    "octoai",
    "oci_generative_ai",
    "oci_data_science",
    "ollama",
    "openai",
    "outlines",
    "reka",
    "perplexity",
    "sambanova",
    "snowflake",
    "sparkllm",
    "tongyi",
    "vertexai",
    "yandex",
    "yuan2",
    "zhipuai",
    "ernie",
    "fake",
    "gpt_router",
    "gigachat",
    "human",
    "jinachat",
    "llama_edge",
    "minimax",
    "moonshot",
    "pai_eas_endpoint",
    "promptlayer_openai",
    "solar",
    "baidu_qianfan_endpoint",
    "volcengine_maas",
    "premai",
    "llamacpp",
    "yi",
]

_PARTNER_CHAT_PROVIDERS_NAMES = {
    "anthropic",
    "azure_openai",
    "bedrock",
    "bedrock_converse",
    "cohere",
    "deepseek",
    "fireworks",
    "google_anthropic_vertex",
    "google_genai",
    "google_vertexai",
    "groq",
    "huggingface",
    "mistralai",
    "nim",
    "ollama",
    "openai",
    "together",
}

# at some point we might care about certain providers
CRITICAL_CHAT_PROVIDERS = [
    "openai",
    "anthropic",
]

# providers that have been renamed or moved in the past
RENAMED_PROVIDERS = {
    "mlflow-chat": "mlflow",
    "databricks-chat": "databricks",
}


def get_langchain_version():
    """Get the installed LangChain version."""
    try:
        return version("langchain")
    except PackageNotFoundError:
        try:
            return version("langchain-community")
        except PackageNotFoundError:
            return "unknown"


def test_critical_chat_providers_available():
    """Test that critical chat providers are available."""
    provider_names = get_community_chat_provider_names()

    for provider in CRITICAL_CHAT_PROVIDERS:
        if provider not in provider_names:
            warnings.warn(
                f"Critical chat provider '{provider}' is not available. "
                f"This might cause compatibility issues with LangChain version {get_langchain_version()}."
            )


def test_renamed_providers():
    """Test for providers that have been renamed or moved."""
    chat_provider_names = get_community_chat_provider_names()

    for old_name, new_name in RENAMED_PROVIDERS.items():
        if old_name in chat_provider_names:
            warnings.warn(
                f"Provider '{old_name}' has been renamed to '{new_name}' in newer versions of LangChain. "
                f"Consider updating your code to use the new name."
            )


def test_provider_registry_stability():
    """Test that the provider registry is stable and doesn't change unexpectedly."""
    current_chat_providers = set(get_community_chat_provider_names())
    expected_chat_providers = set(_chat_providers.keys())

    assert current_chat_providers == expected_chat_providers, (
        f"Chat provider registry has changed unexpectedly. "
        f"Expected: {expected_chat_providers}, Got: {current_chat_providers}"
    )


def test_provider_imports():
    """Test that all chat providers can be imported without errors."""
    chat_provider_names = get_community_chat_provider_names()

    for provider_name in chat_provider_names:
        try:
            provider_cls = _chat_providers[provider_name]
            assert provider_cls is not None, f"Provider class for '{provider_name}' is None"
        except Exception as e:
            pytest.fail(f"Failed to import chat provider '{provider_name}': {str(e)}")


def test_discover_langchain_community_chat_providers():
    """Test that the function correctly discovers LangChain community chat providers."""

    providers = _discover_langchain_community_chat_providers()
    chat_provider_names = get_community_chat_provider_names()
    assert set(chat_provider_names) == set(providers.keys()), (
        "it seems that we are registering a provider that is not in the LC community chat provider"
    )
    assert _COMMUNITY_CHAT_PROVIDERS_NAMES == list(providers.keys()), (
        "LangChain chat community providers may have changed. please investigate and update the test if necessary."
    )


def test_discover_partner_chat_providers_no_providers_attr(monkeypatch):
    """Test fallback when neither _BUILTIN_PROVIDERS nor _SUPPORTED_PROVIDERS exists."""
    import langchain.chat_models.base as _base

    monkeypatch.delattr(_base, "_BUILTIN_PROVIDERS", raising=False)
    monkeypatch.delattr(_base, "_SUPPORTED_PROVIDERS", raising=False)

    from nemoguardrails.integrations.langchain.providers.providers import _CUSTOM_CHAT_PROVIDERS

    result = _discover_langchain_partner_chat_providers()
    assert result == _CUSTOM_CHAT_PROVIDERS


def test_discover_partner_chat_providers_set_type(monkeypatch):
    """Test branch when _SUPPORTED_PROVIDERS is a set (older langchain versions)."""
    import langchain.chat_models.base as _base

    providers_set = {"openai", "anthropic"}
    monkeypatch.delattr(_base, "_BUILTIN_PROVIDERS", raising=False)
    monkeypatch.setattr(_base, "_SUPPORTED_PROVIDERS", providers_set, raising=False)

    from nemoguardrails.integrations.langchain.providers.providers import _CUSTOM_CHAT_PROVIDERS

    result = _discover_langchain_partner_chat_providers()
    assert result == providers_set | _CUSTOM_CHAT_PROVIDERS


def test_discover_partner_chat_providers_supported_dict(monkeypatch):
    """Test branch when _SUPPORTED_PROVIDERS is a dict (langchain ~1.2.1)."""
    import langchain.chat_models.base as _base

    providers_dict = {
        "openai": ("langchain_openai", "ChatOpenAI"),
        "anthropic": ("langchain_anthropic", "ChatAnthropic"),
    }
    monkeypatch.delattr(_base, "_BUILTIN_PROVIDERS", raising=False)
    monkeypatch.setattr(_base, "_SUPPORTED_PROVIDERS", providers_dict, raising=False)

    from nemoguardrails.integrations.langchain.providers.providers import _CUSTOM_CHAT_PROVIDERS

    result = _discover_langchain_partner_chat_providers()
    assert result == set(providers_dict.keys()) | _CUSTOM_CHAT_PROVIDERS


def test_discover_partner_chat_providers_builtin_set(monkeypatch):
    """Test branch when _BUILTIN_PROVIDERS is a set (hypothetical)."""
    import langchain.chat_models.base as _base

    providers_set = {"openai", "anthropic"}
    monkeypatch.setattr(_base, "_BUILTIN_PROVIDERS", providers_set)
    monkeypatch.delattr(_base, "_SUPPORTED_PROVIDERS", raising=False)

    from nemoguardrails.integrations.langchain.providers.providers import _CUSTOM_CHAT_PROVIDERS

    result = _discover_langchain_partner_chat_providers()
    assert result == providers_set | _CUSTOM_CHAT_PROVIDERS


def test_dicsover_partner_chat_providers():
    """Test that the function correctly discovers LangChain partner chat providers."""

    partner_chat_providers = _discover_langchain_partner_chat_providers()
    assert _PARTNER_CHAT_PROVIDERS_NAMES.issubset(partner_chat_providers), (
        "LangChain partner chat providers may have changed. Update "
        "_PARTNER_CHAT_PROVIDERS_NAMES to include all expected providers."
    )
    chat_providers = get_chat_provider_names()

    assert partner_chat_providers.issubset(chat_providers), (
        "partner chat providers are not a subset of the list of chat providers"
    )

    if not partner_chat_providers == _PARTNER_CHAT_PROVIDERS_NAMES:
        warnings.warn(
            "LangChain partner chat providers may have changed. Update "
            "_PARTNER_CHAT_PROVIDERS_NAMES to include all expected providers."
        )


def test_langchain_provider_compatibility():
    """Test compatibility with different LangChain versions."""
    common_chat_providers = ["openai", "anthropic", "huggingface"]

    # check for chat providers
    for provider in common_chat_providers:
        if provider not in _chat_providers:
            raise RuntimeError(
                f"Common chat provider '{provider}' is not available. "
                "This might be due to a version mismatch with LangChain."
            )

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

import warnings
from importlib.metadata import PackageNotFoundError, version

import pytest

from nemoguardrails.llm.providers.providers import (
    _chat_providers,
    _discover_langchain_community_chat_providers,
    _discover_langchain_community_llm_providers,
    _discover_langchain_partner_chat_providers,
    _llm_providers,
    get_chat_provider_names,
    get_community_chat_provider_names,
    get_llm_provider_names,
)

# valid for 0.3.13 till 0.3.21
# previous 0.3 versions miss   -     'openllm_client',
# 0.2 versions have more and less
# name         : langchain-community
# version      : 0.3.16
# description  : Community contributed LangChain integrations.

_LLM_PROVIDERS_NAMES = [
    "ai21",
    "aleph_alpha",
    "amazon_api_gateway",
    "amazon_bedrock",
    "anthropic",
    "anyscale",
    "arcee",
    "aviary",
    "azure",
    "azureml_endpoint",
    "baichuan",
    "bananadev",
    "baseten",
    "beam",
    "cerebriumai",
    "chat_glm",
    "clarifai",
    "cohere",
    "ctransformers",
    "ctranslate2",
    "databricks",
    "deepinfra",
    "deepsparse",
    "edenai",
    "fake-list",
    "forefrontai",
    "friendli",
    "giga-chat-model",
    "google_palm",
    "gooseai",
    "gradient",
    "gpt4all",
    "huggingface_endpoint",
    "huggingface_hub",
    "huggingface_pipeline",
    "huggingface_textgen_inference",
    "human-input",
    "koboldai",
    "konko",
    "llamacpp",
    "llamafile",
    "textgen",
    "minimax",
    "mlflow",
    "mlflow-ai-gateway",
    "mlx_pipeline",
    "modal",
    "mosaic",
    "nebula",
    "nibittensor",
    "nlpcloud",
    "oci_model_deployment_tgi_endpoint",
    "oci_model_deployment_vllm_endpoint",
    "oci_model_deployment_endpoint",
    "oci_generative_ai",
    "octoai_endpoint",
    "ollama",
    "openai",
    "openlm",
    "pai_eas_endpoint",
    "petals",
    "pipelineai",
    "predibase",
    "opaqueprompts",
    "replicate",
    "rwkv",
    "sagemaker_endpoint",
    "sambanovacloud",
    "sambastudio",
    "self_hosted",
    "self_hosted_hugging_face",
    "stochasticai",
    "together",
    "tongyi",
    "titan_takeoff",
    "titan_takeoff_pro",
    "vertexai",
    "vertexai_model_garden",
    "openllm",
    "outlines",
    "vllm",
    "vllm_openai",
    "watsonxllm",
    "weight_only_quantization",
    "writer",
    "xinference",
    "javelin-ai-gateway",
    "qianfan_endpoint",
    "yandex_gpt",
    "yuan2",
    "VolcEngineMaasLLM",
    "SparkLLM",
    "yi",
    "you",
]
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
CRITICAL_LLM_PROVIDERS = [
    "openai",
    "anthropic",
]

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


def test_critical_llm_providers_available():
    """Test that critical LLM providers are available."""
    provider_names = get_llm_provider_names()

    # ensure we have critical providers
    for provider in CRITICAL_LLM_PROVIDERS:
        if provider not in provider_names:
            warnings.warn(
                f"Critical LLM provider '{provider}' is not available. "
                f"This might cause compatibility issues with LangChain version {get_langchain_version()}."
            )


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
    llm_provider_names = get_llm_provider_names()
    chat_provider_names = get_community_chat_provider_names()

    for old_name, new_name in RENAMED_PROVIDERS.items():
        if old_name in llm_provider_names or old_name in chat_provider_names:
            warnings.warn(
                f"Provider '{old_name}' has been renamed to '{new_name}' in newer versions of LangChain. "
                f"Consider updating your code to use the new name."
            )


def test_provider_registry_stability():
    """Test that the provider registry is stable and doesn't change unexpectedly."""
    # Get the current providers
    current_llm_providers = set(get_llm_provider_names())
    current_chat_providers = set(get_community_chat_provider_names())

    # This test will fail if the registry changes unexpectedly
    expected_llm_providers = set(_llm_providers.keys())
    expected_chat_providers = set(_chat_providers.keys())

    assert current_llm_providers == expected_llm_providers, (
        f"LLM provider registry has changed unexpectedly. "
        f"Expected: {expected_llm_providers}, Got: {current_llm_providers}"
    )

    assert current_chat_providers == expected_chat_providers, (
        f"Chat provider registry has changed unexpectedly. "
        f"Expected: {expected_chat_providers}, Got: {current_chat_providers}"
    )


def test_provider_imports():
    """Test that all providers can be imported without errors."""
    # This test ensures that all providers can be imported without errors
    # It's useful for catching import errors early

    # get all provider names
    llm_provider_names = get_llm_provider_names()
    chat_provider_names = get_community_chat_provider_names()

    # try to import each provider
    for provider_name in llm_provider_names:
        try:
            provider_cls = _llm_providers[provider_name]
            assert provider_cls is not None, f"Provider class for '{provider_name}' is None"
        except Exception as e:
            pytest.fail(f"Failed to import LLM provider '{provider_name}': {str(e)}")

    for provider_name in chat_provider_names:
        try:
            # This is a simplified example - you might need to adjust this
            # based on how your providers are actually imported
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


def test_discover_langchain_community_llm_providers():
    providers = _discover_langchain_community_llm_providers()
    llm_provider_names = get_llm_provider_names()

    custom_registered_providers = {"trt_llm"}
    assert set(llm_provider_names) - custom_registered_providers == set(providers.keys()), (
        "it seems that we are registering a provider that is not in the LC community llm provider"
    )
    assert _LLM_PROVIDERS_NAMES == list(providers.keys()), (
        "LangChain LLM community providers may have changed. Please investigate and update the test if necessary."
    )


def test_langchain_provider_compatibility():
    """Test compatibility with different LangChain versions."""
    # This test checks for compatibility with different LangChain versions
    # It's useful for catching compatibility issues early

    # check for common providers that should be available
    common_llm_providers = ["openai", "anthropic"]
    common_chat_providers = ["openai", "anthropic", "huggingface"]

    # check for LLM providers
    for provider in common_llm_providers:
        if provider not in _llm_providers:
            raise RuntimeError(
                f"Common LLM provider '{provider}' is not available. "
                "This might be due to a version mismatch with LangChain."
            )

    # check for chat providers
    for provider in common_chat_providers:
        if provider not in _chat_providers:
            raise RuntimeError(
                f"Common chat provider '{provider}' is not available. "
                "This might be due to a version mismatch with LangChain."
            )


# TODO: we might need this
# def test_provider_version_compatibility():
#     """Test compatibility with different LangChain versions."""
#     langchain_version = get_langchain_version()
#
#     if langchain_version != "unknown":
#         version_tuple = _parse_version(langchain_version)
#
#         # we can check for version-specific compatibility issues
#         if version_tuple >= (0, 1, 0):
#             #  we can check for changes introduced in version 0.1.0 for example
#             pass
#
#         if version_tuple >= (0, 2, 0):
#             #  we can check for changes introduced in version 0.2.0 for example
#             pass

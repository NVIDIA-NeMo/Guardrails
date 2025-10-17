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

from nemoguardrails.actions.llm.utils import _infer_provider_from_module


class MockOpenAILLM:
    __module__ = "langchain_openai.chat_models"


class MockAnthropicLLM:
    __module__ = "langchain_anthropic.chat_models"


class MockNVIDIALLM:
    __module__ = "langchain_nvidia_ai_endpoints.chat_models"


class MockCommunityOllama:
    __module__ = "langchain_community.chat_models.ollama"


class MockUnknownLLM:
    __module__ = "some_custom_package.models"


class MockNVIDIAOriginal:
    __module__ = "langchain_nvidia_ai_endpoints.chat_models"


class MockPatchedNVIDIA(MockNVIDIAOriginal):
    __module__ = "nemoguardrails.llm.providers._langchain_nvidia_ai_endpoints_patch"


def test_infer_provider_openai():
    llm = MockOpenAILLM()
    provider = _infer_provider_from_module(llm)
    assert provider == "openai"


def test_infer_provider_anthropic():
    llm = MockAnthropicLLM()
    provider = _infer_provider_from_module(llm)
    assert provider == "anthropic"


def test_infer_provider_nvidia_ai_endpoints():
    llm = MockNVIDIALLM()
    provider = _infer_provider_from_module(llm)
    assert provider == "nvidia_ai_endpoints"


def test_infer_provider_community_ollama():
    llm = MockCommunityOllama()
    provider = _infer_provider_from_module(llm)
    assert provider == "ollama"


def test_infer_provider_unknown():
    llm = MockUnknownLLM()
    provider = _infer_provider_from_module(llm)
    assert provider is None


def test_infer_provider_from_patched_class():
    llm = MockPatchedNVIDIA()
    provider = _infer_provider_from_module(llm)
    assert provider == "nvidia_ai_endpoints"


def test_infer_provider_checks_base_classes():
    class BaseOpenAI:
        __module__ = "langchain_openai.chat_models"

    class CustomWrapper(BaseOpenAI):
        __module__ = "my_custom_wrapper.llms"

    llm = CustomWrapper()
    provider = _infer_provider_from_module(llm)
    assert provider == "openai"


def test_infer_provider_multiple_inheritance():
    class BaseNVIDIA:
        __module__ = "langchain_nvidia_ai_endpoints.chat_models"

    class Mixin:
        __module__ = "some_mixin.utils"

    class MultipleInheritance(Mixin, BaseNVIDIA):
        __module__ = "custom_package.models"

    llm = MultipleInheritance()
    provider = _infer_provider_from_module(llm)
    assert provider == "nvidia_ai_endpoints"


def test_infer_provider_deeply_nested_inheritance():
    class Original:
        __module__ = "langchain_anthropic.chat_models"

    class Wrapper1(Original):
        __module__ = "wrapper1.models"

    class Wrapper2(Wrapper1):
        __module__ = "wrapper2.models"

    class Wrapper3(Wrapper2):
        __module__ = "wrapper3.models"

    llm = Wrapper3()
    provider = _infer_provider_from_module(llm)
    assert provider == "anthropic"

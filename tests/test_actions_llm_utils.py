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

from nemoguardrails.actions.llm.utils import (
    _extract_and_remove_think_tags,
    _infer_provider_from_module,
    _store_reasoning_traces,
)
from nemoguardrails.context import reasoning_trace_var


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


class MockResponse:
    def __init__(self, content="", additional_kwargs=None):
        self.content = content
        self.additional_kwargs = additional_kwargs or {}


def test_store_reasoning_traces_from_additional_kwargs():
    reasoning_trace_var.set(None)

    response = MockResponse(
        content="The answer is 42",
        additional_kwargs={"reasoning_content": "Let me think about this..."},
    )

    _store_reasoning_traces(response)

    assert reasoning_trace_var.get() == "Let me think about this..."


def test_store_reasoning_traces_from_think_tags():
    reasoning_trace_var.set(None)

    response = MockResponse(
        content="<think>Let me think about this...</think>The answer is 42"
    )

    _store_reasoning_traces(response)

    assert reasoning_trace_var.get() == "Let me think about this..."
    assert response.content == "The answer is 42"


def test_store_reasoning_traces_multiline_think_tags():
    reasoning_trace_var.set(None)

    response = MockResponse(
        content="<think>Step 1: Analyze the problem\nStep 2: Consider options\nStep 3: Choose solution</think>The answer is 42"
    )

    _store_reasoning_traces(response)

    assert (
        reasoning_trace_var.get()
        == "Step 1: Analyze the problem\nStep 2: Consider options\nStep 3: Choose solution"
    )
    assert response.content == "The answer is 42"


def test_store_reasoning_traces_prefers_additional_kwargs():
    reasoning_trace_var.set(None)

    response = MockResponse(
        content="<think>This should not be used</think>The answer is 42",
        additional_kwargs={"reasoning_content": "This should be used"},
    )

    _store_reasoning_traces(response)

    assert reasoning_trace_var.get() == "This should be used"


def test_store_reasoning_traces_no_reasoning_content():
    reasoning_trace_var.set(None)

    response = MockResponse(content="The answer is 42")

    _store_reasoning_traces(response)

    assert reasoning_trace_var.get() is None


def test_store_reasoning_traces_empty_reasoning_content():
    reasoning_trace_var.set(None)

    response = MockResponse(
        content="The answer is 42", additional_kwargs={"reasoning_content": ""}
    )

    _store_reasoning_traces(response)

    assert reasoning_trace_var.get() is None


def test_store_reasoning_traces_incomplete_think_tags():
    reasoning_trace_var.set(None)

    response = MockResponse(content="<think>This is incomplete")

    _store_reasoning_traces(response)

    assert reasoning_trace_var.get() is None


def test_store_reasoning_traces_no_content_attribute():
    reasoning_trace_var.set(None)

    class ResponseWithoutContent:
        def __init__(self):
            self.additional_kwargs = {}

    response = ResponseWithoutContent()

    _store_reasoning_traces(response)

    assert reasoning_trace_var.get() is None


def test_store_reasoning_traces_removes_think_tags_with_whitespace():
    reasoning_trace_var.set(None)

    response = MockResponse(
        content="  <think>reasoning here</think>  \n\n  Final answer  "
    )

    _store_reasoning_traces(response)

    assert reasoning_trace_var.get() == "reasoning here"
    assert response.content == "Final answer"


def test_extract_and_remove_think_tags_basic():
    response = MockResponse(content="<think>reasoning</think>answer")

    result = _extract_and_remove_think_tags(response)

    assert result == "reasoning"
    assert response.content == "answer"


def test_extract_and_remove_think_tags_multiline():
    response = MockResponse(content="<think>line1\nline2\nline3</think>final answer")

    result = _extract_and_remove_think_tags(response)

    assert result == "line1\nline2\nline3"
    assert response.content == "final answer"


def test_extract_and_remove_think_tags_no_tags():
    response = MockResponse(content="just a normal response")

    result = _extract_and_remove_think_tags(response)

    assert result is None
    assert response.content == "just a normal response"


def test_extract_and_remove_think_tags_incomplete():
    response = MockResponse(content="<think>incomplete")

    result = _extract_and_remove_think_tags(response)

    assert result is None
    assert response.content == "<think>incomplete"


def test_extract_and_remove_think_tags_no_content_attribute():
    class ResponseWithoutContent:
        pass

    response = ResponseWithoutContent()

    result = _extract_and_remove_think_tags(response)

    assert result is None


def test_extract_and_remove_think_tags_wrong_order():
    response = MockResponse(content="</think> text here <think>")

    result = _extract_and_remove_think_tags(response)

    assert result is None
    assert response.content == "</think> text here <think>"

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

import pytest
from langchain_core.messages import AIMessage

from nemoguardrails.actions.llm.utils import (
    _extract_reasoning_from_additional_kwargs,
    _extract_reasoning_from_content_blocks,
    _extract_tool_calls_from_attribute,
    _extract_tool_calls_from_content_blocks,
    _infer_provider_from_module,
    _store_reasoning_traces,
    _store_tool_calls,
)
from nemoguardrails.context import reasoning_trace_var, tool_calls_var


@pytest.fixture(autouse=True)
def reset_context_vars():
    reasoning_token = reasoning_trace_var.set(None)
    tool_calls_token = tool_calls_var.set(None)

    yield

    reasoning_trace_var.reset(reasoning_token)
    tool_calls_var.reset(tool_calls_token)


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
    def __init__(self, content_blocks=None, additional_kwargs=None, tool_calls=None):
        if content_blocks is not None:
            self.content_blocks = content_blocks
        if additional_kwargs is not None:
            self.additional_kwargs = additional_kwargs
        if tool_calls is not None:
            self.tool_calls = tool_calls


def test_extract_reasoning_from_content_blocks_single_reasoning():
    response = MockResponse(
        content_blocks=[
            {"type": "reasoning", "reasoning": "foo"},
        ]
    )
    reasoning = _extract_reasoning_from_content_blocks(response)
    assert reasoning == "foo"


def test_extract_reasoning_from_content_blocks_with_text_and_reasoning():
    response = MockResponse(
        content_blocks=[
            {"type": "text", "text": "bar"},
            {"type": "reasoning", "reasoning": "Let me think about this problem..."},
        ]
    )
    reasoning = _extract_reasoning_from_content_blocks(response)
    assert reasoning == "Let me think about this problem..."


def test_extract_reasoning_from_content_blocks_returns_first_reasoning():
    response = MockResponse(
        content_blocks=[
            {"type": "reasoning", "reasoning": "First thought"},
            {"type": "reasoning", "reasoning": "Second thought"},
        ]
    )
    reasoning = _extract_reasoning_from_content_blocks(response)
    assert reasoning == "First thought"


def test_extract_reasoning_from_content_blocks_no_reasoning():
    response = MockResponse(
        content_blocks=[
            {"type": "text", "text": "Hello"},
            {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"},
        ]
    )
    reasoning = _extract_reasoning_from_content_blocks(response)
    assert reasoning is None


def test_extract_reasoning_from_content_blocks_no_attribute():
    response = MockResponse()
    reasoning = _extract_reasoning_from_content_blocks(response)
    assert reasoning is None


def test_extract_reasoning_from_additional_kwargs_with_reasoning_content():
    response = MockResponse(additional_kwargs={"reasoning_content": "Let me think about this problem..."})
    reasoning = _extract_reasoning_from_additional_kwargs(response)
    assert reasoning == "Let me think about this problem..."


def test_extract_reasoning_from_additional_kwargs_no_reasoning_content():
    response = MockResponse(additional_kwargs={"other_field": "some value"})
    reasoning = _extract_reasoning_from_additional_kwargs(response)
    assert reasoning is None


def test_extract_reasoning_from_additional_kwargs_no_attribute():
    response = MockResponse()
    reasoning = _extract_reasoning_from_additional_kwargs(response)
    assert reasoning is None


def test_extract_reasoning_from_additional_kwargs_not_dict():
    response = MockResponse(additional_kwargs="not a dict")
    reasoning = _extract_reasoning_from_additional_kwargs(response)
    assert reasoning is None


def test_extract_tool_calls_from_content_blocks_single_tool_call():
    expected_tool_call = {
        "type": "tool_call",
        "name": "foo",
        "args": {"a": "b"},
        "id": "abc_123",
    }
    response = MockResponse(content_blocks=[expected_tool_call])
    tool_calls = _extract_tool_calls_from_content_blocks(response)
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0] == expected_tool_call


def test_extract_tool_calls_from_content_blocks_multiple_tool_calls():
    response = MockResponse(
        content_blocks=[
            {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"},
            {"type": "tool_call", "name": "bar", "args": {"c": "d"}, "id": "abc_234"},
        ]
    )
    tool_calls = _extract_tool_calls_from_content_blocks(response)
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0]["name"] == "foo"
    assert tool_calls[1]["name"] == "bar"


def test_extract_tool_calls_from_content_blocks_mixed_content():
    response = MockResponse(
        content_blocks=[
            {"type": "text", "text": "Hello"},
            {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"},
            {"type": "reasoning", "reasoning": "Thinking..."},
            {"type": "tool_call", "name": "bar", "args": {"c": "d"}, "id": "abc_234"},
        ]
    )
    tool_calls = _extract_tool_calls_from_content_blocks(response)
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0]["name"] == "foo"
    assert tool_calls[1]["name"] == "bar"


def test_extract_tool_calls_from_content_blocks_no_tool_calls():
    response = MockResponse(
        content_blocks=[
            {"type": "text", "text": "Hello"},
            {"type": "reasoning", "reasoning": "Thinking..."},
        ]
    )
    tool_calls = _extract_tool_calls_from_content_blocks(response)
    assert tool_calls is None


def test_extract_tool_calls_from_content_blocks_no_attribute():
    response = MockResponse()
    tool_calls = _extract_tool_calls_from_content_blocks(response)
    assert tool_calls is None


def test_extract_tool_calls_from_attribute_with_tool_calls():
    response = MockResponse(
        tool_calls=[
            {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"},
            {"type": "tool_call", "name": "bar", "args": {"c": "d"}, "id": "abc_234"},
        ]
    )
    tool_calls = _extract_tool_calls_from_attribute(response)
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0]["name"] == "foo"
    assert tool_calls[1]["name"] == "bar"


def test_extract_tool_calls_from_attribute_no_attribute():
    response = MockResponse()
    tool_calls = _extract_tool_calls_from_attribute(response)
    assert tool_calls is None


def test_store_reasoning_traces_from_content_blocks():
    response = MockResponse(
        content_blocks=[
            {"type": "text", "text": "The answer is 42."},
            {"type": "reasoning", "reasoning": "Let me think about this problem..."},
        ]
    )
    _store_reasoning_traces(response)

    reasoning = reasoning_trace_var.get()
    assert reasoning == "Let me think about this problem..."


def test_store_reasoning_traces_from_additional_kwargs():
    response = MockResponse(additional_kwargs={"reasoning_content": "Provider specific reasoning"})
    _store_reasoning_traces(response)

    reasoning = reasoning_trace_var.get()
    assert reasoning == "Provider specific reasoning"


def test_store_reasoning_traces_prefers_content_blocks_over_additional_kwargs():
    response = MockResponse(
        content_blocks=[
            {"type": "reasoning", "reasoning": "Content blocks reasoning"},
        ],
        additional_kwargs={"reasoning_content": "Additional kwargs reasoning"},
    )
    _store_reasoning_traces(response)

    reasoning = reasoning_trace_var.get()
    assert reasoning == "Content blocks reasoning"


def test_store_reasoning_traces_fallback_to_additional_kwargs():
    response = MockResponse(
        content_blocks=[
            {"type": "text", "text": "No reasoning here"},
        ],
        additional_kwargs={"reasoning_content": "Fallback reasoning"},
    )
    _store_reasoning_traces(response)

    reasoning = reasoning_trace_var.get()
    assert reasoning == "Fallback reasoning"


def test_store_reasoning_traces_no_reasoning():
    response = MockResponse(
        content_blocks=[
            {"type": "text", "text": "Just text"},
        ]
    )
    _store_reasoning_traces(response)

    reasoning = reasoning_trace_var.get()
    assert reasoning is None


def test_store_tool_calls_from_content_blocks():
    response = MockResponse(
        content_blocks=[
            {"type": "text", "text": "Hello"},
            {
                "type": "tool_call",
                "name": "search",
                "args": {"query": "weather"},
                "id": "call_1",
            },
            {
                "type": "tool_call",
                "name": "calculator",
                "args": {"expr": "2+2"},
                "id": "call_2",
            },
        ]
    )
    _store_tool_calls(response)

    tool_calls = tool_calls_var.get()
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0]["name"] == "search"
    assert tool_calls[1]["name"] == "calculator"


def test_store_tool_calls_from_attribute():
    response = MockResponse(
        tool_calls=[
            {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"},
            {"type": "tool_call", "name": "bar", "args": {"c": "d"}, "id": "abc_234"},
        ]
    )
    _store_tool_calls(response)

    tool_calls = tool_calls_var.get()
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0]["name"] == "foo"
    assert tool_calls[1]["name"] == "bar"


def test_store_tool_calls_prefers_content_blocks_over_attribute():
    response = MockResponse(
        content_blocks=[
            {"type": "tool_call", "name": "from_blocks", "args": {}, "id": "1"},
        ],
        tool_calls=[
            {"type": "tool_call", "name": "from_attribute", "args": {}, "id": "2"},
        ],
    )
    _store_tool_calls(response)

    tool_calls = tool_calls_var.get()
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "from_blocks"


def test_store_tool_calls_fallback_to_attribute():
    response = MockResponse(
        content_blocks=[
            {"type": "text", "text": "No tool calls here"},
        ],
        tool_calls=[
            {"type": "tool_call", "name": "fallback_tool", "args": {}, "id": "1"},
        ],
    )
    _store_tool_calls(response)

    tool_calls = tool_calls_var.get()
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "fallback_tool"


def test_store_tool_calls_no_tool_calls():
    response = MockResponse(
        content_blocks=[
            {"type": "text", "text": "Just text"},
        ]
    )
    _store_tool_calls(response)

    tool_calls = tool_calls_var.get()
    assert tool_calls is None


def test_store_reasoning_traces_with_real_aimessage_from_content_blocks():
    message = AIMessage(
        content="The answer is 42.",
        additional_kwargs={"reasoning_content": "Let me think about this problem..."},
    )

    _store_reasoning_traces(message)

    reasoning = reasoning_trace_var.get()
    assert reasoning == "Let me think about this problem..."


def test_store_reasoning_traces_with_real_aimessage_no_reasoning():
    message = AIMessage(
        content="The answer is 42.",
        additional_kwargs={"other_field": "some value"},
    )

    _store_reasoning_traces(message)

    reasoning = reasoning_trace_var.get()
    assert reasoning is None


def test_store_tool_calls_with_real_aimessage_from_content_blocks():
    message = AIMessage(
        "",
        tool_calls=[{"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"}],
    )

    _store_tool_calls(message)

    tool_calls = tool_calls_var.get()
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["type"] == "tool_call"
    assert tool_calls[0]["name"] == "foo"
    assert tool_calls[0]["args"] == {"a": "b"}
    assert tool_calls[0]["id"] == "abc_123"


def test_store_tool_calls_with_real_aimessage_mixed_content():
    message = AIMessage(
        "foo",
        tool_calls=[{"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"}],
    )

    _store_tool_calls(message)

    tool_calls = tool_calls_var.get()
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["type"] == "tool_call"
    assert tool_calls[0]["name"] == "foo"


def test_store_tool_calls_with_real_aimessage_multiple_tool_calls():
    message = AIMessage(
        "",
        tool_calls=[
            {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"},
            {"type": "tool_call", "name": "bar", "args": {"c": "d"}, "id": "abc_234"},
        ],
    )

    _store_tool_calls(message)

    tool_calls = tool_calls_var.get()
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0]["name"] == "foo"
    assert tool_calls[1]["name"] == "bar"

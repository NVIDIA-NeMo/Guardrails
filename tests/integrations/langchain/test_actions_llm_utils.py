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

from unittest.mock import AsyncMock, MagicMock

import pytest

from nemoguardrails.actions.llm.utils import (
    _log_completion,
    _store_reasoning_traces,
    _store_tool_calls,
    _stream_llm_call,
    _update_token_stats_from_chunk,
    llm_call,
)
from nemoguardrails.context import (
    llm_call_info_var,
    llm_response_metadata_var,
    llm_stats_var,
    reasoning_trace_var,
    tool_calls_var,
)
from nemoguardrails.exceptions import LLMCallException
from nemoguardrails.integrations.langchain.llm_adapter import (
    LangChainLLMAdapter,
    _infer_provider_from_module,
)
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.streaming import StreamingHandler
from nemoguardrails.types import ChatMessage, LLMResponse, LLMResponseChunk, Role, ToolCall, ToolCallFunction, UsageInfo


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


def test_store_reasoning_traces_from_reasoning_field():
    response = LLMResponse(
        content="The answer is 42.",
        reasoning="Let me think about this problem...",
    )
    _store_reasoning_traces(response)

    reasoning = reasoning_trace_var.get()
    assert reasoning == "Let me think about this problem..."


def test_store_reasoning_traces_no_reasoning():
    response = LLMResponse(content="Just text")
    _store_reasoning_traces(response)

    reasoning = reasoning_trace_var.get()
    assert reasoning is None


def test_store_tool_calls_from_attribute():
    response = LLMResponse(
        content="",
        tool_calls=[
            ToolCall(id="abc_123", function=ToolCallFunction(name="foo", arguments={"a": "b"})),
            ToolCall(id="abc_234", function=ToolCallFunction(name="bar", arguments={"c": "d"})),
        ],
    )
    _store_tool_calls(response)

    tool_calls = tool_calls_var.get()
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "foo"
    assert tool_calls[0]["function"]["arguments"] == {"a": "b"}
    assert tool_calls[1]["function"]["name"] == "bar"
    assert tool_calls[1]["function"]["arguments"] == {"c": "d"}


def test_store_tool_calls_no_tool_calls():
    response = LLMResponse(content="Just text")
    _store_tool_calls(response)

    tool_calls = tool_calls_var.get()
    assert tool_calls is None


def test_store_reasoning_traces_with_reasoning():
    response = LLMResponse(
        content="The answer is 42.",
        reasoning="Let me think about this problem...",
    )

    _store_reasoning_traces(response)

    reasoning = reasoning_trace_var.get()
    assert reasoning == "Let me think about this problem..."


def test_store_reasoning_traces_with_no_reasoning():
    response = LLMResponse(content="The answer is 42.")

    _store_reasoning_traces(response)

    reasoning = reasoning_trace_var.get()
    assert reasoning is None


def test_store_tool_calls_with_tool_call_objects():
    response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="abc_123", function=ToolCallFunction(name="foo", arguments={"a": "b"}))],
    )

    _store_tool_calls(response)

    tool_calls = tool_calls_var.get()
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "foo"
    assert tool_calls[0]["function"]["arguments"] == {"a": "b"}
    assert tool_calls[0]["id"] == "abc_123"


def test_store_tool_calls_with_content_and_tool_calls():
    response = LLMResponse(
        content="foo",
        tool_calls=[ToolCall(id="abc_123", function=ToolCallFunction(name="foo", arguments={"a": "b"}))],
    )

    _store_tool_calls(response)

    tool_calls = tool_calls_var.get()
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "foo"


def test_store_tool_calls_with_multiple_tool_call_objects():
    response = LLMResponse(
        content="",
        tool_calls=[
            ToolCall(id="abc_123", function=ToolCallFunction(name="foo", arguments={"a": "b"})),
            ToolCall(id="abc_234", function=ToolCallFunction(name="bar", arguments={"c": "d"})),
        ],
    )

    _store_tool_calls(response)

    tool_calls = tool_calls_var.get()
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "foo"
    assert tool_calls[1]["function"]["name"] == "bar"


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_params", [None, {}])
async def test_llm_call_stop_tokens_passed_without_llm_params(llm_params):
    from unittest.mock import AsyncMock, MagicMock

    from nemoguardrails.actions.llm.utils import llm_call

    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(content="response")

    wrapped = LangChainLLMAdapter(mock_llm)
    await llm_call(wrapped, "prompt", stop=["User:"], llm_params=llm_params)

    assert mock_llm.ainvoke.call_args[1]["stop"] == ["User:"]


@pytest.mark.asyncio
async def test_llm_call_exception_enrichment_with_model_and_provider():
    mock_llm = MockOpenAILLM()
    mock_llm.model_name = "gpt-4"
    mock_llm.ainvoke = AsyncMock(side_effect=ConnectionError("Connection refused"))

    wrapped = LangChainLLMAdapter(mock_llm)
    with pytest.raises(LLMCallException) as exc_info:
        await llm_call(wrapped, "test prompt")

    exc_str = str(exc_info.value)
    assert "gpt-4" in exc_str
    assert "provider=openai" in exc_str
    assert "Connection refused" in exc_str
    assert isinstance(exc_info.value.inner_exception, ConnectionError)


@pytest.mark.asyncio
async def test_llm_call_exception_without_provider():
    mock_llm = MockUnknownLLM()
    mock_llm.model_name = "custom-model"
    mock_llm.ainvoke = AsyncMock(side_effect=ValueError("Invalid request"))

    wrapped = LangChainLLMAdapter(mock_llm)
    with pytest.raises(LLMCallException) as exc_info:
        await llm_call(wrapped, "test prompt")

    exc_str = str(exc_info.value)
    assert "custom-model" in exc_str
    assert "Invalid request" in exc_str


@pytest.mark.asyncio
async def test_llm_call_kwargs_flow_through_to_generate():
    mock_llm = MagicMock()
    mock_llm.model_name = "gpt-4"
    bound_llm = AsyncMock()
    bound_llm.ainvoke.return_value = MagicMock(content="response")
    mock_llm.bind.return_value = bound_llm

    wrapped = LangChainLLMAdapter(mock_llm)
    await llm_call(wrapped, "prompt", llm_params={"temperature": 0.5, "max_tokens": 100})

    mock_llm.bind.assert_called_once_with(temperature=0.5, max_tokens=100)


class TestLogCompletion:
    def test_logs_completion_to_llm_call_info(self):
        llm_call_info = LLMCallInfo()
        llm_call_info_var.set(llm_call_info)

        response = LLMResponse(content="This is the response")
        _log_completion(response)

        assert llm_call_info.completion == "This is the response"

    def test_handles_reasoning_content(self):
        llm_call_info = LLMCallInfo()
        llm_call_info_var.set(llm_call_info)

        response = LLMResponse(
            content="Final answer",
            reasoning="Step 1: Think",
        )
        _log_completion(response)

        assert llm_call_info.completion == "Final answer"


class TestUpdateTokenStatsFromChunk:
    def test_extracts_from_usage(self):
        llm_call_info = LLMCallInfo()
        llm_call_info_var.set(llm_call_info)

        llm_stats = LLMStats()
        llm_stats_var.set(llm_stats)

        chunk = LLMResponseChunk(
            delta_content="",
            usage=UsageInfo(total_tokens=25, input_tokens=15, output_tokens=10),
        )

        _update_token_stats_from_chunk(chunk)

        assert llm_call_info.total_tokens == 25
        assert llm_call_info.prompt_tokens == 15
        assert llm_call_info.completion_tokens == 10

    def test_extracts_from_usage_metadata_via_adapter(self):
        llm_call_info = LLMCallInfo()
        llm_call_info_var.set(llm_call_info)

        llm_stats = LLMStats()
        llm_stats_var.set(llm_stats)

        chunk = LLMResponseChunk(
            delta_content="",
            usage=UsageInfo(total_tokens=30, input_tokens=20, output_tokens=10),
        )

        _update_token_stats_from_chunk(chunk)

        assert llm_call_info.total_tokens == 30
        assert llm_call_info.prompt_tokens == 20
        assert llm_call_info.completion_tokens == 10


class TestLlmCallDictToChatMessageConversion:
    @pytest.mark.asyncio
    async def test_llm_call_converts_dict_prompt_to_chat_messages(self):
        received_prompt = None

        class CaptureLLM:
            async def generate_async(self, prompt, *, stop=None, **kwargs):
                nonlocal received_prompt
                received_prompt = prompt
                return LLMResponse(content="ok")

            async def stream_async(self, prompt, *, stop=None, **kwargs):
                yield LLMResponseChunk(delta_content="ok")

            @property
            def model_name(self):
                return "test"

            @property
            def provider_name(self):
                return None

            @property
            def provider_url(self):
                return None

        model = CaptureLLM()
        dict_prompt = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        await llm_call(model, dict_prompt)

        assert received_prompt is not None
        assert isinstance(received_prompt, list)
        assert len(received_prompt) == 2
        assert all(isinstance(m, ChatMessage) for m in received_prompt)
        assert received_prompt[0].role == Role.SYSTEM
        assert received_prompt[0].content == "You are helpful."
        assert received_prompt[1].role == Role.USER
        assert received_prompt[1].content == "Hello"

    @pytest.mark.asyncio
    async def test_llm_call_passes_string_prompt_unchanged(self):
        received_prompt = None

        class CaptureLLM:
            async def generate_async(self, prompt, *, stop=None, **kwargs):
                nonlocal received_prompt
                received_prompt = prompt
                return LLMResponse(content="ok")

            async def stream_async(self, prompt, *, stop=None, **kwargs):
                yield LLMResponseChunk(delta_content="ok")

            @property
            def model_name(self):
                return "test"

            @property
            def provider_name(self):
                return None

            @property
            def provider_url(self):
                return None

        model = CaptureLLM()
        await llm_call(model, "simple string prompt")

        assert received_prompt == "simple string prompt"

    @pytest.mark.asyncio
    async def test_llm_call_handles_empty_list(self):
        received_prompt = None

        class CaptureLLM:
            async def generate_async(self, prompt, *, stop=None, **kwargs):
                nonlocal received_prompt
                received_prompt = prompt
                return LLMResponse(content="ok")

            async def stream_async(self, prompt, *, stop=None, **kwargs):
                yield LLMResponseChunk(delta_content="ok")

            @property
            def model_name(self):
                return "test"

            @property
            def provider_name(self):
                return None

            @property
            def provider_url(self):
                return None

        model = CaptureLLM()
        await llm_call(model, [])

        assert received_prompt == []


def _make_chunk_model(chunks):
    class _Model:
        model_name = "test-model"
        provider_name = "test"
        provider_url = None

        async def generate_async(self, prompt, *, stop=None, **kwargs):
            return LLMResponse(content="")

        async def stream_async(self, prompt, *, stop=None, **kwargs):
            for c in chunks:
                yield c

    return _Model()


class TestStreamLlmCallAccumulation:
    @pytest.mark.asyncio
    async def test_accumulates_tool_calls(self):
        tc = [ToolCall(id="call_1", function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"}))]
        model = _make_chunk_model(
            [
                LLMResponseChunk(model="gpt-4o"),
                LLMResponseChunk(delta_tool_calls=tc, finish_reason="tool_calls"),
                LLMResponseChunk(usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15)),
            ]
        )

        result = await _stream_llm_call(model, "test", StreamingHandler(), stop=None)

        assert result.tool_calls == tc
        assert result.model == "gpt-4o"
        assert result.finish_reason == "tool_calls"
        assert result.usage.total_tokens == 15
        assert tool_calls_var.get() is not None

    @pytest.mark.asyncio
    async def test_accumulates_reasoning(self):
        model = _make_chunk_model(
            [
                LLMResponseChunk(delta_reasoning="Let me ", model="gpt-4o"),
                LLMResponseChunk(delta_reasoning="think..."),
                LLMResponseChunk(delta_content="42", finish_reason="stop"),
                LLMResponseChunk(usage=UsageInfo(input_tokens=5, output_tokens=3, total_tokens=8)),
            ]
        )

        result = await _stream_llm_call(model, "test", StreamingHandler(), stop=None)

        assert result.content == "42"
        assert result.reasoning == "Let me think..."
        assert result.model == "gpt-4o"
        assert result.finish_reason == "stop"
        assert reasoning_trace_var.get() == "Let me think..."

    @pytest.mark.asyncio
    async def test_text_only(self):
        model = _make_chunk_model(
            [
                LLMResponseChunk(delta_content="Hello", model="gpt-4o"),
                LLMResponseChunk(delta_content=" world", finish_reason="stop"),
                LLMResponseChunk(usage=UsageInfo(input_tokens=5, output_tokens=2, total_tokens=7)),
            ]
        )

        result = await _stream_llm_call(model, "test", StreamingHandler(), stop=None)

        assert result.content == "Hello world"
        assert result.tool_calls is None
        assert result.reasoning is None
        assert result.model == "gpt-4o"
        assert result.finish_reason == "stop"
        assert result.usage.total_tokens == 7

    @pytest.mark.asyncio
    async def test_request_id_accumulated(self):
        model = _make_chunk_model(
            [
                LLMResponseChunk(delta_content="hi", request_id="req-123", model="gpt-4o"),
                LLMResponseChunk(finish_reason="stop"),
            ]
        )

        result = await _stream_llm_call(model, "test", StreamingHandler(), stop=None)

        assert result.request_id == "req-123"

    @pytest.mark.asyncio
    async def test_clears_tool_calls_var_when_none(self):
        tool_calls_var.set([{"id": "stale", "type": "function", "function": {"name": "old", "arguments": {}}}])

        model = _make_chunk_model(
            [
                LLMResponseChunk(delta_content="no tools here", finish_reason="stop"),
            ]
        )

        await _stream_llm_call(model, "test", StreamingHandler(), stop=None)

        assert tool_calls_var.get() is None

    @pytest.mark.asyncio
    async def test_clears_reasoning_var_when_none(self):
        reasoning_trace_var.set("stale reasoning")

        model = _make_chunk_model(
            [
                LLMResponseChunk(delta_content="no reasoning", finish_reason="stop"),
            ]
        )

        await _stream_llm_call(model, "test", StreamingHandler(), stop=None)

        assert reasoning_trace_var.get() is None

    @pytest.mark.asyncio
    async def test_provider_metadata_stored_flat(self):
        model = _make_chunk_model(
            [
                LLMResponseChunk(
                    delta_content="hi",
                    provider_metadata={"system_fingerprint": "fp_abc"},
                    finish_reason="stop",
                ),
            ]
        )

        await _stream_llm_call(model, "test", StreamingHandler(), stop=None)

        metadata = llm_response_metadata_var.get()
        assert metadata == {"system_fingerprint": "fp_abc"}

    @pytest.mark.asyncio
    async def test_clears_metadata_var_when_none(self):
        llm_response_metadata_var.set({"stale": True})

        model = _make_chunk_model(
            [
                LLMResponseChunk(delta_content="no metadata", finish_reason="stop"),
            ]
        )

        await _stream_llm_call(model, "test", StreamingHandler(), stop=None)

        assert llm_response_metadata_var.get() is None

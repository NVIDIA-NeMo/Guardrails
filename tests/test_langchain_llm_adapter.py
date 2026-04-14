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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage

from nemoguardrails.integrations.langchain.llm_adapter import (
    LangChainFramework,
    LangChainLLMAdapter,
    _langchain_chunk_to_llm_response_chunk,
    _langchain_response_to_llm_response,
)
from nemoguardrails.integrations.langchain.message_utils import (
    chatmessage_to_langchain_message,
    chatmessages_to_langchain_messages,
)
from nemoguardrails.types import ChatMessage, LLMModel, LLMResponse, LLMResponseChunk, Role, ToolCall, ToolCallFunction


class TestLangChainLLMAdapter:
    def test_raw_llm_property(self):
        mock_llm = MagicMock()
        adapter = LangChainLLMAdapter(mock_llm)
        assert adapter.raw_llm is mock_llm

    def test_model_name_property(self):
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"
        adapter = LangChainLLMAdapter(mock_llm)
        assert adapter.model_name == "gpt-4"

    def test_provider_name_property(self):
        mock_llm = MagicMock()
        mock_llm.__module__ = "langchain_openai.chat_models"
        type(mock_llm).__module__ = "langchain_openai.chat_models"
        adapter = LangChainLLMAdapter(mock_llm)
        assert adapter.provider_name == "openai"

    def test_provider_url_from_base_url(self):
        mock_llm = MagicMock(spec=[])
        mock_llm.base_url = "https://api.openai.com/v1"
        adapter = LangChainLLMAdapter(mock_llm)
        assert adapter.provider_url == "https://api.openai.com/v1"

    def test_provider_url_from_azure_endpoint(self):
        mock_llm = MagicMock(spec=[])
        mock_llm.azure_endpoint = "https://example.openai.azure.com"
        adapter = LangChainLLMAdapter(mock_llm)
        assert adapter.provider_url == "https://example.openai.azure.com"

    def test_provider_url_from_server_url(self):
        mock_llm = MagicMock(spec=[])
        mock_llm.server_url = "https://triton.example.com:8000"
        adapter = LangChainLLMAdapter(mock_llm)
        assert adapter.provider_url == "https://triton.example.com:8000"

    def test_provider_url_from_nested_client(self):
        mock_llm = MagicMock(spec=[])
        mock_llm.client = MagicMock()
        mock_llm.client.base_url = "https://custom.endpoint.com/v1"
        adapter = LangChainLLMAdapter(mock_llm)
        assert adapter.provider_url == "https://custom.endpoint.com/v1"

    def test_provider_url_returns_none_when_not_available(self):
        mock_llm = MagicMock(spec=[])
        adapter = LangChainLLMAdapter(mock_llm)
        assert adapter.provider_url is None

    @pytest.mark.asyncio
    async def test_generate_with_string_prompt(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="hello world")
        adapter = LangChainLLMAdapter(mock_llm)

        result = await adapter.generate_async("say hello")

        assert isinstance(result, LLMResponse)
        assert result.content == "hello world"
        mock_llm.ainvoke.assert_called_once_with("say hello", stop=None)

    @pytest.mark.asyncio
    async def test_generate_with_chat_message_list(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="hi there")
        adapter = LangChainLLMAdapter(mock_llm)

        result = await adapter.generate_async([ChatMessage.from_user("hello")])

        assert isinstance(result, LLMResponse)
        assert result.content == "hi there"
        mock_llm.ainvoke.assert_called_once()
        call_args = mock_llm.ainvoke.call_args
        assert len(call_args[0][0]) == 1

    @pytest.mark.asyncio
    async def test_generate_returns_llm_response_with_reasoning(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(
            content="response",
            additional_kwargs={"reasoning_content": "thinking"},
            response_metadata={"model": "gpt-4"},
        )
        adapter = LangChainLLMAdapter(mock_llm)

        result = await adapter.generate_async("prompt")

        assert isinstance(result, LLMResponse)
        assert result.content == "response"
        assert result.reasoning == "thinking"
        assert result.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_generate_maps_tool_calls(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {"name": "search", "args": {"q": "weather"}, "id": "tc_1", "type": "tool_call"},
            ],
        )
        adapter = LangChainLLMAdapter(mock_llm)

        result = await adapter.generate_async("prompt")

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert isinstance(result.tool_calls[0], ToolCall)
        assert result.tool_calls[0].function.name == "search"
        assert result.tool_calls[0].function.arguments == {"q": "weather"}
        assert result.tool_calls[0].id == "tc_1"
        assert result.tool_calls[0].type == "function"

    @pytest.mark.asyncio
    async def test_generate_maps_usage_metadata(self):
        mock_llm = AsyncMock()
        response = AIMessage(content="ok")
        response.usage_metadata = {"total_tokens": 100, "input_tokens": 60, "output_tokens": 40}
        mock_llm.ainvoke.return_value = response
        adapter = LangChainLLMAdapter(mock_llm)

        result = await adapter.generate_async("prompt")

        assert result.usage is not None
        assert result.usage.total_tokens == 100
        assert result.usage.input_tokens == 60
        assert result.usage.output_tokens == 40

    @pytest.mark.asyncio
    async def test_stream_yields_llm_response_chunks(self):
        mock_llm = MagicMock()
        chunks = [
            MagicMock(content="hel", response_metadata=None, usage_metadata=None, generation_info=None),
            MagicMock(content="lo", response_metadata=None, usage_metadata=None, generation_info=None),
        ]

        async def mock_astream(*args, **kwargs):
            for c in chunks:
                yield c

        mock_llm.astream = mock_astream
        adapter = LangChainLLMAdapter(mock_llm)

        results = []
        async for chunk in adapter.stream_async("say hello"):
            results.append(chunk)

        assert len(results) == 2
        assert all(isinstance(r, LLMResponseChunk) for r in results)
        assert results[0].delta_content == "hel"
        assert results[1].delta_content == "lo"

    @pytest.mark.asyncio
    async def test_generate_passes_kwargs_via_bind(self):
        mock_llm = MagicMock()
        bound_llm = AsyncMock()
        bound_llm.ainvoke.return_value = AIMessage(content="bound response")
        mock_llm.bind.return_value = bound_llm
        adapter = LangChainLLMAdapter(mock_llm)

        result = await adapter.generate_async("prompt", temperature=0.5, max_tokens=100)

        mock_llm.bind.assert_called_once_with(temperature=0.5, max_tokens=100)
        bound_llm.ainvoke.assert_called_once_with("prompt", stop=None)
        assert result.content == "bound response"

    @pytest.mark.asyncio
    async def test_generate_no_kwargs_skips_bind(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="direct response")
        adapter = LangChainLLMAdapter(mock_llm)

        result = await adapter.generate_async("prompt")

        mock_llm.bind.assert_not_called()
        assert result.content == "direct response"

    @pytest.mark.asyncio
    async def test_generate_filters_reasoning_model_params(self):
        mock_llm = MagicMock()
        mock_llm.model_name = "o1-preview"
        bound_llm = AsyncMock()
        bound_llm.ainvoke.return_value = AIMessage(content="reasoning response")
        mock_llm.bind.return_value = bound_llm
        adapter = LangChainLLMAdapter(mock_llm)

        result = await adapter.generate_async("prompt", temperature=0.5, max_tokens=100)

        mock_llm.bind.assert_called_once_with(max_tokens=100)
        assert result.content == "reasoning response"

    def test_satisfies_llm_model_protocol(self):
        mock_llm = MagicMock()
        mock_llm.model_name = "test"
        adapter = LangChainLLMAdapter(mock_llm)
        assert isinstance(adapter, LLMModel)


class TestLangChainFramework:
    def test_framework_creates_model(self):
        framework = LangChainFramework()

        mock_raw_llm = MagicMock()
        with patch(
            "nemoguardrails.integrations.langchain.langchain_initializer.init_langchain_model",
            return_value=mock_raw_llm,
        ) as mock_init:
            model = framework.create_model(
                model_name="gpt-4",
                provider_name="openai",
                model_kwargs={"mode": "chat", "temperature": 0.5},
            )

        assert isinstance(model, LangChainLLMAdapter)
        assert model.raw_llm is mock_raw_llm
        mock_init.assert_called_once_with(
            model_name="gpt-4",
            provider_name="openai",
            mode="chat",
            kwargs={"temperature": 0.5},
        )


class TestFilterReasoningModelParams:
    def _make_adapter(self, model_name):
        mock_llm = MagicMock()
        mock_llm.model_name = model_name
        return LangChainLLMAdapter(mock_llm)

    @pytest.mark.parametrize(
        "model,params,expected",
        [
            ("gpt-4", {"temperature": 0.5, "max_tokens": 100}, {"temperature": 0.5, "max_tokens": 100}),
            ("gpt-4o", {"temperature": 0.7}, {"temperature": 0.7}),
            ("gpt-4o-mini", {"temperature": 0.3, "max_tokens": 50}, {"temperature": 0.3, "max_tokens": 50}),
            ("gpt-5-chat", {"temperature": 0.5}, {"temperature": 0.5}),
            ("o1-preview", {"temperature": 0.001, "max_tokens": 100}, {"max_tokens": 100}),
            ("o1-mini", {"temperature": 0.5}, {}),
            ("o3", {"temperature": 0.001, "max_tokens": 200}, {"max_tokens": 200}),
            ("o3-mini", {"temperature": 0.1}, {}),
            ("gpt-5", {"temperature": 0.001}, {}),
            ("gpt-5-mini", {"temperature": 0.5, "max_tokens": 100}, {"max_tokens": 100}),
            ("gpt-5-nano", {"temperature": 0.001}, {}),
            ("o1-preview", {"max_tokens": 100}, {"max_tokens": 100}),
            ("o1-preview", {}, {}),
        ],
    )
    def test_filter_params(self, model, params, expected):
        adapter = self._make_adapter(model)
        result = adapter._filter_reasoning_model_params(params)
        assert result == expected

    def test_returns_none_when_params_is_none(self):
        adapter = self._make_adapter("gpt-4")
        result = adapter._filter_reasoning_model_params(None)
        assert result is None

    def test_does_not_modify_original_params(self):
        adapter = self._make_adapter("o1-preview")
        params = {"temperature": 0.5, "max_tokens": 100}
        adapter._filter_reasoning_model_params(params)
        assert params == {"temperature": 0.5, "max_tokens": 100}


class TestConversionHelpers:
    def test_langchain_response_to_llm_response_basic(self):
        response = AIMessage(content="hello")
        result = _langchain_response_to_llm_response(response)

        assert isinstance(result, LLMResponse)
        assert result.content == "hello"

    def test_langchain_response_to_llm_response_with_tool_calls(self):
        response = AIMessage(
            content="",
            tool_calls=[
                {"name": "fn", "args": {"x": 1}, "id": "tc1", "type": "tool_call"},
            ],
        )
        result = _langchain_response_to_llm_response(response)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "fn"
        assert result.tool_calls[0].function.arguments == {"x": 1}

    def test_langchain_response_fallback_to_str(self):
        result = _langchain_response_to_llm_response("plain text")

        assert isinstance(result, LLMResponse)
        assert result.content == "plain text"

    def test_response_list_content_anthropic_thinking(self):
        response = AIMessage(
            content=[
                {"type": "thinking", "thinking": "Let me reason about this..."},
                {"type": "text", "text": "The answer is 42."},
            ],
        )
        result = _langchain_response_to_llm_response(response)

        assert isinstance(result.content, str)
        assert result.content == "The answer is 42."

    def test_response_list_content_mixed_blocks(self):
        response = AIMessage(
            content=[
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ],
        )
        result = _langchain_response_to_llm_response(response)

        assert result.content == "Hello world"

    def test_chunk_list_content_flattened(self):
        chunk = AIMessageChunk(
            content=[{"type": "text", "text": "streamed"}],
        )
        result = _langchain_chunk_to_llm_response_chunk(chunk)

        assert isinstance(result.delta_content, str)
        assert result.delta_content == "streamed"

    def test_chunk_midstream_content_only(self):
        chunk = AIMessageChunk(content="Hello", response_metadata={"model_provider": "openai"})

        result = _langchain_chunk_to_llm_response_chunk(chunk)

        assert isinstance(result, LLMResponseChunk)
        assert result.delta_content == "Hello"
        assert result.finish_reason is None
        assert result.model is None
        assert result.delta_reasoning is None
        assert result.usage is None

    def test_chunk_openai_finish_reason(self):
        chunk = AIMessageChunk(
            content="",
            response_metadata={
                "finish_reason": "stop",
                "model_name": "gpt-4o-mini-2024-07-18",
                "system_fingerprint": "fp_abc123",
                "service_tier": "default",
            },
        )

        result = _langchain_chunk_to_llm_response_chunk(chunk)

        assert result.finish_reason == "stop"
        assert result.model == "gpt-4o-mini-2024-07-18"
        assert result.provider_metadata["system_fingerprint"] == "fp_abc123"
        assert "finish_reason" not in result.provider_metadata
        assert "model_name" not in result.provider_metadata

    def test_chunk_openai_usage(self):
        chunk = AIMessageChunk(
            content="",
            usage_metadata={
                "input_tokens": 50,
                "output_tokens": 20,
                "total_tokens": 70,
            },
        )

        result = _langchain_chunk_to_llm_response_chunk(chunk)

        assert result.usage is not None
        assert result.usage.input_tokens == 50
        assert result.usage.output_tokens == 20
        assert result.usage.total_tokens == 70

    def test_chunk_nvidia_finish_reason(self):
        chunk = AIMessageChunk(
            content="",
            response_metadata={
                "finish_reason": "stop",
                "model_name": "meta/llama-3.3-70b-instruct",
            },
        )

        result = _langchain_chunk_to_llm_response_chunk(chunk)

        assert result.finish_reason == "stop"
        assert result.model == "meta/llama-3.3-70b-instruct"

    def test_chunk_anthropic_reasoning(self):
        chunk = AIMessageChunk(
            content="",
            additional_kwargs={"reasoning_content": "Let me think..."},
            response_metadata={
                "stop_reason": "end_turn",
                "model": "claude-sonnet-4-20250514",
            },
        )

        result = _langchain_chunk_to_llm_response_chunk(chunk)

        assert result.delta_reasoning == "Let me think..."
        assert result.finish_reason == "stop"
        assert result.model == "claude-sonnet-4-20250514"

    def test_chunk_tool_calls_finish_reason(self):
        chunk = AIMessageChunk(
            content="",
            response_metadata={
                "finish_reason": "tool_calls",
                "model_name": "gpt-4o-mini-2024-07-18",
            },
        )

        result = _langchain_chunk_to_llm_response_chunk(chunk)

        assert result.finish_reason == "tool_calls"

    def test_chunk_trailing_empty(self):
        chunk = AIMessageChunk(content="", response_metadata={})

        result = _langchain_chunk_to_llm_response_chunk(chunk)

        assert result.delta_content == ""
        assert result.finish_reason is None
        assert result.model is None
        assert result.usage is None
        assert result.provider_metadata is None

    def test_chunk_with_text_fallback(self):
        chunk = MagicMock(spec=[])
        chunk.text = "world"

        result = _langchain_chunk_to_llm_response_chunk(chunk)

        assert result.delta_content == "world"

    def test_chunk_fallback_str(self):
        result = _langchain_chunk_to_llm_response_chunk("raw string")

        assert isinstance(result, LLMResponseChunk)
        assert result.delta_content == "raw string"


class TestChatMessageToLangChain:
    def test_user_message(self):
        msg = ChatMessage(role=Role.USER, content="hello")
        result = chatmessage_to_langchain_message(msg)

        assert isinstance(result, HumanMessage)
        assert result.content == "hello"

    def test_system_message(self):
        msg = ChatMessage(role=Role.SYSTEM, content="you are helpful")
        result = chatmessage_to_langchain_message(msg)

        assert isinstance(result, SystemMessage)
        assert result.content == "you are helpful"

    def test_assistant_message(self):
        msg = ChatMessage(role=Role.ASSISTANT, content="hi there")
        result = chatmessage_to_langchain_message(msg)

        assert isinstance(result, AIMessage)
        assert result.content == "hi there"
        assert result.tool_calls == []

    def test_assistant_message_with_tool_calls(self):
        msg = ChatMessage(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                ToolCall(
                    id="call_123",
                    function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"}),
                )
            ],
        )
        result = chatmessage_to_langchain_message(msg)

        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["args"] == {"city": "Paris"}
        assert result.tool_calls[0]["id"] == "call_123"

    def test_tool_message(self):
        msg = ChatMessage(role=Role.TOOL, content='{"temp": 22}', tool_call_id="call_123")
        result = chatmessage_to_langchain_message(msg)

        assert isinstance(result, ToolMessage)
        assert result.content == '{"temp": 22}'
        assert result.tool_call_id == "call_123"

    def test_tool_message_missing_tool_call_id(self):
        msg = ChatMessage(role=Role.TOOL, content="result")
        result = chatmessage_to_langchain_message(msg)

        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == ""

    def test_none_content_becomes_empty_string(self):
        msg = ChatMessage(role=Role.USER, content=None)
        result = chatmessage_to_langchain_message(msg)

        assert isinstance(result, HumanMessage)
        assert result.content == ""

    def test_unsupported_role_raises(self):
        msg = ChatMessage(role=Role.USER, content="test")
        msg.role = "unknown"
        with pytest.raises(ValueError, match="Unsupported role"):
            chatmessage_to_langchain_message(msg)

    def test_batch_conversion(self):
        msgs = [
            ChatMessage(role=Role.SYSTEM, content="be helpful"),
            ChatMessage(role=Role.USER, content="hi"),
            ChatMessage(role=Role.ASSISTANT, content="hello"),
        ]
        results = chatmessages_to_langchain_messages(msgs)

        assert len(results) == 3
        assert isinstance(results[0], SystemMessage)
        assert isinstance(results[1], HumanMessage)
        assert isinstance(results[2], AIMessage)


def _make_lc_chunk(content="", tool_call_chunks=None, response_metadata=None, usage_metadata=None):
    return MagicMock(
        content=content,
        tool_call_chunks=tool_call_chunks or [],
        response_metadata=response_metadata,
        usage_metadata=usage_metadata,
        generation_info=None,
    )


def _make_adapter_with_chunks(chunks):
    mock_llm = MagicMock()

    async def mock_astream(*args, **kwargs):
        for c in chunks:
            yield c

    mock_llm.astream = mock_astream
    return LangChainLLMAdapter(mock_llm)


async def _collect_stream(adapter, prompt="test"):
    results = []
    async for chunk in adapter.stream_async(prompt):
        results.append(chunk)
    return results


class TestStreamingToolCalls:
    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        adapter = _make_adapter_with_chunks(
            [
                _make_lc_chunk(tool_call_chunks=[{"index": 0, "id": "call_abc", "name": "get_weather", "args": ""}]),
                _make_lc_chunk(tool_call_chunks=[{"index": 0, "args": '{"city"'}]),
                _make_lc_chunk(tool_call_chunks=[{"index": 0, "args": ':"Paris"}'}]),
                _make_lc_chunk(response_metadata={"finish_reason": "tool_calls"}),
            ]
        )

        results = await _collect_stream(adapter)

        assert results[0].delta_tool_calls is None
        assert results[1].delta_tool_calls is None
        assert results[2].delta_tool_calls is None
        final = results[3]
        assert final.finish_reason == "tool_calls"
        assert len(final.delta_tool_calls) == 1
        assert final.delta_tool_calls[0].id == "call_abc"
        assert final.delta_tool_calls[0].function.name == "get_weather"
        assert final.delta_tool_calls[0].function.arguments == {"city": "Paris"}

    @pytest.mark.asyncio
    async def test_parallel_tool_calls(self):
        adapter = _make_adapter_with_chunks(
            [
                _make_lc_chunk(
                    tool_call_chunks=[
                        {"index": 0, "id": "call_1", "name": "get_weather", "args": ""},
                        {"index": 1, "id": "call_2", "name": "get_time", "args": ""},
                    ]
                ),
                _make_lc_chunk(
                    tool_call_chunks=[
                        {"index": 0, "args": '{"city":"Paris"}'},
                        {"index": 1, "args": '{"city":"Paris"}'},
                    ]
                ),
                _make_lc_chunk(response_metadata={"finish_reason": "tool_calls"}),
            ]
        )

        results = await _collect_stream(adapter)

        final = results[-1]
        assert len(final.delta_tool_calls) == 2
        assert final.delta_tool_calls[0].function.name == "get_weather"
        assert final.delta_tool_calls[1].function.name == "get_time"
        assert final.delta_tool_calls[0].function.arguments == {"city": "Paris"}
        assert final.delta_tool_calls[1].function.arguments == {"city": "Paris"}

    @pytest.mark.asyncio
    async def test_invalid_json_falls_back_to_empty_dict(self):
        adapter = _make_adapter_with_chunks(
            [
                _make_lc_chunk(tool_call_chunks=[{"index": 0, "id": "call_x", "name": "broken", "args": ""}]),
                _make_lc_chunk(tool_call_chunks=[{"index": 0, "args": "{not valid json"}]),
                _make_lc_chunk(response_metadata={"finish_reason": "tool_calls"}),
            ]
        )

        results = await _collect_stream(adapter)

        assert results[-1].delta_tool_calls[0].function.arguments == {}

    @pytest.mark.asyncio
    async def test_text_only_stream_unaffected(self):
        adapter = _make_adapter_with_chunks(
            [
                _make_lc_chunk(content="Hello"),
                _make_lc_chunk(content=" world", response_metadata={"finish_reason": "stop"}),
            ]
        )

        results = await _collect_stream(adapter)

        assert len(results) == 2
        assert results[0].delta_content == "Hello"
        assert results[0].delta_tool_calls is None
        assert results[1].delta_content == " world"
        assert results[1].finish_reason == "stop"
        assert results[1].delta_tool_calls is None

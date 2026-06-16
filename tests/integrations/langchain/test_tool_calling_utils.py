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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from nemoguardrails.actions.llm.utils import (
    _extract_content,
    _store_tool_calls,
    get_and_clear_tool_calls_contextvar,
    llm_call,
)
from nemoguardrails.context import tool_calls_var
from nemoguardrails.exceptions import LLMCallException
from nemoguardrails.integrations.langchain.llm_adapter import LangChainLLMAdapter
from nemoguardrails.integrations.langchain.message_utils import dicts_to_messages
from nemoguardrails.rails.llm.llmrails import GenerationResponse
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallFunction


def test_get_and_clear_tool_calls_contextvar():
    test_tool_calls = [{"name": "test_func", "args": {}, "id": "call_123", "type": "tool_call"}]
    tool_calls_var.set(test_tool_calls)

    result = get_and_clear_tool_calls_contextvar()

    assert result == test_tool_calls
    assert tool_calls_var.get() is None


def test_get_and_clear_tool_calls_contextvar_empty():
    tool_calls_var.set(None)

    result = get_and_clear_tool_calls_contextvar()

    assert result is None


def test_dicts_to_messages_user():
    messages = [{"role": "user", "content": "Hello"}]

    result = dicts_to_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "Hello"


def test_dicts_to_messages_assistant():
    messages = [{"role": "assistant", "content": "Hi there"}]

    result = dicts_to_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], AIMessage)
    assert result[0].content == "Hi there"


def test_dicts_to_messages_bot():
    messages = [{"type": "bot", "content": "Hello from bot"}]

    result = dicts_to_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], AIMessage)
    assert result[0].content == "Hello from bot"


def test_dicts_to_messages_system():
    messages = [{"role": "system", "content": "You are a helpful assistant"}]

    result = dicts_to_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == "You are a helpful assistant"


def test_dicts_to_messages_tool():
    messages = [{"role": "tool", "content": "Tool result", "tool_call_id": "call_123"}]

    result = dicts_to_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], ToolMessage)
    assert result[0].content == "Tool result"
    assert result[0].tool_call_id == "call_123"


def test_dicts_to_messages_tool_no_id():
    messages = [{"role": "tool", "content": "Tool result"}]

    result = dicts_to_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], ToolMessage)
    assert result[0].content == "Tool result"
    assert result[0].tool_call_id == ""


def test_dicts_to_messages_mixed():
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User message"},
        {"type": "bot", "content": "Bot response"},
        {"role": "tool", "content": "Tool output", "tool_call_id": "call_456"},
    ]

    result = dicts_to_messages(messages)

    assert len(result) == 4
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)
    assert isinstance(result[3], ToolMessage)
    assert result[3].tool_call_id == "call_456"


def test_dicts_to_messages_unknown_type():
    messages = [{"role": "unknown", "content": "Unknown message"}]

    with pytest.raises(ValueError, match="Unknown message type: unknown"):
        dicts_to_messages(messages)


def test_store_tool_calls():
    response = LLMResponse(
        content="",
        tool_calls=[
            ToolCall(id="call_789", function=ToolCallFunction(name="another_func", arguments={})),
        ],
    )

    _store_tool_calls(response)

    result = tool_calls_var.get()
    assert result is not None
    assert len(result) == 1
    assert result[0]["function"]["name"] == "another_func"
    assert result[0]["id"] == "call_789"


def test_store_tool_calls_no_tool_calls():
    response = LLMResponse(content="no tools")

    _store_tool_calls(response)

    assert tool_calls_var.get() is None


def test_extract_content_with_content_attr():
    response = LLMResponse(content="Response content")

    result = _extract_content(response)

    assert result == "Response content"


@pytest.mark.asyncio
async def test_llm_call_with_string_prompt():
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "LLM response"
    mock_llm.ainvoke.return_value = mock_response

    wrapped = LangChainLLMAdapter(mock_llm)
    result = await llm_call(wrapped, "Test prompt")

    assert result.content == "LLM response"
    mock_llm.ainvoke.assert_called_once()
    call_args = mock_llm.ainvoke.call_args
    assert call_args[0][0] == "Test prompt"


@pytest.mark.asyncio
async def test_llm_call_with_message_list():
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "LLM response"
    mock_llm.ainvoke.return_value = mock_response

    wrapped = LangChainLLMAdapter(mock_llm)
    messages = [{"role": "user", "content": "Hello"}]
    result = await llm_call(wrapped, messages)

    assert result.content == "LLM response"
    mock_llm.ainvoke.assert_called_once()
    call_args = mock_llm.ainvoke.call_args
    assert len(call_args[0][0]) == 1
    assert isinstance(call_args[0][0][0], HumanMessage)


@pytest.mark.asyncio
async def test_llm_call_stores_tool_calls():
    mock_llm = AsyncMock()
    mock_response = AIMessage(
        content="Response with tools",
        tool_calls=[{"name": "test", "args": {}, "id": "call_test", "type": "tool_call"}],
    )
    mock_llm.ainvoke.return_value = mock_response

    wrapped = LangChainLLMAdapter(mock_llm)
    result = await llm_call(wrapped, "Test prompt")

    assert result.content == "Response with tools"
    stored = tool_calls_var.get()
    assert stored is not None
    assert len(stored) == 1
    assert stored[0]["function"]["name"] == "test"


@pytest.mark.asyncio
async def test_llm_call_with_llm_params():
    mock_llm = MagicMock()
    mock_bound_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "LLM response with params"

    mock_llm.bind = MagicMock(return_value=mock_bound_llm)
    mock_bound_llm.ainvoke.return_value = mock_response

    wrapped = LangChainLLMAdapter(mock_llm)
    llm_params = {"temperature": 0.5, "max_tokens": 100}
    result = await llm_call(wrapped, "Test prompt", llm_params=llm_params)

    assert result.content == "LLM response with params"
    mock_llm.bind.assert_called_once_with(**llm_params)
    mock_bound_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_llm_call_with_empty_llm_params():
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "LLM response"
    mock_llm.ainvoke.return_value = mock_response

    wrapped = LangChainLLMAdapter(mock_llm)
    result = await llm_call(wrapped, "Test prompt", llm_params={})

    assert result.content == "LLM response"
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_llm_call_with_none_llm_params():
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "LLM response no params"
    mock_llm.ainvoke.return_value = mock_response

    wrapped = LangChainLLMAdapter(mock_llm)
    result = await llm_call(wrapped, "Test prompt", llm_params=None)

    assert result.content == "LLM response no params"
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_llm_call_with_llm_params_temperature_max_tokens():
    mock_llm = MagicMock()
    mock_bound_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "Response with temp and tokens"

    mock_llm.bind = MagicMock(return_value=mock_bound_llm)
    mock_bound_llm.ainvoke.return_value = mock_response

    wrapped = LangChainLLMAdapter(mock_llm)
    llm_params = {"temperature": 0.8, "max_tokens": 50}
    result = await llm_call(wrapped, "Test prompt", llm_params=llm_params)

    assert result.content == "Response with temp and tokens"
    mock_llm.bind.assert_called_once_with(temperature=0.8, max_tokens=50)
    mock_bound_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_llm_call_with_none_llm_and_params():
    with pytest.raises(LLMCallException):
        await llm_call(None, "Test prompt", llm_params={"temperature": 0.5})


def test_generation_response_tool_calls_field():
    test_tool_calls = [{"name": "test_function", "args": {}, "id": "call_test", "type": "tool_call"}]

    response = GenerationResponse(response=[{"role": "assistant", "content": "Hello"}], tool_calls=test_tool_calls)

    assert response.tool_calls == test_tool_calls

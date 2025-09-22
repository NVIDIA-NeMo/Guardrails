# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    _convert_messages_to_langchain_format,
    _extract_content,
    _store_tool_calls,
    get_and_clear_tool_calls_contextvar,
    llm_call,
)
from nemoguardrails.context import tool_calls_var
from nemoguardrails.rails.llm.llmrails import GenerationResponse


def test_get_and_clear_tool_calls_contextvar():
    test_tool_calls = [
        {"name": "test_func", "args": {}, "id": "call_123", "type": "tool_call"}
    ]
    tool_calls_var.set(test_tool_calls)

    result = get_and_clear_tool_calls_contextvar()

    assert result == test_tool_calls
    assert tool_calls_var.get() is None


def test_get_and_clear_tool_calls_contextvar_empty():
    """Test that it returns None when no tool calls exist."""
    tool_calls_var.set(None)

    result = get_and_clear_tool_calls_contextvar()

    assert result is None


def test_convert_messages_to_langchain_format_user():
    """Test converting user messages to LangChain format."""
    messages = [{"role": "user", "content": "Hello"}]

    result = _convert_messages_to_langchain_format(messages)

    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "Hello"


def test_convert_messages_to_langchain_format_assistant():
    """Test converting assistant messages to LangChain format."""
    messages = [{"role": "assistant", "content": "Hi there"}]

    result = _convert_messages_to_langchain_format(messages)

    assert len(result) == 1
    assert isinstance(result[0], AIMessage)
    assert result[0].content == "Hi there"


def test_convert_messages_to_langchain_format_bot():
    """Test converting bot messages to LangChain format."""
    messages = [{"type": "bot", "content": "Hello from bot"}]

    result = _convert_messages_to_langchain_format(messages)

    assert len(result) == 1
    assert isinstance(result[0], AIMessage)
    assert result[0].content == "Hello from bot"


def test_convert_messages_to_langchain_format_system():
    """Test converting system messages to LangChain format."""
    messages = [{"role": "system", "content": "You are a helpful assistant"}]

    result = _convert_messages_to_langchain_format(messages)

    assert len(result) == 1
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == "You are a helpful assistant"


def test_convert_messages_to_langchain_format_tool():
    """Test converting tool messages to LangChain format."""
    messages = [{"role": "tool", "content": "Tool result", "tool_call_id": "call_123"}]

    result = _convert_messages_to_langchain_format(messages)

    assert len(result) == 1
    assert isinstance(result[0], ToolMessage)
    assert result[0].content == "Tool result"
    assert result[0].tool_call_id == "call_123"


def test_convert_messages_to_langchain_format_tool_no_id():
    """Test converting tool messages without tool_call_id."""
    messages = [{"role": "tool", "content": "Tool result"}]

    result = _convert_messages_to_langchain_format(messages)

    assert len(result) == 1
    assert isinstance(result[0], ToolMessage)
    assert result[0].content == "Tool result"
    assert result[0].tool_call_id == ""


def test_convert_messages_to_langchain_format_mixed():
    """Test converting mixed message types."""
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User message"},
        {"type": "bot", "content": "Bot response"},
        {"role": "tool", "content": "Tool output", "tool_call_id": "call_456"},
    ]

    result = _convert_messages_to_langchain_format(messages)

    assert len(result) == 4
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)
    assert isinstance(result[3], ToolMessage)
    assert result[3].tool_call_id == "call_456"


def test_convert_messages_to_langchain_format_unknown_type():
    """Test that unknown message types raise ValueError."""
    messages = [{"role": "unknown", "content": "Unknown message"}]

    with pytest.raises(ValueError, match="Unknown message type unknown"):
        _convert_messages_to_langchain_format(messages)


def test_store_tool_calls():
    """Test storing tool calls from response."""
    mock_response = MagicMock()
    test_tool_calls = [
        {"name": "another_func", "args": {}, "id": "call_789", "type": "tool_call"}
    ]
    mock_response.tool_calls = test_tool_calls

    _store_tool_calls(mock_response)

    assert tool_calls_var.get() == test_tool_calls


def test_store_tool_calls_no_tool_calls():
    """Test storing tool calls when response has no tool_calls attribute."""
    mock_response = MagicMock()
    del mock_response.tool_calls

    _store_tool_calls(mock_response)

    assert tool_calls_var.get() is None


def test_extract_content_with_content_attr():
    """Test extracting content from response with content attribute."""
    mock_response = MagicMock()
    mock_response.content = "Response content"

    result = _extract_content(mock_response)

    assert result == "Response content"


def test_extract_content_without_content_attr():
    """Test extracting content from response without content attribute."""
    mock_response = "Plain string response"

    result = _extract_content(mock_response)

    assert result == "Plain string response"


@pytest.mark.asyncio
async def test_llm_call_with_string_prompt():
    """Test llm_call with string prompt."""
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "LLM response"
    mock_llm.ainvoke.return_value = mock_response

    result = await llm_call(mock_llm, "Test prompt")

    assert result == "LLM response"
    mock_llm.ainvoke.assert_called_once()
    call_args = mock_llm.ainvoke.call_args
    assert call_args[0][0] == "Test prompt"


@pytest.mark.asyncio
async def test_llm_call_with_message_list():
    """Test llm_call with message list."""
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "LLM response"
    mock_llm.ainvoke.return_value = mock_response

    messages = [{"role": "user", "content": "Hello"}]
    result = await llm_call(mock_llm, messages)

    assert result == "LLM response"
    mock_llm.ainvoke.assert_called_once()
    call_args = mock_llm.ainvoke.call_args
    assert len(call_args[0][0]) == 1
    assert isinstance(call_args[0][0][0], HumanMessage)


@pytest.mark.asyncio
async def test_llm_call_stores_tool_calls():
    """Test that llm_call stores tool calls from response."""
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "Response with tools"
    test_tool_calls = [
        {"name": "test", "args": {}, "id": "call_test", "type": "tool_call"}
    ]
    mock_response.tool_calls = test_tool_calls
    mock_llm.ainvoke.return_value = mock_response

    result = await llm_call(mock_llm, "Test prompt")

    assert result == "Response with tools"
    assert tool_calls_var.get() == test_tool_calls


def test_generation_response_tool_calls_field():
    """Test that GenerationResponse can store tool calls."""
    test_tool_calls = [
        {"name": "test_function", "args": {}, "id": "call_test", "type": "tool_call"}
    ]

    response = GenerationResponse(
        response=[{"role": "assistant", "content": "Hello"}], tool_calls=test_tool_calls
    )

    assert response.tool_calls == test_tool_calls

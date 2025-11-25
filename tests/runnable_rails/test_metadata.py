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

"""Tests for metadata preservation in RunnableRails."""

from typing import List, Optional
from unittest.mock import Mock

import pytest
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails


class MetadataMockChatModel(BaseChatModel):
    """Mock chat model that returns AIMessage with full metadata for testing."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """Generate chat result with metadata."""

        ai_message = AIMessage(
            content="Test response from mock LLM",
            additional_kwargs={"custom_field": "custom_value"},
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "model_name": "test-model",
                "finish_reason": "stop",
            },
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            id="test-message-id",
            tool_calls=[
                {
                    "name": "test_tool",
                    "args": {"arg1": "value1"},
                    "id": "tool_call_id",
                    "type": "tool_call",
                }
            ],
        )

        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """Async generate chat result with metadata."""
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "metadata_mock"


@pytest.fixture(autouse=True)
def metadata_mock_provider():
    """Fixture that registers mock chat provider for testing."""
    from nemoguardrails.llm.providers import register_chat_provider

    register_chat_provider("metadata_mock_llm", MetadataMockChatModel)

    yield

    from nemoguardrails.llm.providers.providers import _chat_providers

    _chat_providers.pop("metadata_mock_llm", None)


@pytest.fixture
def mock_rails_config():
    """Create a mock RailsConfig for testing."""
    config = RailsConfig(
        models=[{"type": "main", "engine": "metadata_mock_llm", "model": "test-model"}],
        rails={
            "input": {"flows": []},
            "dialog": {"flows": []},
            "output": {"flows": []},
        },
    )
    return config


@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns structured responses."""
    mock_llm = Mock()

    mock_response = AIMessage(
        content="Test response",
        additional_kwargs={"custom_field": "custom_value"},
        response_metadata={
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "model_name": "test-model",
            "finish_reason": "stop",
        },
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        id="test-message-id",
        tool_calls=[
            {
                "name": "test_tool",
                "args": {"arg1": "value1"},
                "id": "tool_call_id",
                "type": "tool_call",
            }
        ],
    )

    mock_llm.invoke = Mock(return_value=mock_response)
    mock_llm.ainvoke = Mock(return_value=mock_response)

    return mock_llm


@pytest.fixture
def runnable_rails_with_metadata(mock_rails_config, mock_llm):
    """Create RunnableRails instance that should preserve metadata."""

    mock_rails = Mock()

    mock_generation_response = Mock()
    mock_generation_response.response = "Test response from rails"
    mock_generation_response.output_data = {}
    mock_generation_response.tool_calls = [
        {
            "name": "test_tool",
            "args": {"arg1": "value1"},
            "id": "tool_call_id",
            "type": "tool_call",
        }
    ]
    mock_generation_response.llm_metadata = {
        "additional_kwargs": {"custom_field": "custom_value"},
        "response_metadata": {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "model_name": "test-model",
            "finish_reason": "stop",
        },
        "usage_metadata": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "id": "test-message-id",
        "tool_calls": [
            {
                "name": "test_tool",
                "args": {"arg1": "value1"},
                "id": "tool_call_id",
                "type": "tool_call",
            }
        ],
    }

    mock_rails.generate = Mock(return_value=mock_generation_response)

    runnable_rails = RunnableRails(config=mock_rails_config, passthrough=True)
    runnable_rails.rails = mock_rails

    return runnable_rails


class TestMetadataPreservation:
    """Test cases for metadata preservation in RunnableRails."""

    def test_metadata_preserved_with_chat_prompt_value(self, runnable_rails_with_metadata):
        """Test that metadata is preserved with ChatPromptValue input."""
        prompt = ChatPromptTemplate.from_messages([("human", "Test message")])
        chat_prompt_value = prompt.format_prompt()

        result = runnable_rails_with_metadata.invoke(chat_prompt_value)

        assert isinstance(result, AIMessage)
        assert result.content == "Test response from rails"
        assert result.additional_kwargs == {"custom_field": "custom_value"}
        assert result.response_metadata == {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "model_name": "test-model",
            "finish_reason": "stop",
        }
        assert result.usage_metadata == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        assert result.id == "test-message-id"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "test_tool"

    def test_metadata_preserved_with_base_message(self, runnable_rails_with_metadata):
        """Test that metadata is preserved with BaseMessage input."""
        message = HumanMessage(content="Test message")

        result = runnable_rails_with_metadata.invoke(message)

        assert isinstance(result, AIMessage)
        assert result.content == "Test response from rails"
        assert result.additional_kwargs == {"custom_field": "custom_value"}
        assert result.response_metadata["model_name"] == "test-model"
        assert result.usage_metadata["total_tokens"] == 15

    def test_metadata_preserved_with_message_list(self, runnable_rails_with_metadata):
        """Test that metadata is preserved with list of messages input."""
        messages = [HumanMessage(content="Test message")]

        result = runnable_rails_with_metadata.invoke(messages)

        assert isinstance(result, AIMessage)
        assert result.content == "Test response from rails"
        assert result.additional_kwargs == {"custom_field": "custom_value"}
        assert result.usage_metadata is not None

    def test_metadata_preserved_with_dict_input_base_message(self, runnable_rails_with_metadata):
        """Test that metadata is preserved with dictionary input containing BaseMessage."""
        input_dict = {"input": HumanMessage(content="Test message")}

        result = runnable_rails_with_metadata.invoke(input_dict)

        assert isinstance(result, dict)
        assert "output" in result
        ai_message = result["output"]
        assert isinstance(ai_message, AIMessage)
        assert ai_message.content == "Test response from rails"
        assert ai_message.additional_kwargs == {"custom_field": "custom_value"}

    def test_metadata_preserved_with_dict_input_message_list(self, runnable_rails_with_metadata):
        """Test that metadata is preserved with dictionary input containing message list."""
        input_dict = {"input": [HumanMessage(content="Test message")]}

        result = runnable_rails_with_metadata.invoke(input_dict)

        assert isinstance(result, dict)
        assert "output" in result
        ai_message = result["output"]
        assert isinstance(ai_message, AIMessage)
        assert ai_message.usage_metadata["total_tokens"] == 15

    def test_content_not_overwritten_by_metadata(self, runnable_rails_with_metadata):
        """Test that rails-processed content is not overwritten by metadata content."""
        prompt = ChatPromptTemplate.from_messages([("human", "Test message")])
        chat_prompt_value = prompt.format_prompt()

        result = runnable_rails_with_metadata.invoke(chat_prompt_value)

        assert result.content == "Test response from rails"

    def test_tool_calls_precedence(self, mock_rails_config):
        """Test tool_calls precedence: passed tool_calls should override metadata tool_calls when both exist."""

        mock_rails = Mock()
        mock_generation_response = Mock()
        mock_generation_response.response = "Test response"
        mock_generation_response.output_data = {}
        mock_generation_response.tool_calls = [
            {"name": "rails_tool", "args": {"param": "rails_value"}, "id": "rails_id"}
        ]
        mock_generation_response.llm_metadata = {
            "tool_calls": [
                {
                    "name": "metadata_tool",
                    "args": {"param": "metadata_value"},
                    "id": "metadata_id",
                }
            ],
            "additional_kwargs": {},
            "response_metadata": {},
        }

        mock_rails.generate = Mock(return_value=mock_generation_response)

        runnable_rails = RunnableRails(config=mock_rails_config, passthrough=True)
        runnable_rails.rails = mock_rails

        prompt = ChatPromptTemplate.from_messages([("human", "Test message")])
        result = runnable_rails.invoke(prompt.format_prompt())

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "rails_tool"

    def test_no_metadata_fallback_behavior(self, mock_rails_config):
        """Test fallback behavior when no metadata is available."""

        mock_rails = Mock()
        mock_generation_response = Mock()
        mock_generation_response.response = "Test response"
        mock_generation_response.output_data = {}
        mock_generation_response.tool_calls = None
        mock_generation_response.llm_metadata = None

        mock_rails.generate = Mock(return_value=mock_generation_response)

        runnable_rails = RunnableRails(config=mock_rails_config, passthrough=True)
        runnable_rails.rails = mock_rails

        prompt = ChatPromptTemplate.from_messages([("human", "Test message")])
        result = runnable_rails.invoke(prompt.format_prompt())

        assert isinstance(result, AIMessage)
        assert result.content == "Test response"
        assert result.additional_kwargs == {}
        assert result.response_metadata == {}
        assert result.tool_calls == []

    def test_partial_metadata(self, mock_rails_config):
        """Test behavior with partial metadata (some fields missing)."""

        mock_rails = Mock()
        mock_generation_response = Mock()
        mock_generation_response.response = "Test response"
        mock_generation_response.output_data = {}
        mock_generation_response.tool_calls = None
        mock_generation_response.llm_metadata = {
            "additional_kwargs": {"custom_field": "value"},
        }

        mock_rails.generate = Mock(return_value=mock_generation_response)

        runnable_rails = RunnableRails(config=mock_rails_config, passthrough=True)
        runnable_rails.rails = mock_rails

        prompt = ChatPromptTemplate.from_messages([("human", "Test message")])
        result = runnable_rails.invoke(prompt.format_prompt())

        assert isinstance(result, AIMessage)
        assert result.content == "Test response"
        assert result.additional_kwargs == {"custom_field": "value"}
        assert result.response_metadata is None or result.response_metadata == {}

    def test_streaming_metadata_preservation(self, mock_rails_config):
        """Test that streaming preserves metadata in chunks."""
        from unittest.mock import AsyncMock

        mock_rails = AsyncMock()
        mock_generation_response = Mock()
        mock_generation_response.response = "Streaming response"
        mock_generation_response.output_data = {}
        mock_generation_response.tool_calls = None
        mock_generation_response.llm_metadata = {
            "additional_kwargs": {"finish_reason": "stop"},
            "response_metadata": {"model_name": "test-model"},
        }

        async def mock_stream(*args, **kwargs):
            chunks = [
                {
                    "text": "Hello ",
                    "generation_info": {"model": "test-model", "finish_reason": "stop"},
                },
                {
                    "text": "world!",
                    "generation_info": {"model": "test-model", "finish_reason": "stop"},
                },
            ]
            for chunk in chunks:
                yield chunk

        mock_rails.stream_async = mock_stream

        runnable_rails = RunnableRails(config=mock_rails_config, passthrough=True)
        runnable_rails.rails = mock_rails

        chunks = list(runnable_rails.stream("Test input"))

        assert len(chunks) == 2
        for chunk in chunks:
            assert hasattr(chunk, "content")
            assert hasattr(chunk, "additional_kwargs") or hasattr(chunk, "model")
            assert hasattr(chunk, "response_metadata") or hasattr(chunk, "finish_reason")

    @pytest.mark.asyncio
    async def test_async_streaming_metadata_preservation(self, mock_rails_config):
        """Test that async streaming preserves metadata in chunks."""
        from unittest.mock import AsyncMock

        mock_rails = AsyncMock()

        async def mock_stream(*args, **kwargs):
            chunks = [
                {"text": "Async ", "generation_info": {"model": "test-model"}},
                {"text": "stream!", "generation_info": {"model": "test-model"}},
            ]
            for chunk in chunks:
                yield chunk

        mock_rails.stream_async = mock_stream

        runnable_rails = RunnableRails(config=mock_rails_config, passthrough=True)
        runnable_rails.rails = mock_rails

        chunks = []
        async for chunk in runnable_rails.astream("Test input"):
            chunks.append(chunk)

        assert len(chunks) == 2
        for chunk in chunks:
            assert hasattr(chunk, "content")
            assert hasattr(chunk, "additional_kwargs") or hasattr(chunk, "model")

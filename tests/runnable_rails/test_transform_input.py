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

"""Tests for RunnableRails input transformation methods."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from tests.utils import FakeLLM


@pytest.fixture
def rails():
    """Create a RunnableRails instance for testing."""
    config = RailsConfig.from_content(config={"models": []})
    llm = FakeLLM(responses=["test response"])
    return RunnableRails(config, llm=llm)


@pytest.fixture
def rails_passthrough():
    """Create a RunnableRails instance with passthrough mode and runnable."""
    config = RailsConfig.from_content(config={"models": []})
    llm = FakeLLM(responses=["test response"])

    from langchain_core.runnables import RunnableLambda

    mock_runnable = RunnableLambda(lambda x: "Mock response")

    return RunnableRails(config, llm=llm, passthrough=True, runnable=mock_runnable)


def test_transform_string_input(rails):
    """Test transformation of string input."""
    result = rails._transform_input_to_rails_format("Hello world")
    expected = [{"role": "user", "content": "Hello world"}]
    assert result == expected


def test_transform_chat_prompt_value(rails):
    """Test transformation of ChatPromptValue input."""
    messages = [
        SystemMessage(content="You are helpful"),
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
    ]
    chat_prompt = ChatPromptValue(messages=messages)

    result = rails._transform_input_to_rails_format(chat_prompt)
    expected = [
        {
            "role": "system",
            "content": "You are helpful",
            "additional_kwargs": {},
            "response_metadata": {},
        },
        {
            "role": "user",
            "content": "Hello",
            "additional_kwargs": {},
            "response_metadata": {},
        },
        {
            "role": "assistant",
            "content": "Hi there",
            "additional_kwargs": {},
            "response_metadata": {},
            "tool_calls": [],
            "invalid_tool_calls": [],
        },
    ]
    assert result == expected


def test_transform_string_prompt_value(rails):
    """Test transformation of StringPromptValue input."""
    string_prompt = StringPromptValue(text="What is AI?")

    result = rails._transform_input_to_rails_format(string_prompt)
    expected = [{"role": "user", "content": "What is AI?"}]
    assert result == expected


def test_transform_dict_input_with_input_key(rails):
    """Test transformation of dict input with 'input' key."""
    input_dict = {"input": "Tell me about Python"}

    result = rails._transform_input_to_rails_format(input_dict)
    expected = [{"role": "user", "content": "Tell me about Python"}]
    assert result == expected


def test_transform_dict_input_with_custom_input_key(rails):
    """Test transformation of dict input with custom input key."""
    rails.passthrough_user_input_key = "question"
    input_dict = {"question": "What is the weather?"}

    result = rails._transform_input_to_rails_format(input_dict)
    expected = [{"role": "user", "content": "What is the weather?"}]
    assert result == expected


def test_transform_dict_input_with_context(rails):
    """Test transformation of dict input with context."""
    input_dict = {
        "input": "Hello",
        "context": {"user_name": "John", "session_id": "123"},
    }

    result = rails._transform_input_to_rails_format(input_dict)
    expected = [
        {"role": "context", "content": {"user_name": "John", "session_id": "123"}},
        {"role": "user", "content": "Hello"},
    ]
    assert result == expected


def test_transform_dict_input_with_message_list(rails):
    """Test transformation of dict input with list of dict messages."""
    input_dict = {
        "input": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    }

    result = rails._transform_input_to_rails_format(input_dict)
    expected = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    assert result == expected


def test_transform_list_of_base_messages(rails):
    """Test transformation of list of BaseMessage objects."""
    messages = [
        HumanMessage(content="What is Python?"),
        AIMessage(content="Python is a programming language"),
    ]

    result = rails._transform_input_to_rails_format(messages)
    expected = [
        {
            "role": "user",
            "content": "What is Python?",
            "additional_kwargs": {},
            "response_metadata": {},
        },
        {
            "role": "assistant",
            "content": "Python is a programming language",
            "additional_kwargs": {},
            "response_metadata": {},
            "tool_calls": [],
            "invalid_tool_calls": [],
        },
    ]
    assert result == expected


def test_transform_single_human_message(rails):
    """Test transformation of single HumanMessage."""
    message = HumanMessage(content="Hello there")

    result = rails._transform_input_to_rails_format(message)
    expected = [
        {
            "role": "user",
            "content": "Hello there",
            "additional_kwargs": {},
            "response_metadata": {},
        }
    ]
    assert result == expected


def test_transform_single_ai_message(rails):
    """Test transformation of single AIMessage."""
    message = AIMessage(content="Hello back")

    result = rails._transform_input_to_rails_format(message)
    expected = [
        {
            "role": "assistant",
            "content": "Hello back",
            "additional_kwargs": {},
            "response_metadata": {},
            "tool_calls": [],
            "invalid_tool_calls": [],
        }
    ]
    assert result == expected


def test_transform_passthrough_mode_string(rails_passthrough):
    """Test transformation in passthrough mode with string input."""
    result = rails_passthrough._transform_input_to_rails_format("Hello world")

    assert len(result) == 2
    assert result[0]["role"] == "context"
    assert result[0]["content"]["passthrough_input"] == "Hello world"
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "Hello world"


def test_transform_passthrough_mode_dict(rails_passthrough):
    """Test transformation in passthrough mode with dict input."""
    input_dict = {"input": "Test message", "param1": "value1"}
    result = rails_passthrough._transform_input_to_rails_format(input_dict)

    assert len(result) == 2
    assert result[0]["role"] == "context"
    assert result[0]["content"]["passthrough_input"] == input_dict
    assert result[0]["content"]["input"] == "Test message"
    assert result[0]["content"]["param1"] == "value1"
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "Test message"


def test_transform_invalid_dict_input(rails):
    """Test transformation of dict without required keys raises exception."""
    input_dict = {"wrong_key": "some value"}

    with pytest.raises(Exception) as excinfo:
        rails._transform_input_to_rails_format(input_dict)

    assert "Expected 'input' or 'input' key in input dictionary" in str(excinfo.value)


def test_transform_invalid_context_type(rails):
    """Test transformation with invalid context type raises exception."""
    input_dict = {
        "input": "Hello",
        "context": "should be dict",
    }

    with pytest.raises(ValueError) as excinfo:
        rails._transform_input_to_rails_format(input_dict)

    assert "must be a dict" in str(excinfo.value)


def test_transform_unsupported_input_type(rails):
    """Test transformation of unsupported input type raises exception."""
    with pytest.raises(Exception) as excinfo:
        rails._transform_input_to_rails_format(12345)

    assert "Unsupported input type" in str(excinfo.value)

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

"""Tests for RunnableRails output formatting methods."""

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.runnables import RunnableLambda

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
    mock_runnable = RunnableLambda(lambda x: {"result": "Mock response"})
    return RunnableRails(config, llm=llm, passthrough=True, runnable=mock_runnable)


def test_format_output_string_input_string_result(rails):
    """Test formatting with string input and string result."""
    input_str = "What is AI?"
    result = "AI is artificial intelligence."
    context = {}

    formatted = rails._format_output(input_str, result, context)
    assert formatted == "AI is artificial intelligence."


def test_format_output_string_input_dict_result(rails):
    """Test formatting with string input and dict result."""
    input_str = "What is AI?"
    result = {"content": "AI is artificial intelligence."}
    context = {}

    formatted = rails._format_output(input_str, result, context)
    assert formatted == "AI is artificial intelligence."


def test_format_output_chat_prompt_value_input(rails):
    """Test formatting with ChatPromptValue input."""
    messages = [HumanMessage(content="What is AI?")]
    input_chat = ChatPromptValue(messages=messages)
    result = {"content": "AI is artificial intelligence."}
    context = {}

    formatted = rails._format_output(input_chat, result, context)
    assert isinstance(formatted, AIMessage)
    assert formatted.content == "AI is artificial intelligence."


def test_format_output_string_prompt_value_input(rails):
    """Test formatting with StringPromptValue input."""
    input_prompt = StringPromptValue(text="What is AI?")
    result = {"content": "AI is artificial intelligence."}
    context = {}

    formatted = rails._format_output(input_prompt, result, context)
    assert formatted == "AI is artificial intelligence."


def test_format_output_human_message_input(rails):
    """Test formatting with HumanMessage input."""
    input_msg = HumanMessage(content="What is AI?")
    result = {"content": "AI is artificial intelligence."}
    context = {}

    formatted = rails._format_output(input_msg, result, context)
    assert isinstance(formatted, AIMessage)
    assert formatted.content == "AI is artificial intelligence."


def test_format_output_list_messages_input(rails):
    """Test formatting with list of messages input."""
    input_list = [HumanMessage(content="What is AI?")]
    result = {"content": "AI is artificial intelligence."}
    context = {}

    formatted = rails._format_output(input_list, result, context)
    assert isinstance(formatted, AIMessage)
    assert formatted.content == "AI is artificial intelligence."


def test_format_output_dict_string_input(rails):
    """Test formatting with dict input containing string."""
    input_dict = {"input": "What is AI?"}
    result = {"content": "AI is artificial intelligence."}
    context = {}

    formatted = rails._format_output(input_dict, result, context)
    assert isinstance(formatted, dict)
    assert formatted["output"] == "AI is artificial intelligence."


def test_format_output_dict_message_list_input(rails):
    """Test formatting with dict input containing message list."""
    input_dict = {"input": [{"role": "user", "content": "What is AI?"}]}
    result = {"content": "AI is artificial intelligence."}
    context = {}

    formatted = rails._format_output(input_dict, result, context)
    assert isinstance(formatted, dict)
    assert formatted["output"] == {
        "role": "assistant",
        "content": "AI is artificial intelligence.",
    }


def test_format_output_dict_base_message_list_input(rails):
    """Test formatting with dict input containing BaseMessage list."""
    input_dict = {"input": [HumanMessage(content="What is AI?")]}
    result = {"content": "AI is artificial intelligence."}
    context = {}

    formatted = rails._format_output(input_dict, result, context)
    assert isinstance(formatted, dict)
    assert "output" in formatted
    assert isinstance(formatted["output"], AIMessage)
    assert formatted["output"].content == "AI is artificial intelligence."


def test_format_output_dict_base_message_input(rails):
    """Test formatting with dict input containing BaseMessage."""
    input_dict = {"input": HumanMessage(content="What is AI?")}
    result = {"content": "AI is artificial intelligence."}
    context = {}

    formatted = rails._format_output(input_dict, result, context)
    assert isinstance(formatted, dict)
    assert "output" in formatted
    assert isinstance(formatted["output"], AIMessage)
    assert formatted["output"].content == "AI is artificial intelligence."


def test_format_output_custom_output_key(rails):
    """Test formatting with custom output key."""
    rails.passthrough_bot_output_key = "answer"
    input_dict = {"input": "What is AI?"}
    result = {"content": "AI is artificial intelligence."}
    context = {}

    formatted = rails._format_output(input_dict, result, context)
    assert isinstance(formatted, dict)
    assert formatted["answer"] == "AI is artificial intelligence."


def test_format_output_passthrough_mode_string_output(rails_passthrough):
    """Test formatting in passthrough mode with string output."""
    input_str = "What is AI?"
    result = "AI is artificial intelligence."
    context = {"passthrough_output": "Mock passthrough response"}

    formatted = rails_passthrough._format_output(input_str, result, context)
    assert formatted == "AI is artificial intelligence."


def test_format_output_passthrough_mode_dict_output(rails_passthrough):
    """Test formatting in passthrough mode with dict output."""
    input_str = "What is AI?"
    result = "AI is artificial intelligence."
    context = {"passthrough_output": {"result": "Mock response"}}

    formatted = rails_passthrough._format_output(input_str, result, context)
    assert isinstance(formatted, dict)
    assert formatted["output"] == "AI is artificial intelligence."


def test_format_output_passthrough_mode_no_passthrough_output(rails_passthrough):
    """Test formatting in passthrough mode when no passthrough output."""
    input_str = "What is AI?"
    result = {"content": "AI is artificial intelligence."}
    context = {}

    formatted = rails_passthrough._format_output(input_str, result, context)
    assert isinstance(formatted, dict)
    assert formatted["output"] == "AI is artificial intelligence."


def test_format_output_list_result_takes_first(rails):
    """Test that list results take the first item."""
    input_str = "What is AI?"
    result = [{"content": "First response"}, {"content": "Second response"}]
    context = {}

    formatted = rails._format_output(input_str, result, context)
    assert formatted == "First response"


def test_format_output_bot_message_context_override(rails_passthrough):
    """Test that bot_message in context overrides result in passthrough mode."""
    input_str = "What is AI?"
    result = {"content": "Original response"}
    context = {
        "bot_message": "Override response",
        "passthrough_output": {"result": "Mock"},
    }

    formatted = rails_passthrough._format_output(input_str, result, context)
    assert isinstance(formatted, dict)
    assert formatted["output"] == "Override response"


def test_format_output_unsupported_input_type(rails):
    """Test formatting with unsupported input type raises error."""
    input_unsupported = 12345
    result = {"content": "Response"}
    context = {}

    with pytest.raises(ValueError) as excinfo:
        rails._format_output(input_unsupported, result, context)

    assert "Unexpected input type" in str(excinfo.value)

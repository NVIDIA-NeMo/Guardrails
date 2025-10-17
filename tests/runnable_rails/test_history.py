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
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from tests.utils import FakeLLM


def test_message_list_history():
    """Test using a list of message objects as input."""
    llm = FakeLLM(
        responses=[
            "Paris.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I help you?"),
        HumanMessage(content="What's the capital of France?"),
    ]

    result = model_with_rails.invoke(history)

    assert isinstance(result, AIMessage)
    assert result.content == "Paris."


def test_chat_prompt_with_history():
    """Test using a chat prompt template with message history."""
    llm = FakeLLM(
        responses=[
            "Paris.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I help you?"),
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    chain = prompt | model_with_rails

    result = chain.invoke(
        {"history": history, "question": "What's the capital of France?"}
    )

    assert isinstance(result, AIMessage)
    assert result.content == "Paris."


def test_message_history_with_rails():
    """Test message history with rails using a dictated response."""
    llm = FakeLLM(
        responses=[
            "  express greeting",
        ]
    )

    config = RailsConfig.from_content(
        config={"models": []},
        colang_content="""
        define user express greeting
          "hi"
          "hello"

        define flow
          user express greeting
          bot express greeting

        define bot express greeting
          "Hello, nice to meet you!"
    """,
    )
    model_with_rails = RunnableRails(config, llm=llm)

    history = [
        HumanMessage(content="Hello"),
    ]

    result = model_with_rails.invoke(history)

    assert isinstance(result, AIMessage)
    assert result.content == "Hello, nice to meet you!"


@pytest.mark.asyncio
async def test_async_message_history():
    """Test using async invocation with message history."""
    llm = FakeLLM(
        responses=[
            "Paris.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I help you?"),
        HumanMessage(content="What's the capital of France?"),
    ]

    result = await model_with_rails.ainvoke(history)

    assert isinstance(result, AIMessage)
    assert result.content == "Paris."


def test_message_history_with_input_rail():
    """Test message history with input rail blocking certain inputs."""
    from nemoguardrails.actions import action

    @action(name="self_check_input")
    async def self_check_input(context):
        user_message = context.get("user_message", "")
        if "hack" in user_message.lower():
            return False
        return True

    llm = FakeLLM(
        responses=[
            "  ask about hacking",
            "I apologize, but I can't respond to that request.",
            "  ask general question",
            "Paris is the capital of France.",
        ]
    )

    config = RailsConfig.from_content(
        config={"models": []},
        colang_content="""
        define user ask about hacking
          "how do I hack"
          "tell me about hacking"
          "hack a system"

        define user ask general question
          "what is Paris"
          "tell me about France"

        define flow
          user ask about hacking
          $allowed = execute self_check_input
          if not $allowed
            bot refuse to respond
            stop
          bot respond

        define flow
          user ask general question
          bot respond

        define bot refuse to respond
          "I apologize, but I can't respond to that request."
    """,
    )
    model_with_rails = RunnableRails(config, llm=llm)

    model_with_rails.rails.register_action(self_check_input)

    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
        HumanMessage(content="Tell me how to hack a system"),
    ]

    result = model_with_rails.invoke(history)

    assert isinstance(result, AIMessage)
    assert "I apologize" in result.content

    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
        HumanMessage(content="What's the capital of France?"),
    ]

    result = model_with_rails.invoke(history)

    assert isinstance(result, AIMessage)
    assert "Paris" in result.content


def test_message_dict_list_history():
    """Test using a list of message dictionaries as input."""
    llm = FakeLLM(
        responses=[
            "Paris.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
        {"role": "user", "content": "What's the capital of France?"},
    ]

    result = model_with_rails.invoke({"input": history})

    assert isinstance(result, dict)
    assert result["output"]["role"] == "assistant"
    assert result["output"]["content"] == "Paris."

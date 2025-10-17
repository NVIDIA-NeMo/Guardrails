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

"""
Tests for basic RunnableRails operations (invoke, async, batch, stream).
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from tests.utils import FakeLLM


def test_runnable_rails_basic():
    """Test basic functionality of updated RunnableRails."""
    llm = FakeLLM(
        responses=[
            "Hello there! How can I help you today?",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    result = model_with_rails.invoke("Hi there")

    assert isinstance(result, str)
    assert "Hello there" in result


@pytest.mark.asyncio
async def test_runnable_rails_async():
    """Test async functionality of updated RunnableRails."""
    llm = FakeLLM(
        responses=[
            "Hello there! How can I help you today?",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    result = await model_with_rails.ainvoke("Hi there")

    assert isinstance(result, str)
    assert "Hello there" in result


def test_runnable_rails_batch():
    """Test batch functionality of updated RunnableRails."""
    llm = FakeLLM(
        responses=[
            "Response 1",
            "Response 2",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    results = model_with_rails.batch(["Question 1", "Question 2"])

    assert len(results) == 2
    assert results[0] == "Response 1"
    assert results[1] == "Response 2"


def test_updated_runnable_rails_stream():
    """Test streaming functionality of updated RunnableRails."""
    llm = FakeLLM(
        responses=[
            "Hello there!",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    chunks = []
    for chunk in model_with_rails.stream("Hi there"):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].content == "Hello "
    assert chunks[1].content == "there!"


def test_runnable_rails_with_message_history():
    """Test handling of message history with updated RunnableRails."""
    llm = FakeLLM(
        responses=[
            "Yes, Paris is the capital of France.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
        HumanMessage(content="What's the capital of France?"),
    ]

    result = model_with_rails.invoke(history)

    assert isinstance(result, AIMessage)
    assert "Paris" in result.content


def test_runnable_rails_with_chat_template():
    """Test updated RunnableRails with chat templates."""
    llm = FakeLLM(
        responses=[
            "Yes, Paris is the capital of France.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    chain = prompt | model_with_rails

    result = chain.invoke(
        {
            "history": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
            ],
            "question": "What's the capital of France?",
        }
    )

    assert isinstance(result, AIMessage)
    assert "Paris" in result.content

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

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from tests.utils import FakeLLM


def test_batch_processing():
    """Test batch processing of multiple inputs."""
    llm = FakeLLM(
        responses=[
            "Paris.",
            "Rome.",
            "Berlin.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    inputs = [
        "What's the capital of France?",
        "What's the capital of Italy?",
        "What's the capital of Germany?",
    ]

    results = model_with_rails.batch(inputs)

    assert len(results) == 3
    assert results[0] == "Paris."
    assert results[1] == "Rome."
    assert results[2] == "Berlin."


@pytest.mark.asyncio
async def test_abatch_processing():
    """Test async batch processing of multiple inputs."""
    llm = FakeLLM(
        responses=[
            "Paris.",
            "Rome.",
            "Berlin.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    inputs = [
        "What's the capital of France?",
        "What's the capital of Italy?",
        "What's the capital of Germany?",
    ]

    results = await model_with_rails.abatch(inputs)

    assert len(results) == 3
    assert results[0] == "Paris."
    assert results[1] == "Rome."
    assert results[2] == "Berlin."


def test_batch_with_different_input_types():
    """Test batch processing with different input types."""
    llm = FakeLLM(
        responses=[
            "Paris.",
            "Rome.",
            "Berlin.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    inputs = [
        "What's the capital of France?",
        HumanMessage(content="What's the capital of Italy?"),
        {"input": "What's the capital of Germany?"},
    ]

    results = model_with_rails.batch(inputs)

    assert len(results) == 3
    assert results[0] == "Paris."
    assert isinstance(results[1], AIMessage)
    assert results[1].content == "Rome."
    assert isinstance(results[2], dict)
    assert results[2]["output"] == "Berlin."


def test_stream_output():
    """Test streaming output (simplified for now)."""
    llm = FakeLLM(
        responses=[
            "Paris.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    chunks = []
    for chunk in model_with_rails.stream("What's the capital of France?"):
        chunks.append(chunk)

    # Currently, stream just yields the full response as a single chunk
    assert len(chunks) == 1
    assert chunks[0].content == "Paris."


@pytest.mark.asyncio
async def test_astream_output():
    """Test async streaming output (simplified for now)."""
    llm = FakeLLM(
        responses=[
            "hello what can you do?",
        ],
        streaming=True,
    )
    config = RailsConfig.from_content(config={"models": [], "streaming": True})
    model_with_rails = RunnableRails(config, llm=llm)

    # Collect all chunks from the stream
    chunks = []
    async for chunk in model_with_rails.astream("What's the capital of France?"):
        chunks.append(chunk)

    # Stream should yield individual word chunks
    assert len(chunks) == 5
    assert chunks[0].content == "hello "
    assert chunks[1].content == "what "
    assert chunks[2].content == "can "
    assert chunks[3].content == "you "
    assert chunks[4].content == "do?"

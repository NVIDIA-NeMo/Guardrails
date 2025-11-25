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
Tests for streaming functionality in RunnableRails.
"""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.prompt_values import StringPromptValue
from langchain_core.runnables import RunnablePassthrough

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from tests.utils import FakeLLM


class StreamingFakeLLM(FakeLLM):
    """A fake LLM that supports streaming by breaking responses into tokens."""

    def __init__(self, responses, **kwargs):
        super().__init__(responses=responses, **kwargs)
        self.streaming = True

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        """Stream the response by breaking it into tokens."""
        response = self._call(messages, stop, run_manager, **kwargs)
        tokens = response.split()
        for i, token in enumerate(tokens):
            if i == 0:
                yield token
            else:
                yield " " + token

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        """Async stream the response by breaking it into tokens."""
        response = await self._acall(messages, stop, run_manager, **kwargs)
        tokens = response.split()
        for i, token in enumerate(tokens):
            if i == 0:
                yield token
            else:
                yield " " + token


def test_runnable_rails_basic_streaming():
    """Test basic synchronous streaming functionality."""
    llm = StreamingFakeLLM(responses=["Hello there! How can I help you today?"])
    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config, llm=llm)

    chunks = []
    for chunk in rails.stream("Hi there"):
        chunks.append(chunk)

    assert len(chunks) > 1

    full_content = "".join(chunk if isinstance(chunk, str) else chunk.content for chunk in chunks)
    assert "Hello there!" in full_content


@pytest.mark.asyncio
async def test_runnable_rails_async_streaming():
    """Test asynchronous streaming functionality."""
    llm = StreamingFakeLLM(responses=["Hello there! How can I help you?"])
    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config, llm=llm)

    chunks = []
    async for chunk in rails.astream("Hi there"):
        chunks.append(chunk)

    assert len(chunks) > 1

    full_content = "".join(chunk if isinstance(chunk, str) else chunk.content for chunk in chunks)
    assert "Hello there!" in full_content


def test_runnable_rails_message_streaming():
    """Test streaming with message inputs and outputs."""
    llm = StreamingFakeLLM(responses=["Hello there! How can I help you?"])
    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config, llm=llm)

    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
        HumanMessage(content="How are you?"),
    ]

    chunks = []
    for chunk in rails.stream(history):
        chunks.append(chunk)

    assert len(chunks) > 1

    for chunk in chunks:
        if hasattr(chunk, "content"):
            from langchain_core.messages import AIMessageChunk

            assert isinstance(chunk, AIMessageChunk)

    full_content = "".join(chunk.content for chunk in chunks if hasattr(chunk, "content"))
    assert "Hello there!" in full_content


def test_runnable_rails_dict_streaming():
    """Test streaming with dictionary inputs and outputs."""
    llm = StreamingFakeLLM(responses=["Paris is the capital of France."])
    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config, llm=llm, input_key="question", output_key="answer")

    input_dict = {"question": "What's the capital of France?"}

    chunks = []
    for chunk in rails.stream(input_dict):
        chunks.append(chunk)

    assert len(chunks) > 1

    for chunk in chunks:
        if isinstance(chunk, dict) and "answer" in chunk and chunk["answer"]:
            break
    else:
        assert False, "No valid answer chunk found"

    full_content = "".join(chunk["answer"] if isinstance(chunk, dict) and "answer" in chunk else "" for chunk in chunks)
    assert "Paris" in full_content


def test_runnable_rails_prompt_streaming():
    """Test streaming with prompt values."""
    llm = StreamingFakeLLM(responses=["Hello World!"])
    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config, llm=llm)

    prompt = StringPromptValue(text="Say hello")

    chunks = []
    for chunk in rails.stream(prompt):
        chunks.append(chunk)

    assert len(chunks) > 1

    full_content = "".join(str(chunk) for chunk in chunks)
    assert "Hello World!" in full_content


def test_runnable_rails_input_rail_streaming():
    """Test streaming with input rails."""

    @action(name="check_input")
    async def check_input(context):
        user_message = context.get("user_message", "")
        if "blocked" in user_message.lower():
            return False
        return True

    llm = StreamingFakeLLM(
        responses=[
            "I apologize, but I can't respond to that request.",
            "Hello there! How can I help you?",
        ]
    )

    config = RailsConfig.from_content(
        config={"models": []},
        colang_content="""
        define flow
          user ...
          $allowed = execute check_input
          if not $allowed
            bot refuse to respond
            stop
          bot respond

        define bot refuse to respond
          "I apologize, but I can't respond to that request."
        """,
    )

    rails = RunnableRails(config, llm=llm)
    rails.rails.register_action(check_input)

    blocked_chunks = []
    for chunk in rails.stream("This contains a blocked word"):
        blocked_chunks.append(chunk)

    assert len(blocked_chunks) > 1

    full_blocked_content = "".join(
        chunk if isinstance(chunk, str) else chunk.content for chunk in blocked_chunks if chunk
    )
    assert "I apologize" in full_blocked_content

    llm2 = StreamingFakeLLM(responses=["Hello there! How can I help you?"])
    rails2 = RunnableRails(config, llm=llm2)
    rails2.rails.register_action(check_input)

    allowed_chunks = []
    for chunk in rails2.stream("This is allowed content"):
        allowed_chunks.append(chunk)

    assert len(allowed_chunks) > 1

    full_allowed_content = "".join(
        chunk if isinstance(chunk, str) else chunk.content for chunk in allowed_chunks if chunk
    )
    assert "Hello there" in full_allowed_content


@pytest.mark.skip(reason="Complex chain streaming requires further investigation")
def test_runnable_rails_chain_streaming():
    """Test streaming with a chain."""
    llm = StreamingFakeLLM(responses=["Hello from Paris, France!"])
    config = RailsConfig.from_content(config={"models": []})

    chain = RunnablePassthrough.assign(output=RunnableRails(config, llm=llm))

    chunks = []
    for chunk in chain.stream({"input": "Tell me about Paris"}):
        chunks.append(chunk)

    assert len(chunks) >= 1

    assert isinstance(chunks[0], dict)
    assert "Hello from Paris" in chunks[0]["output"]


@pytest.mark.parametrize(
    "input_type,expected_type",
    [
        (
            "string",
            AIMessageChunk,
        ),
        (
            "message",
            AIMessageChunk,
        ),
        ("dict", dict),
        ("prompt", str),
    ],
)
def test_runnable_rails_output_types(input_type, expected_type):
    """Test that streaming maintains correct output types for different input types."""
    llm = StreamingFakeLLM(responses=["This is a test response"])
    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config, llm=llm)

    if input_type == "string":
        test_input = "Hello"
    elif input_type == "message":
        test_input = HumanMessage(content="Hello")
    elif input_type == "dict":
        test_input = {"input": "Hello"}
    elif input_type == "prompt":
        test_input = StringPromptValue(text="Hello")

    chunks = []
    for chunk in rails.stream(test_input):
        chunks.append(chunk)

    assert isinstance(chunks[-1], expected_type)


def test_auto_streaming_without_streaming_flag():
    """Test that streaming works without explicitly setting streaming=True on the LLM."""
    llm = StreamingFakeLLM(responses=["Auto-streaming test response"])

    assert llm.streaming

    from tests.utils import FakeLLM

    non_streaming_llm = FakeLLM(responses=["Auto-streaming test response"])
    assert not getattr(non_streaming_llm, "streaming", False)

    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config, llm=non_streaming_llm)

    chunks = []
    for chunk in rails.stream("Test auto-streaming"):
        chunks.append(chunk)

    assert len(chunks) > 1

    full_content = "".join(chunk.content if hasattr(chunk, "content") else str(chunk) for chunk in chunks)
    assert "Auto-streaming test response" in full_content


@pytest.mark.asyncio
async def test_streaming_state_restoration():
    """Test that streaming state is properly restored after streaming calls."""
    from tests.utils import FakeLLM

    llm = FakeLLM(responses=["State restoration test"])
    llm.streaming = False

    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config, llm=llm)

    original_streaming = llm.streaming
    assert not original_streaming

    chunks = []
    async for chunk in rails.astream("Test state restoration"):
        chunks.append(chunk)

    assert len(chunks) > 0

    assert llm.streaming == original_streaming
    assert not llm.streaming


def test_langchain_parity_ux():
    """Test that RunnableRails provides the same UX as regular LangChain streaming."""
    from tests.utils import FakeLLM

    llm = FakeLLM(responses=["LangChain parity test"])

    assert not getattr(llm, "streaming", False)

    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config, llm=llm)
    guarded_llm = rails

    chunks = []
    for chunk in guarded_llm.stream("Test LangChain parity"):
        chunks.append(chunk)

    assert len(chunks) > 1

    for chunk in chunks:
        if hasattr(chunk, "content"):
            assert isinstance(chunk.content, str)

    full_content = "".join(chunk.content if hasattr(chunk, "content") else str(chunk) for chunk in chunks)
    assert "LangChain parity test" in full_content


def test_mixed_streaming_and_non_streaming_calls():
    """Test that streaming and non-streaming calls work together seamlessly."""
    from tests.utils import FakeLLM

    llm = FakeLLM(responses=["Mixed call test 1", "Mixed call test 2", "Mixed call test 3"])
    llm.streaming = False

    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config, llm=llm)

    response1 = rails.invoke("First call")
    assert "Mixed call test" in str(response1)
    assert not llm.streaming

    chunks = []
    for chunk in rails.stream("Second call"):
        chunks.append(chunk)

    assert len(chunks) > 1
    assert not llm.streaming

    response2 = rails.invoke("Third call")
    assert "Mixed call test" in str(response2)
    assert not llm.streaming


def test_streaming_with_different_input_types():
    """Test auto-streaming with various input types."""
    from tests.utils import FakeLLM

    llm = FakeLLM(responses=["Input type test"] * 4)
    llm.streaming = False

    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config, llm=llm)

    chunks1 = list(rails.stream("String input"))
    assert len(chunks1) > 1

    from langchain_core.messages import HumanMessage

    chunks2 = list(rails.stream(HumanMessage(content="Message input")))
    assert len(chunks2) > 1

    chunks3 = list(rails.stream({"input": "Dict input"}))
    assert len(chunks3) > 1

    from langchain_core.prompt_values import StringPromptValue

    chunks4 = list(rails.stream(StringPromptValue(text="Prompt input")))
    assert len(chunks4) > 1

    test_cases = [
        (chunks1, "string input"),
        (chunks2, "message input"),
        (chunks3, "dict input"),
        (chunks4, "prompt input"),
    ]

    for chunks, input_type in test_cases:
        if input_type == "dict input":
            full_content = "".join(
                chunk.get("output", "")
                if isinstance(chunk, dict)
                else (chunk.content if hasattr(chunk, "content") else str(chunk))
                for chunk in chunks
            )
        else:
            full_content = "".join(chunk.content if hasattr(chunk, "content") else str(chunk) for chunk in chunks)
        assert "Input type test" in full_content, f"Failed for {input_type}: {full_content}"

    assert not llm.streaming


def test_streaming_metadata_preservation():
    """Test that streaming chunks preserve metadata structure."""
    llm = FakeLLM(responses=["Test response"])
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    chunks = []
    for chunk in model_with_rails.stream("Test input"):
        chunks.append(chunk)

    assert len(chunks) > 0

    for chunk in chunks:
        assert hasattr(chunk, "content")
        assert hasattr(chunk, "additional_kwargs")
        assert hasattr(chunk, "response_metadata")
        assert isinstance(chunk.additional_kwargs, dict)
        assert isinstance(chunk.response_metadata, dict)


@pytest.mark.asyncio
async def test_async_streaming_metadata_preservation():
    """Test that async streaming chunks preserve metadata structure."""
    llm = FakeLLM(responses=["Test async response"])
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    chunks = []
    async for chunk in model_with_rails.astream("Test input"):
        chunks.append(chunk)

    assert len(chunks) > 0

    for chunk in chunks:
        assert hasattr(chunk, "content")
        assert hasattr(chunk, "additional_kwargs")
        assert hasattr(chunk, "response_metadata")
        assert isinstance(chunk.additional_kwargs, dict)
        assert isinstance(chunk.response_metadata, dict)


def test_streaming_chunk_types():
    """Test that streaming returns proper AIMessageChunk types."""
    llm = FakeLLM(responses=["Hello world"])
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    chunks = list(model_with_rails.stream("Hi"))

    for chunk in chunks:
        assert chunk.__class__.__name__ == "AIMessageChunk"

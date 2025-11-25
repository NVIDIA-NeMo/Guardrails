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

from nemoguardrails.rails.llm.buffer import (
    BufferStrategy,
    RollingBuffer,
    get_buffer_strategy,
)
from nemoguardrails.rails.llm.config import OutputRailsStreamingConfig


async def fake_streaming_handler():
    # Fake streaming handler that yields chunks
    for i in range(15):
        yield f"chunk{i}"


async def realistic_streaming_handler():
    """Simulate realistic LLM streaming with proper tokens including spaces."""
    response = "This is a safe and compliant response that should pass."
    tokens = []
    words = response.split(" ")
    for i, word in enumerate(words):
        if i < len(words) - 1:
            # add space to all tokens except the last one
            tokens.append(word + " ")
        else:
            tokens.append(word)

    for token in tokens:
        yield token


async def short_streaming_handler():
    """Stream shorter than buffer size."""
    for token in ["Hello", " ", "world"]:
        yield token


async def empty_streaming_handler():
    """Empty stream."""
    return
    yield  # unreachable


@pytest.mark.asyncio
async def test_buffer_strategy():
    buffer_strategy = RollingBuffer(buffer_context_size=5, buffer_chunk_size=10)
    streaming_handler = fake_streaming_handler()

    expected_processing_contexts = [
        [
            "chunk0",
            "chunk1",
            "chunk2",
            "chunk3",
            "chunk4",
            "chunk5",
            "chunk6",
            "chunk7",
            "chunk8",
            "chunk9",
        ],
        [
            "chunk5",
            "chunk6",
            "chunk7",
            "chunk8",
            "chunk9",
            "chunk10",
            "chunk11",
            "chunk12",
            "chunk13",
            "chunk14",
        ],
        ["chunk10", "chunk11", "chunk12", "chunk13", "chunk14"],
    ]

    expected_user_output_chunks = [
        [
            "chunk0",
            "chunk1",
            "chunk2",
            "chunk3",
            "chunk4",
            "chunk5",
            "chunk6",
            "chunk7",
            "chunk8",
            "chunk9",
        ],
        ["chunk10", "chunk11", "chunk12", "chunk13", "chunk14"],
        [],
    ]

    results = []
    async for idx, chunk_batch in async_enumerate(buffer_strategy(streaming_handler)):
        results.append(
            {
                "processing_context": chunk_batch.processing_context,
                "user_output_chunks": chunk_batch.user_output_chunks,
            }
        )

    for idx, result in enumerate(results):
        assert result["processing_context"] == expected_processing_contexts[idx]
        assert result["user_output_chunks"] == expected_user_output_chunks[idx]


@pytest.mark.asyncio
async def test_buffer_strategy_realistic_data():
    """Test with realistic token data including spaces."""
    buffer_strategy = RollingBuffer(buffer_context_size=2, buffer_chunk_size=4)
    streaming_handler = realistic_streaming_handler()

    expected_results = [
        {
            "processing_context": ["This ", "is ", "a ", "safe "],
            "user_output_chunks": ["This ", "is ", "a ", "safe "],
        },
        {
            "processing_context": ["a ", "safe ", "and ", "compliant "],
            "user_output_chunks": ["and ", "compliant "],
        },
        {
            "processing_context": ["and ", "compliant ", "response ", "that "],
            "user_output_chunks": ["response ", "that "],
        },
        {
            "processing_context": ["response ", "that ", "should ", "pass."],
            "user_output_chunks": ["should ", "pass."],
        },
        {
            "processing_context": ["should ", "pass."],
            "user_output_chunks": [],
        },
    ]

    results = []
    async for chunk_batch in buffer_strategy(streaming_handler):
        results.append(
            {
                "processing_context": chunk_batch.processing_context,
                "user_output_chunks": chunk_batch.user_output_chunks,
            }
        )

    assert results == expected_results


@pytest.mark.asyncio
async def test_both_interfaces_identical():
    """Test both process_stream() and __call__() interfaces work identically."""
    buffer_strategy = RollingBuffer(buffer_context_size=1, buffer_chunk_size=3)

    # process_stream interface
    results_process_stream = []
    async for chunk_batch in buffer_strategy.process_stream(realistic_streaming_handler()):
        results_process_stream.append(
            (
                chunk_batch.processing_context.copy(),
                chunk_batch.user_output_chunks.copy(),
            )
        )

    # __call__ interface
    results_call = []
    async for chunk_batch in buffer_strategy(realistic_streaming_handler()):
        results_call.append(
            (
                chunk_batch.processing_context.copy(),
                chunk_batch.user_output_chunks.copy(),
            )
        )

    assert results_process_stream == results_call


@pytest.mark.asyncio
async def test_edge_cases():
    """Test various edge cases."""

    # empty stream
    buffer_strategy = RollingBuffer(buffer_context_size=2, buffer_chunk_size=4)
    results = []
    async for chunk_batch in buffer_strategy(empty_streaming_handler()):
        results.append(chunk_batch)
    assert results == [], "Empty stream should yield no results"

    # stream shorter than buffer
    results = []
    async for chunk_batch in buffer_strategy(short_streaming_handler()):
        results.append(chunk_batch)

    assert len(results) == 1
    assert results[0].processing_context == ["Hello", " ", "world"]
    assert results[0].user_output_chunks == ["Hello", " ", "world"]


def test_validation():
    """Test input validation."""
    with pytest.raises(ValueError, match="buffer_context_size must be non-negative"):
        RollingBuffer(buffer_context_size=-1)

    with pytest.raises(ValueError, match="buffer_chunk_size must be non-negative"):
        RollingBuffer(buffer_chunk_size=-1)

    buffer = RollingBuffer(buffer_context_size=0, buffer_chunk_size=1)
    assert buffer.buffer_context_size == 0
    assert buffer.buffer_chunk_size == 1


def test_from_config():
    """Test configuration-based instantiation."""
    config = OutputRailsStreamingConfig(context_size=3, chunk_size=6)
    buffer = RollingBuffer.from_config(config)

    assert buffer.buffer_context_size == 3
    assert buffer.buffer_chunk_size == 6


def test_get_buffer_strategy():
    """Test factory function."""
    config = OutputRailsStreamingConfig(context_size=2, chunk_size=5)
    strategy = get_buffer_strategy(config)

    assert isinstance(strategy, RollingBuffer)
    assert strategy.buffer_context_size == 2
    assert strategy.buffer_chunk_size == 5


def test_format_chunks():
    buffer_strategy = RollingBuffer(buffer_context_size=5, buffer_chunk_size=10)
    chunks = ["chunk0", "chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]

    result = buffer_strategy.format_chunks(chunks)
    assert result == "chunk0chunk1chunk2chunk3chunk4chunk5"


def test_format_chunks_realistic():
    """Test format_chunks with realistic token data."""
    buffer_strategy = RollingBuffer()

    chunks = ["Hello", " ", "world", "!"]
    result = buffer_strategy.format_chunks(chunks)
    assert result == "Hello world!"

    # empty chunks
    assert buffer_strategy.format_chunks([]) == ""

    # single chunk
    assert buffer_strategy.format_chunks(["test"]) == "test"


@pytest.mark.asyncio
async def test_total_yielded_tracking():
    """Test that total_yielded is correctly tracked and reset."""
    buffer_strategy = RollingBuffer(buffer_context_size=1, buffer_chunk_size=2)

    # first stream
    user_chunks_1 = []
    async for chunk_batch in buffer_strategy(short_streaming_handler()):
        user_chunks_1.extend(chunk_batch.user_output_chunks)

    # second stream: total_yielded should reset
    user_chunks_2 = []
    async for chunk_batch in buffer_strategy(short_streaming_handler()):
        user_chunks_2.extend(chunk_batch.user_output_chunks)

    # verifies reset worked
    assert user_chunks_1 == user_chunks_2


@pytest.mark.asyncio
async def test_boundary_conditions():
    """Test exact buffer size boundaries."""

    async def exact_size_handler():
        """Stream exactly buffer_chunk_size tokens."""
        for i in range(4):
            yield f"token{i} "

    buffer_strategy = RollingBuffer(buffer_context_size=1, buffer_chunk_size=4)
    results = []
    async for chunk_batch in buffer_strategy(exact_size_handler()):
        results.append(chunk_batch)

    # should get exactly one full chunk plus final empty
    assert len(results) == 2
    assert len(results[0].user_output_chunks) == 4
    # final empty yield
    assert len(results[1].user_output_chunks) == 0


@pytest.mark.asyncio
async def test_subword_token_preservation():
    """Test that subword tokens are preserved without extra spaces (issue #1197)."""

    async def subword_token_stream():
        # simulate subword tokens like BPE tokenization
        # example: "assisting" becomes ["ass", "isting"]
        yield "ass"
        yield "isting"
        yield " with "
        yield "help"
        yield "ing"
        yield " you"

    buffer_strategy = RollingBuffer(buffer_context_size=2, buffer_chunk_size=3)

    # Collect all data in a single pass to avoid creating duplicate streams
    processing_contexts = []
    user_output_parts = []

    async for chunk_batch in buffer_strategy(subword_token_stream()):
        formatted_text = buffer_strategy.format_chunks(chunk_batch.processing_context)
        processing_contexts.append(formatted_text)

        user_chunk_text = buffer_strategy.format_chunks(chunk_batch.user_output_chunks)
        user_output_parts.append(user_chunk_text)

    # reconstruct the full text from user output chunks
    full_text = "".join(user_output_parts)

    # subword tokens should be properly joined
    assert "assisting" in full_text, f"Expected 'assisting' but got: {full_text}"
    assert "helping" in full_text, f"Expected 'helping' but got: {full_text}"

    # verify no extra spaces were introduced between subword tokens
    assert "ass isting" not in full_text, f"Found extra space in subword tokens: {full_text}"
    assert "help ing" not in full_text, f"Found extra space in subword tokens: {full_text}"

    # expected result should be: "assisting with helping you"
    expected = "assisting with helping you"
    assert full_text == expected, f"Expected '{expected}' but got '{full_text}'"


async def async_enumerate(aiterable, start=0):
    idx = start
    async for item in aiterable:
        yield idx, item
        idx += 1


def test_abstract_base_class_cannot_be_instantiated():
    """Test that the abstract BufferStrategy cannot be instantiated directly."""

    with pytest.raises(TypeError):
        BufferStrategy()


def test_incomplete_implementation_raises_error():
    """Test that incomplete implementations of BufferStrategy raise TypeError."""

    class IncompleteBufferStrategy(BufferStrategy):
        pass

    with pytest.raises(TypeError):
        IncompleteBufferStrategy()

    class MissingProcessStreamStrategy(BufferStrategy):
        @classmethod
        def from_config(cls, config):
            return cls()

        def format_chunks(self, chunks):
            return "".join(chunks)

    with pytest.raises(TypeError):
        MissingProcessStreamStrategy()

    class MissingFormatChunksStrategy(BufferStrategy):
        @classmethod
        def from_config(cls, config):
            return cls()

        async def process_stream(self, streaming_handler):
            async for chunk in streaming_handler:
                yield chunk

    with pytest.raises(TypeError):
        MissingFormatChunksStrategy()

    class MissingFromConfigStrategy(BufferStrategy):
        def format_chunks(self, chunks):
            return "".join(chunks)

        async def process_stream(self, streaming_handler):
            async for chunk in streaming_handler:
                yield chunk

    with pytest.raises(TypeError):
        MissingFromConfigStrategy()


def test_additional_validation_errors():
    """Test additional validation errors beyond the existing ones."""

    with pytest.raises(ValueError, match="buffer_context_size must be non-negative"):
        RollingBuffer(buffer_context_size=-100)

    with pytest.raises(ValueError, match="buffer_chunk_size must be non-negative"):
        RollingBuffer(buffer_chunk_size=-1000)

    with pytest.raises(ValueError, match="buffer_context_size must be non-negative"):
        RollingBuffer(buffer_context_size=-1, buffer_chunk_size=-1)


def test_validation_with_zero_values():
    """Test that zero values are accepted for buffer parameters."""

    buffer = RollingBuffer(buffer_context_size=0, buffer_chunk_size=5)
    assert buffer.buffer_context_size == 0
    assert buffer.buffer_chunk_size == 5

    buffer = RollingBuffer(buffer_context_size=5, buffer_chunk_size=0)
    assert buffer.buffer_context_size == 5
    assert buffer.buffer_chunk_size == 0

    buffer = RollingBuffer(buffer_context_size=0, buffer_chunk_size=0)
    assert buffer.buffer_context_size == 0
    assert buffer.buffer_chunk_size == 0


@pytest.mark.asyncio
async def test_complete_implementation_works():
    """Test that a complete implementation of BufferStrategy works correctly."""

    class CompleteBufferStrategy(BufferStrategy):
        def __init__(self, test_param=None):
            self.test_param = test_param

        @classmethod
        def from_config(cls, config):
            return cls(test_param="from_config")

        def format_chunks(self, chunks):
            return "|".join(chunks)

        async def process_stream(self, streaming_handler):
            buffer = []
            async for chunk in streaming_handler:
                buffer.append(chunk)
                if len(buffer) >= 2:
                    from nemoguardrails.rails.llm.buffer import ChunkBatch

                    yield ChunkBatch(processing_context=buffer, user_output_chunks=buffer)
                    buffer = []

            if buffer:
                from nemoguardrails.rails.llm.buffer import ChunkBatch

                yield ChunkBatch(processing_context=buffer, user_output_chunks=buffer)

    strategy = CompleteBufferStrategy()
    assert strategy.test_param is None

    config = OutputRailsStreamingConfig(context_size=1, chunk_size=1)
    strategy = CompleteBufferStrategy.from_config(config)
    assert strategy.test_param == "from_config"

    chunks = ["hello", "world"]
    result = strategy.format_chunks(chunks)
    assert result == "hello|world"

    async def test_handler():
        for chunk in ["a", "b", "c"]:
            yield chunk

    results = []
    async for chunk_batch in strategy.process_stream(test_handler()):
        results.append(chunk_batch)

    assert len(results) == 2
    assert results[0].processing_context == ["a", "b"]
    assert results[0].user_output_chunks == ["a", "b"]
    assert results[1].processing_context == ["c"]
    assert results[1].user_output_chunks == ["c"]

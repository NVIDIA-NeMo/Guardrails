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

import pytest

from nemoguardrails.rails.llm.buffer import RollingBuffer as BufferStrategy
from nemoguardrails.rails.llm.buffer import get_buffer_strategy
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
    buffer_strategy = BufferStrategy(buffer_context_size=5, buffer_chunk_size=10)
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
    buffer_strategy = BufferStrategy(buffer_context_size=2, buffer_chunk_size=4)
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
    buffer_strategy = BufferStrategy(buffer_context_size=1, buffer_chunk_size=3)

    # process_stream interface
    results_process_stream = []
    async for chunk_batch in buffer_strategy.process_stream(
        realistic_streaming_handler()
    ):
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
    buffer_strategy = BufferStrategy(buffer_context_size=2, buffer_chunk_size=4)
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
        BufferStrategy(buffer_context_size=-1)

    with pytest.raises(ValueError, match="buffer_chunk_size must be non-negative"):
        BufferStrategy(buffer_chunk_size=-1)

    buffer = BufferStrategy(buffer_context_size=0, buffer_chunk_size=1)
    assert buffer.buffer_context_size == 0
    assert buffer.buffer_chunk_size == 1


def test_from_config():
    """Test configuration-based instantiation."""
    config = OutputRailsStreamingConfig(context_size=3, chunk_size=6)
    buffer = BufferStrategy.from_config(config)

    assert buffer.buffer_context_size == 3
    assert buffer.buffer_chunk_size == 6


def test_get_buffer_strategy():
    """Test factory function."""
    config = OutputRailsStreamingConfig(context_size=2, chunk_size=5)
    strategy = get_buffer_strategy(config)

    assert isinstance(strategy, BufferStrategy)
    assert strategy.buffer_context_size == 2
    assert strategy.buffer_chunk_size == 5


def test_format_chunks():
    buffer_strategy = BufferStrategy(buffer_context_size=5, buffer_chunk_size=10)
    chunks = ["chunk0", "chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]

    result = buffer_strategy.format_chunks(chunks)
    assert result == "chunk0chunk1chunk2chunk3chunk4chunk5"


def test_format_chunks_realistic():
    """Test format_chunks with realistic token data."""
    buffer_strategy = BufferStrategy()

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
    buffer_strategy = BufferStrategy(buffer_context_size=1, buffer_chunk_size=2)

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

    buffer_strategy = BufferStrategy(buffer_context_size=1, buffer_chunk_size=4)
    results = []
    async for chunk_batch in buffer_strategy(exact_size_handler()):
        results.append(chunk_batch)

    # should get exactly one full chunk plus final empty
    assert len(results) == 2
    assert len(results[0].user_output_chunks) == 4
    # final empty yield
    assert len(results[1].user_output_chunks) == 0


async def async_enumerate(aiterable, start=0):
    idx = start
    async for item in aiterable:
        yield idx, item
        idx += 1

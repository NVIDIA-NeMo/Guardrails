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

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, NamedTuple

from nemoguardrails.rails.llm.config import OutputRailsStreamingConfig

__all__ = ["ChunkBatch", "BufferStrategy", "RollingBuffer", "get_buffer_strategy"]


class ChunkBatch(NamedTuple):
    """Represents a batch of processed chunks from a buffer strategy.

    This class contains the raw chunk data from buffer processing. For string
    representation of chunks, use the buffer strategy's format_chunks() method.

    Attributes:
        processing_context (List[str]): Chunks to be used for output rails processing,
            including context from previous chunks.
        user_output_chunks (List[str]): New chunks to be streamed to the end user
            in their original token format. Use this for user output or when you
            only need the newly processed content.

    Example:
        >>> async for chunk_batch in buffer_strategy.process_stream(handler):
        ...     # for output rails processing (needs context):
        ...     context_str = buffer_strategy.format_chunks(chunk_batch.processing_context)
        ...     analyze_content(context_str)
        ...
        ...     # for user output (only new content):
        ...     user_output = buffer_strategy.format_chunks(chunk_batch.user_output_chunks)
        ...     yield_to_user(user_output)
        ...
        ...     # or iterate over raw chunks:
        ...     for chunk in chunk_batch.user_output_chunks:
        ...         process_individual_chunk(chunk)
    """

    processing_context: List[str]
    user_output_chunks: List[str]


class BufferStrategy(ABC):
    """Abstract base class for buffer strategies in streaming output rails.

    This class defines the interface for buffer strategies that manage how
    streaming chunks are buffered and processed for output rails.
    Concrete implementations should handle the accumulation and yielding of
    chunks in a way that optimizes output rails processing while maintaining
    streaming performance.

    The interface separates concerns:
    - Buffer management logic (process_stream)
    - Chunk representation formatting (format_chunks)

    Note:
        All concrete implementations must implement `from_config`, `process_stream`,
        and `format_chunks` methods to provide configuration-based
        instantiation, chunk processing, and string representation capabilities.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: OutputRailsStreamingConfig) -> "BufferStrategy":
        """Create a buffer strategy instance from configuration.

        Args:
            config (OutputRailsStreamingConfig): Configuration object containing
                buffer strategy parameters.

        Returns:
            BufferStrategy: A configured buffer strategy instance.

        """
        ...

    @abstractmethod
    def format_chunks(self, chunks: List[str]) -> str:
        """Format chunks into a string representation for user consumption.

        This method defines how chunks should be formatted into a string
        representation. Different strategies might join chunks differently
        (e.g., preserving spaces, adding separators, etc.).

        Args:
            chunks (List[str]): List of chunk tokens to be formatted.

        Returns:
            str: String representation of the chunks ready for consumers.


        Example:
            >>> strategy = SomeBufferStrategy()
            >>> chunks = ["Hello", " ", "world"]
            >>> result = strategy.format_chunks(chunks)
            >>> print(result)  # "Hello world"
        """
        ...

    @abstractmethod
    async def process_stream(
        self, streaming_handler
    ) -> AsyncGenerator[ChunkBatch, None]:
        """Process streaming chunks and yield chunk batches.

        This is the main method that concrete buffer strategies must implement.
        It defines how chunks from the streaming handler should be buffered,
        processed, and yielded as ChunkBatch objects.

        Args:
            streaming_handler: An async iterator that yields individual string
                chunks from the LLM stream.

        Yields:
            ChunkBatch: Named tuple containing processing_context and user_output_chunks.


        Example:
            >>> strategy = SomeBufferStrategy()
            >>> async for chunk_batch in strategy.process_stream(handler):
            ...     # for output rails processing (needs context):
            ...     context_formatted = strategy.format_chunks(chunk_batch.processing_context)
            ...     # for user output (new content only):
            ...     user_formatted = strategy.format_chunks(chunk_batch.user_output_chunks)
            ...     print(f"Processing: {context_formatted}")
            ...     print(f"User: {user_formatted}")
        """
        raise NotImplementedError
        yield

    async def __call__(self, streaming_handler) -> AsyncGenerator[ChunkBatch, None]:
        """Callable interface that delegates to process_stream.

        It delegates to the `process_stream` method and can
        be extended to add common functionality like validation, logging,
        or error handling.

        Args:
            streaming_handler: An async iterator that yields individual string
                chunks from the LLM stream.

        Yields:
            ChunkBatch: Named tuple containing processing_context and user_output_chunks.

        Example:
            >>> strategy = SomeBufferStrategy()
            >>> # both of these work:
            >>> async for batch in strategy.process_stream(handler):
            ...     context_formatted = strategy.format_chunks(batch.processing_context)
            >>> async for batch in strategy(handler):  # delegates to process_stream
            ...     user_formatted = strategy.format_chunks(batch.user_output_chunks)
        """
        async for chunk_batch in self.process_stream(streaming_handler):
            yield chunk_batch


class RollingBuffer(BufferStrategy):
    """A rolling buffer strategy for streaming output rails processing.

    This strategy accumulates incoming chunks in a buffer and yields them in
    batches when the buffer reaches the specified chunk size. It maintains
    context from previous chunks to ensure continuity in processing output rails.

    The buffer operates by:
    1. Accumulating incoming chunks until reaching the chunk size threshold
    2. Yielding a processing buffer (with context) and new chunks to process
    3. Retaining context tokens for the next processing round
    4. Yielding any remaining chunks at the end of the stream

    Args:
        buffer_context_size (int, optional): Number of tokens carried over from
            previous chunks to provide context for continuity. Defaults to 5.
        buffer_chunk_size (int, optional): Number of tokens in each processing
            chunk. This determines the size of token blocks on which output
            rails are applied. Defaults to 10.

    Attributes:
        buffer_context_size (int): Number of context tokens retained between chunks.
        buffer_chunk_size (int): Number of tokens in each processing chunk.
        total_yielded (int): Tracks the total number of chunks yielded to the user.

    Example:
        >>> config = OutputRailsStreamingConfig(context_size=2, chunk_size=4)
        >>> buffer = RollingBuffer.from_config(config)
        >>> async for chunk_batch in buffer.process_stream(stream_handler):
        ...     # for output rails processing (needs context)
        ...     processing_text = buffer.format_chunks(chunk_batch.processing_context)
        ...     # For user output (new content only)
        ...     user_text = buffer.format_chunks(chunk_batch.user_output_chunks)
        ...     pass
        >>> # or use the callable interface:
        >>> async for chunk_batch in buffer(stream_handler):
        ...     # same as above, delegates to process_stream
        ...     processing_text = buffer.format_chunks(chunk_batch.processing_context)
        ...     pass

    Note:
        The processing buffer includes context from previous chunks, while
        user_output_chunks contains only the tokens to be yielded to the user.
    """

    def __init__(self, buffer_context_size: int = 5, buffer_chunk_size: int = 10):
        """Initialize the RollingBuffer with specified buffer sizes.

        Args:
            buffer_context_size (int, optional): Number of context tokens to
                retain between chunks. Defaults to 5.
            buffer_chunk_size (int, optional): Number of tokens per processing
                chunk. Defaults to 10.

        Returns:
            None

        Raises:
            ValueError: If buffer_context_size or buffer_chunk_size is negative.
        """
        if buffer_context_size < 0:
            raise ValueError("buffer_context_size must be non-negative")
        if buffer_chunk_size < 0:
            raise ValueError("buffer_chunk_size must be non-negative")

        self.buffer_context_size = buffer_context_size
        self.buffer_chunk_size = buffer_chunk_size
        # track total chunks yielded to user
        self.total_yielded = 0

    @classmethod
    def from_config(cls, config: OutputRailsStreamingConfig):
        """Create a RollingBuffer instance from a streaming configuration.

        Args:
            config (OutputRailsStreamingConfig): Configuration object containing
                context_size and chunk_size parameters.

        Returns:
            RollingBuffer: A new RollingBuffer instance configured with the
                provided parameters.

        Example:
            >>> config = OutputRailsStreamingConfig(context_size=3, chunk_size=6)
            >>> buffer = RollingBuffer.from_config(config)
        """
        return cls(
            buffer_context_size=config.context_size, buffer_chunk_size=config.chunk_size
        )

    async def process_stream(
        self, streaming_handler
    ) -> AsyncGenerator[ChunkBatch, None]:
        """Process streaming chunks using rolling buffer strategy.

        This method implements the rolling buffer logic, accumulating chunks
        and yielding them in batches with context for output rails processing.
        The buffer maintains a sliding window of context tokens for continuity.

        Args:
            streaming_handler: An async iterator that yields individual string
                chunks from the LLM stream.

        Yields:
            ChunkBatch: Named tuple containing processing_context and user_output_chunks.

        Example:
            >>> async def stream_handler():
            ...     for chunk in ["Hello", " ", "world", "!"]:
            ...         yield chunk
            >>>
            >>> buffer = RollingBuffer(context_size=1, chunk_size=2)
            >>> async for chunk_batch in buffer.process_stream(stream_handler()):
            ...     print(f"Processing buffer: {chunk_batch.processing_context}")
            ...     print(f"New chunks: {chunk_batch.user_output_chunks}")
            ...     # for output rails processing (with context):
            ...     context_str = buffer.format_chunks(chunk_batch.processing_context)
            ...     # for user output (new content only):
            ...     user_str = buffer.format_chunks(chunk_batch.user_output_chunks)
            ...     print(f"Processing: '{context_str}', User: '{user_str}'")

        Note:
            The method resets the total_yielded counter at the start of each
            streaming session to ensure accurate tracking.
        """
        # reset state for each streaming session
        self.total_yielded = 0
        buffer = []
        total_chunks = 0

        async for chunk in streaming_handler:
            buffer.append(chunk)
            total_chunks += 1

            if len(buffer) >= self.buffer_chunk_size:
                # calculate how many new chunks should be yielded
                new_chunks_to_yield = min(
                    self.buffer_chunk_size, total_chunks - self.total_yielded
                )

                # create the processing buffer (includes context)
                processing_buffer = buffer[
                    -self.buffer_chunk_size - self.buffer_context_size :
                ]

                # get the new chunks to yield to user (preserve original token format)
                # the new chunks are at the end of the buffer
                chunks_to_yield = buffer[-new_chunks_to_yield:]
                self.total_yielded += new_chunks_to_yield

                yield ChunkBatch(
                    processing_context=processing_buffer,
                    user_output_chunks=chunks_to_yield,
                )
                buffer = buffer[-self.buffer_context_size :]

        # yield any remaining buffer if it's not empty
        if buffer:
            # calculate how many chunks from the remaining buffer haven't been yielded yet
            remaining_chunks_to_yield = total_chunks - self.total_yielded
            chunks_to_yield = (
                buffer[-remaining_chunks_to_yield:]
                if remaining_chunks_to_yield > 0
                else []
            )

            yield ChunkBatch(
                processing_context=buffer,
                user_output_chunks=chunks_to_yield,
            )

    def format_chunks(self, chunks: List[str]) -> str:
        """Generate string representation of chunks preserving original token format.

        The RollingBuffer strategy preserves the original token format by
        joining chunks without modification, maintaining spaces and formatting
        as they appeared in the original LLM output.

        Args:
            chunks (List[str]): List of chunk tokens to be formatted.

        Returns:
            str: String representation preserving original token spacing and format.

        Example:
            >>> buffer = RollingBuffer()
            >>> chunks = ["Hello", " ", "world", "!"]
            >>> result = buffer.format_chunks(chunks)
            >>> print(result)  # "Hello world!"
        """
        return "".join(chunks)


def get_buffer_strategy(config: OutputRailsStreamingConfig) -> BufferStrategy:
    """Create a buffer strategy from the given configuration.

    Args:
        config (OutputRailsStreamingConfig): Configuration object specifying
            the buffer strategy parameters.

    Returns:
        BufferStrategy: A configured buffer strategy instance. Currently
            returns a RollingBuffer instance.

    Example:
        >>> config = OutputRailsStreamingConfig(context_size=2, chunk_size=4)
        >>> strategy = get_buffer_strategy(config)
        >>> isinstance(strategy, RollingBuffer)
        True

    Note:
        This is currently a simple factory that only returns RollingBuffer
        instances. Future versions may support multiple buffer strategies
        with a registry pattern.
    """
    # TODO: use a factory function or class
    return RollingBuffer.from_config(config)

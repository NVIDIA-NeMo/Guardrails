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
from typing import AsyncGenerator, List, Tuple

from nemoguardrails.rails.llm.config import OutputRailsStreamingConfig


class BufferStrategy(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config: OutputRailsStreamingConfig) -> "BufferStrategy":
        pass

    # The abstract method is not async to ensure the return type
    # matches the async generator in the concrete implementation.
    @abstractmethod
    def __call__(self, streaming_handler) -> AsyncGenerator[Tuple[List[str], str], None]:
        pass

    @abstractmethod
    def generate_chunk_str(self, *args, **kwargs) -> str:
        pass


class RollingBuffer(BufferStrategy):
    """A minimal buffer strategy that buffers chunks and yields them when the buffer is full.

    Args:
        buffer_context_size (int): The number of tokens carried over from the previous chunk to provide context for continuity in processing.
        buffer_chunk_size (int): The number of tokens in each processing chunk. This is the size of the token block on which output rails are applied.
    """

    def __init__(self, buffer_context_size: int = 5, buffer_chunk_size: int = 10):
        self.buffer_context_size = buffer_context_size
        self.buffer_chunk_size = buffer_chunk_size
        self.last_index = 0

    @classmethod
    def from_config(cls, config: OutputRailsStreamingConfig):
        return cls(buffer_context_size=config.context_size, buffer_chunk_size=config.chunk_size)

    async def __call__(self, streaming_handler) -> AsyncGenerator[Tuple[List[str], str], None]:
        buffer = []
        index = 0

        async for chunk in streaming_handler:
            buffer.append(chunk)
            index += 1

            if len(buffer) >= self.buffer_chunk_size:
                yield (
                    # we apply output rails on the buffer
                    buffer[-self.buffer_chunk_size - self.buffer_context_size :],
                    # generate_chunk_str is what gets printed in the console or yield to user
                    # to avoid repeating the already streamed/printed chunk
                    self.generate_chunk_str(
                        buffer[-self.buffer_chunk_size - self.buffer_context_size :],
                        index,
                    ),
                )
                buffer = buffer[-self.buffer_context_size :]

        # Yield any remaining buffer if it's not empty
        if buffer:
            yield (
                buffer,
                self.generate_chunk_str(buffer[-self.buffer_chunk_size - self.buffer_context_size :], index),
            )

    def generate_chunk_str(self, buffer, current_index) -> str:
        if current_index <= self.last_index:
            return ""

        new_chunks = buffer[self.last_index - current_index :]
        self.last_index = current_index
        # TODO: something causes duplicate whitespaces between tokens, figure out why,
        # If using `return "".join(new_chunks)` works, then the issue might be elsewhere in the code where the chunks are being generated or processed.
        # Ensure that the chunks themselves do not contain extra spaces.
        # WAR: return "".join(new_chunks)
        return "".join(new_chunks)


def get_buffer_strategy(config: OutputRailsStreamingConfig) -> BufferStrategy:
    # TODO: use a factory function or class
    # currently we only have RollingBuffer, in future we use a registry
    return RollingBuffer.from_config(config)

# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import logging
import warnings
from typing import Any, AsyncIterator, Dict, Optional, Union

from nemoguardrails.utils import new_uuid

log = logging.getLogger(__name__)

# sentinel object to indicate end of stream
END_OF_STREAM = object()


class StreamingHandler(AsyncIterator):
    """Provider-agnostic streaming handler with prefix/suffix/stop handling.

    Implements AsyncIterator interface so it can be used directly to stream
    back the response. Chunks are pushed via push_chunk() and consumed via
    async iteration.
    """

    def __init__(
        self,
        enable_print: bool = False,
        enable_buffer: bool = False,
        include_metadata: Optional[bool] = False,
        include_generation_metadata: Optional[bool] = None,
    ):
        if include_generation_metadata is not None:
            warnings.warn(
                "include_generation_metadata is deprecated, use include_metadata instead. "
                "It will be removed in version 0.22.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            include_metadata = include_generation_metadata

        # A unique id for the stream handler
        self.uid = new_uuid()

        # The queue where the chunks are gathered for when the handler also acts as an AsyncIterator
        self.queue = asyncio.Queue()
        self.streaming_finished_event = asyncio.Event()

        # When printing is enabled, the handler will print the processed chunks in green.
        self.enable_print = enable_print

        # When buffering is enabled, the chunks will gather in a buffer.
        self.enable_buffer = enable_buffer

        # The prefix/suffix that should be removed (text-only processing)
        self.prefix = None
        self.suffix = None

        # The current chunk which needs to be checked for prefix/suffix matching
        self.current_chunk = ""

        # The current buffer until we start processing.
        self.buffer = ""

        # The full completion
        self.completion = ""

        # Whether we're interested in the top k non-empty lines
        self.k = 0
        self.top_k_nonempty_lines_event = asyncio.Event()

        # If set, the chunk will be piped to the specified handler rather than added to the queue or printed
        self.pipe_to = None

        # The stop chunks
        self.stop = []

        self.include_metadata = include_metadata
        self.current_metadata = {}

    def set_pattern(self, prefix: Optional[str] = None, suffix: Optional[str] = None):
        """Sets the pattern that is expected.

        If a prefix or a suffix are specified, they will be removed from the output.
        """
        self.prefix = prefix
        self.suffix = suffix

    def set_pipe_to(self, another_handler: "StreamingHandler"):
        self.pipe_to = another_handler

    async def wait(self):
        """Waits until the stream finishes and returns the full completion."""
        await self.streaming_finished_event.wait()
        return self.completion

    async def wait_top_k_nonempty_lines(self, k: int):
        """Waits for top k non-empty lines from the LLM.

        When k lines have been received (and k+1 has been started) it will return
        and remove them from the buffer.
        """
        self.k = k
        await self.top_k_nonempty_lines_event.wait()

        lines = self.buffer.split("\n")
        top_k_lines = []
        i = 0
        for i in range(len(lines)):
            line = lines[i].strip()
            if len(line) > 0 and line[0] != "#":
                top_k_lines.append(lines[i])
            if len(top_k_lines) == k:
                break

        self.buffer = "\n".join(lines[i + 1 :])
        return "\n".join(top_k_lines)

    async def enable_buffering(self):
        self.enable_buffer = True
        self.buffer = ""

    async def disable_buffering(self):
        """When we disable the buffer, we process the buffer as a chunk."""
        self.enable_buffer = False
        await self.push_chunk(self.buffer)
        self.buffer = ""

    def __aiter__(self):
        async def generator():
            while True:
                element = None
                try:
                    element = await self.queue.get()
                except RuntimeError as ex:
                    if "Event loop is closed" not in str(ex):
                        raise ex
                if element is END_OF_STREAM:
                    break

                if isinstance(element, dict):
                    if element.get("text") is END_OF_STREAM:
                        element["text"] = ""
                        yield element
                        break
                yield element

        return generator()

    async def __anext__(self):
        element = None
        try:
            element = await self.queue.get()
        except RuntimeError as ex:
            if "Event loop is closed" not in str(ex):
                raise ex
        if element is END_OF_STREAM:
            raise StopAsyncIteration

        if isinstance(element, dict):
            if element.get("text") is END_OF_STREAM:
                raise StopAsyncIteration
            return element
        else:
            return element

    async def _process(
        self,
        chunk: Union[str, object],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Process a chunk of text.

        If we're in buffering mode, record the text.
        Otherwise, update the full completion, check for stop tokens, and enqueue the chunk.
        """

        if self.include_metadata and metadata:
            self.current_metadata.update(metadata)

        if self.enable_buffer:
            if chunk is not END_OF_STREAM:
                self.buffer += chunk if chunk is not None else ""
                lines = [line.strip() for line in self.buffer.split("\n")]
                lines = [line for line in lines if len(line) > 0 and line[0] != "#"]
                if len(lines) > self.k > 0:
                    self.top_k_nonempty_lines_event.set()

        else:
            prev_completion = self.completion

            if chunk is not None and chunk is not END_OF_STREAM:
                self.completion += chunk
                # Check if the completion contains one of the stop chunks
                for stop_chunk in self.stop:
                    if stop_chunk in self.completion:
                        # Make sure the stop chunk is not included
                        self.completion = self.completion.split(stop_chunk)[0]

                        # If the current chunk does add something new to the final completion
                        # We push that as well.
                        if len(self.completion) > len(prev_completion):
                            self.current_chunk = self.completion[len(prev_completion) :]
                            await self.push_chunk(END_OF_STREAM)
                        # And we stop the streaming
                        self.streaming_finished_event.set()
                        self.top_k_nonempty_lines_event.set()
                        return

            if self.pipe_to:
                # only add explicit empty strings, not ones created during processing
                if chunk is END_OF_STREAM or chunk is not None:
                    asyncio.create_task(self.pipe_to.push_chunk(chunk))
                if chunk is END_OF_STREAM:
                    self.streaming_finished_event.set()
                    self.top_k_nonempty_lines_event.set()
            else:
                if self.enable_print and chunk is not None and chunk is not END_OF_STREAM:
                    print(f"\033[92m{chunk}\033[0m", end="", flush=True)

                # we only want to filter out empty strings that are created during suffix processing,
                # not ones directly pushed by the user
                if chunk is not None:
                    if self.include_metadata:
                        chunk_dict = {"text": chunk if chunk is not END_OF_STREAM else END_OF_STREAM}
                        if chunk is END_OF_STREAM:
                            metadata = self.current_metadata.copy() if self.current_metadata else {}
                            metadata.setdefault("response_metadata", None)
                            metadata.setdefault("usage_metadata", None)
                            chunk_dict["metadata"] = metadata
                        elif self.current_metadata:
                            chunk_dict["metadata"] = self.current_metadata.copy()
                        await self.queue.put(chunk_dict)
                    else:
                        await self.queue.put(chunk)

                    # If the chunk is the special end of stream marker, mark the stream as finished.
                    if chunk is END_OF_STREAM:
                        self.streaming_finished_event.set()
                        self.top_k_nonempty_lines_event.set()

    async def push_chunk(
        self,
        chunk: Union[str, None],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Push a new string chunk to the stream.

        Args:
            chunk: String chunk to push, None to signal end of stream, or END_OF_STREAM sentinel.
            metadata: Optional metadata about the generation.
        """
        if chunk is None:
            chunk = END_OF_STREAM
        elif chunk is END_OF_STREAM:
            # already the correct marker, no conversion needed
            pass
        elif isinstance(chunk, str):
            # empty string is a valid chunk and should be processed normally
            pass
        else:
            raise TypeError(f"StreamingHandler.push_chunk() expects str, got {type(chunk).__name__}")

        if self.streaming_finished_event.is_set():
            log.info(f"{self.uid[0:3]} - CHUNK after finish: {chunk}")
            return

        if self.include_metadata and metadata:
            self.current_metadata.update(metadata)

        # Process prefix: accumulate until the expected prefix is received, then remove it.
        if self.prefix:
            if chunk is not None and chunk is not END_OF_STREAM:
                self.current_chunk += chunk
            if self.current_chunk.startswith(self.prefix):
                self.current_chunk = self.current_chunk[len(self.prefix) :]
                self.prefix = None
                # If we're left with something, we "forward it".
                if self.current_chunk:
                    await self._process(self.current_chunk)
                    self.current_chunk = ""
        # Process suffix/stop tokens: accumulate and check whether the current chunk ends with one.
        elif self.suffix or self.stop:
            if chunk is not None and chunk is not END_OF_STREAM:
                self.current_chunk += chunk
            _chunks = []
            if self.suffix:
                _chunks.append(self.suffix)
            if self.stop:
                _chunks.extend(self.stop)
            skip_processing = False
            for _chunk in _chunks:
                if skip_processing:
                    break
                for _len in range(len(_chunk)):
                    if self.current_chunk.endswith(_chunk[0 : _len + 1]):
                        skip_processing = True
                        break

            if skip_processing and chunk is not END_OF_STREAM:
                return
            else:
                if chunk is END_OF_STREAM:
                    if self.current_chunk and self.suffix and self.current_chunk.endswith(self.suffix):
                        self.current_chunk = self.current_chunk[0 : -1 * len(self.suffix)]

                # only process the current_chunk if it's not empty
                if self.current_chunk:
                    await self._process(self.current_chunk, metadata)
                    self.current_chunk = ""

                # if this is the end of stream, pass it through after processing the current chunk
                if chunk is END_OF_STREAM:
                    await self._process(END_OF_STREAM, metadata)
        else:
            await self._process(chunk, metadata)

    async def finish(self):
        """Signal end of stream."""
        if self.current_chunk:
            if self.suffix and self.current_chunk.endswith(self.suffix):
                self.current_chunk = self.current_chunk[: -1 * len(self.suffix)]

            await self._process(self.current_chunk)
            self.current_chunk = ""
        await self._process(END_OF_STREAM)
        # We explicitly print a new line here
        if self.enable_print:
            print("")

        # Reset prefix/suffix for the next generation.
        self.prefix = None
        self.suffix = None

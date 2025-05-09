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

import asyncio
from typing import List, Optional, Union

import pytest

from nemoguardrails.streaming import StreamingHandler


class StreamingConsumer:
    """Helper class for testing a streaming handler.

    It consumes the chunks from teh stream.
    """

    def __init__(self, streaming_handler: StreamingHandler):
        self.streaming_handler = streaming_handler

        self.chunks = []
        self.finished = False
        self._task = None
        self._start()

    async def process_tokens(self):
        try:
            async for chunk in self.streaming_handler:
                self.chunks.append(chunk)
        except asyncio.CancelledError:
            # task was cancelled. this is expected during cleanup
            pass
        finally:
            self.finished = True

    def _start(self):
        self._task = asyncio.create_task(self.process_tokens())

    async def get_chunks(self):
        """Helper to get the chunks."""
        # We wait a bit to allow all asyncio callbacks to get called.
        await asyncio.sleep(0.01)
        return self.chunks

    async def cancel(self):
        """Cancel the background task and wait for it to finish."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                # this is expected when cancelling the task
                pass


@pytest.mark.asyncio
async def test_single_chunk():
    streaming_handler = StreamingHandler()
    streaming_consumer = StreamingConsumer(streaming_handler)

    try:
        await streaming_handler.push_chunk("a")
        assert await streaming_consumer.get_chunks() == ["a"]
    finally:
        await streaming_consumer.cancel()


@pytest.mark.asyncio
async def test_sequence_of_chunks():
    streaming_handler = StreamingHandler()
    streaming_consumer = StreamingConsumer(streaming_handler)

    try:
        for chunk in ["1", "2", "3", "4", "5"]:
            await streaming_handler.push_chunk(chunk)

        assert await streaming_consumer.get_chunks() == ["1", "2", "3", "4", "5"]
    finally:
        await streaming_consumer.cancel()


async def _test_pattern_case(
    chunks: List[Union[str, None]],
    final_chunks: List[str],
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    stop: Optional[List[str]] = (),
    use_pipe: bool = False,
):
    """Helper for testing a stream with prefix.

    When a None chunk is found, it checks that there are no collected chunks up to that point.
    """
    streaming_handler = StreamingHandler()
    streaming_handler.set_pattern(prefix=prefix, suffix=suffix)
    streaming_handler.stop = stop

    if use_pipe:
        _streaming_handler = StreamingHandler()
        streaming_handler.set_pipe_to(_streaming_handler)
        streaming_consumer = StreamingConsumer(_streaming_handler)
    else:
        streaming_consumer = StreamingConsumer(streaming_handler)

    try:
        for chunk in chunks:
            if chunk is None:
                assert await streaming_consumer.get_chunks() == []
            else:
                await streaming_handler.push_chunk(chunk)

        # Push an empty chunk to signal the ending.
        await streaming_handler.push_chunk("")

        assert await streaming_consumer.get_chunks() == final_chunks
    finally:
        await streaming_consumer.cancel()


@pytest.mark.asyncio
async def test_prefix_1():
    await _test_pattern_case(
        prefix="User intent: ",
        suffix=None,
        chunks=["User", None, " ", None, "intent", None, ":", " ask question"],
        final_chunks=["ask question"],
    )


@pytest.mark.asyncio
async def test_prefix_2():
    await _test_pattern_case(
        prefix="User intent: ",
        suffix=None,
        chunks=["User intent: ask question"],
        final_chunks=["ask question"],
    )


@pytest.mark.asyncio
async def test_prefix_3():
    await _test_pattern_case(
        prefix="User intent: ",
        suffix=None,
        chunks=["User", None, " ", None, "intent", None, ": ask question"],
        final_chunks=["ask question"],
    )


@pytest.mark.asyncio
async def test_suffix_1():
    await _test_pattern_case(
        prefix='Bot message: "',
        suffix='"',
        chunks=["Bot", " message: ", '"', "This is a message", '"'],
        final_chunks=["This is a message"],
    )


@pytest.mark.asyncio
async def test_suffix_with_stop():
    await _test_pattern_case(
        prefix='Bot message: "',
        suffix='"',
        stop=["\nUser intent: "],
        chunks=[
            "Bot",
            " message: ",
            '"',
            "This is a message",
            '"',
            "\n",
            "User ",
            "intent: ",
            "bla",
        ],
        final_chunks=["This is a message"],
    )


@pytest.mark.asyncio
async def test_suffix_with_stop_and_pipe():
    await _test_pattern_case(
        prefix='Bot message: "',
        suffix='"',
        stop=["\nUser intent: "],
        use_pipe=True,
        chunks=[
            "Bot",
            " message: ",
            '"',
            "This is a message",
            '"',
            "\n",
            "User ",
            "intent: ",
            "bla",
        ],
        final_chunks=["This is a message"],
    )


@pytest.mark.asyncio
async def test_suffix_with_stop_and_pipe_2():
    await _test_pattern_case(
        prefix='Bot message: "',
        suffix='"',
        stop=["\nUser intent: "],
        use_pipe=True,
        chunks=[
            "Bot",
            " message: ",
            '"',
            "This is a message",
            '."',
        ],
        final_chunks=["This is a message", "."],
    )


@pytest.mark.asyncio
async def test_suffix_with_stop_and_pipe_3():
    await _test_pattern_case(
        prefix='Bot message: "',
        suffix='"',
        stop=["\nUser intent: "],
        use_pipe=True,
        chunks=[
            "Bot",
            " message: ",
            '"',
            "This is a message",
            '."\nUser',
            " intent: ",
            " xxx",
        ],
        final_chunks=["This is a message", "."],
    )


@pytest.mark.asyncio
async def test_suffix_with_stop_and_pipe_4():
    await _test_pattern_case(
        prefix='Bot message: "',
        suffix='"',
        stop=['"\n'],
        use_pipe=True,
        chunks=[
            "Bot",
            " message: ",
            '"',
            "This is a message",
            '."\nUser',
            " intent: ",
            " xxx",
        ],
        final_chunks=["This is a message", "."],
    )

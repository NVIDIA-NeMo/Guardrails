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
import io
import sys
import unittest.mock as mock
from typing import List, Optional, Union

import pytest

from nemoguardrails.streaming import END_OF_STREAM, StreamingHandler


class StreamingConsumer:
    """Helper class for testing a streaming handler."""

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
        # we wait a bit to allow all asyncio callbacks to get called.
        await asyncio.sleep(0.1)
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
@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("finish", [False, True])
@pytest.mark.parametrize("text", ["", "  ", "Unformatted reply"])
async def test_unmatched_prefix_flushes_on_end(text, buffered, finish):
    handler = StreamingHandler(enable_buffer=buffered)
    handler.set_pattern(prefix=('  "', "  “"), suffix=('"', "”"))
    await handler.push_chunk(text)
    if finish:
        await handler.finish()
    else:
        await handler.push_chunk(END_OF_STREAM)
    if buffered:
        await handler.disable_buffering()
    chunks = await asyncio.wait_for(_collect_handler(handler), timeout=1)
    assert "".join(chunks) == text
    assert handler.completion == text


async def _collect_handler(handler):
    return [chunk async for chunk in handler]


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_before_release", [False, True])
async def test_pattern_set_during_buffering_preserves_chunk_order(finish_before_release):
    handler = StreamingHandler(enable_buffer=True)
    await handler.push_chunk("  “Hello")
    handler.set_pattern(prefix=('  "', "  “"), suffix=('"', "”"))
    await handler.push_chunk(", there!”")
    if finish_before_release:
        await handler.finish()
    await handler.disable_buffering()
    if not finish_before_release:
        await handler.finish()
    chunks = await asyncio.wait_for(_collect_handler(handler), timeout=1)
    assert "".join(chunks) == "Hello, there!"
    assert handler.completion == "Hello, there!"


@pytest.mark.asyncio
@pytest.mark.parametrize("opening,closing", [('"', '"'), ("“", "”")])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("buffered", [False, True])
async def test_alternative_quote_patterns(opening, closing, split, buffered):
    handler = StreamingHandler(enable_buffer=buffered)
    handler.set_pattern(prefix=('  "', "  “"), suffix=('"', "”"))
    body = 'Say “hello” and "goodbye".'
    text = f"  {opening}{body}{closing}"
    for chunk in list(text) if split else [text]:
        await handler.push_chunk(chunk)
    if split and not buffered:
        assert handler.completion == body
    await handler.finish()
    if buffered:
        await handler.disable_buffering()
    chunks = await asyncio.wait_for(_collect_handler(handler), timeout=1)
    assert "".join(chunks) == body
    assert handler.completion == body


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
        await streaming_handler.push_chunk(END_OF_STREAM)

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


@pytest.mark.asyncio
async def test_set_pipe_to():
    """Test set_pipe_to verify streaming is correctly piped to another handler."""

    main_handler = StreamingHandler()
    secondary_handler = StreamingHandler()
    main_consumer = StreamingConsumer(main_handler)
    secondary_consumer = StreamingConsumer(secondary_handler)

    try:
        # piping from main to secondary handler
        main_handler.set_pipe_to(secondary_handler)

        # send chunks to main handler
        await main_handler.push_chunk("chunk1")
        await main_handler.push_chunk("chunk2")
        await main_handler.push_chunk(END_OF_STREAM)

        # main handler received nothing (piped away)
        main_chunks = await main_consumer.get_chunks()
        assert len(main_chunks) == 0

        # ensure secondary handler received the chunks
        secondary_chunks = await secondary_consumer.get_chunks()
        assert len(secondary_chunks) >= 2
        assert "chunk1" in secondary_chunks
        assert "chunk2" in secondary_chunks
    finally:
        await main_consumer.cancel()
        await secondary_consumer.cancel()


@pytest.mark.asyncio
async def test_wait_method():
    """Test the wait method to verify it waits for streaming to finish."""
    handler = StreamingHandler()
    consumer = StreamingConsumer(handler)

    try:

        async def push_chunks_with_delay():
            await handler.push_chunk("chunk1")
            await asyncio.sleep(0.1)
            await handler.push_chunk("chunk2")
            await asyncio.sleep(0.1)
            await handler.push_chunk(END_OF_STREAM)

        push_task = asyncio.create_task(push_chunks_with_delay())

        completion = await handler.wait()

        assert completion == "chunk1chunk2"

        await push_task
    finally:
        await consumer.cancel()


@pytest.mark.asyncio
async def test_wait_top_k_nonempty_lines():
    """Test the wait_top_k_nonempty_lines method with a timeout to prevent hanging."""
    handler = StreamingHandler()

    await handler.enable_buffering()

    # create a background task to push lines
    async def push_lines():
        await handler.push_chunk("Line 1\n")
        # following should be skipped
        await handler.push_chunk("# Comment line\n")
        await handler.push_chunk("Line 2\n")
        await handler.push_chunk("Line 3\n")
        await handler.push_chunk("Line 4\n")
        # Explicitly make sure we have enough non-empty lines to trigger the event
        # this is important as the test could hang if the event isn't set
        handler.top_k_nonempty_lines_event.set()

    # start pushing lines in the background
    push_task = asyncio.create_task(push_lines())

    try:
        # Wait for top 2 non-empty lines with a timeout
        top_k_lines = await asyncio.wait_for(handler.wait_top_k_nonempty_lines(2), timeout=2.0)

        # verify we got the expected lines
        assert top_k_lines == "Line 1\nLine 2"

        # verify the buffer now only contains the remaining lines
        assert handler.buffer == "Line 3\nLine 4\n"
    except asyncio.TimeoutError:
        pytest.fail("wait_top_k_nonempty_lines timed out")
    finally:
        if not push_task.done():
            push_task.cancel()
            try:
                await push_task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_enable_and_disable_buffering():
    """Test the enable_buffering and disable_buffering methods."""

    handler = StreamingHandler()
    consumer = StreamingConsumer(handler)

    try:
        await handler.enable_buffering()

        await handler.push_chunk("chunk1")
        await handler.push_chunk("chunk2")

        # verify chunks were buffered not streamed
        chunks = await consumer.get_chunks()
        assert len(chunks) == 0
        assert handler.buffer == "chunk1chunk2"

        # disable buffering; should process the buffer as a chunk
        await handler.disable_buffering()

        # verify the buffer was processed and streamed
        chunks = await consumer.get_chunks()
        assert len(chunks) >= 1
        assert "chunk1chunk2" in chunks

        assert handler.buffer == ""
    finally:
        await consumer.cancel()


@pytest.mark.asyncio
async def test_multiple_stop_tokens():
    """Test handling of multiple stop tokens."""
    handler = StreamingHandler()
    consumer = StreamingConsumer(handler)

    try:
        handler.stop = ["STOP1", "STOP2", "HALT"]

        # Push text with a stop token in the middle
        await handler.push_chunk("This is some text STOP1 and this should be ignored")
        await handler.push_chunk(END_OF_STREAM)

        # streaming stopped at the stop token
        chunks = await consumer.get_chunks()
        assert len(chunks) >= 1
        assert chunks[0] == "This is some text "
    finally:
        await consumer.cancel()

    handler = StreamingHandler()
    consumer = StreamingConsumer(handler)
    try:
        handler.stop = ["STOP1", "STOP2", "HALT"]

        await handler.push_chunk("Different text with HALT token")
        await handler.push_chunk(END_OF_STREAM)

        chunks = await consumer.get_chunks()
        assert len(chunks) >= 1
        assert chunks[0] == "Different text with "
    finally:
        await consumer.cancel()


@pytest.mark.asyncio
async def test_stop_none_does_not_raise():
    handler = StreamingHandler()
    consumer = StreamingConsumer(handler)

    try:
        handler.stop = None
        await handler.push_chunk("Hello world")
        await handler.push_chunk(END_OF_STREAM)

        chunks = await consumer.get_chunks()
        assert chunks == ["Hello world"]
        assert handler.completion == "Hello world"
    finally:
        await consumer.cancel()


@pytest.mark.asyncio
async def test_stop_none_with_pattern():
    await _test_pattern_case(
        prefix='Bot message: "',
        suffix='"',
        stop=None,
        chunks=[
            "Bot",
            " message: ",
            '"',
            "This is a message",
            '"',
        ],
        final_chunks=["This is a message"],
    )


@pytest.mark.asyncio
async def test_enable_print_functionality():
    """Test the enable_print functionality."""

    original_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        handler = StreamingHandler(enable_print=True)

        await handler.push_chunk("Hello")
        await handler.push_chunk(" World")

        await handler.finish()

        printed_output = sys.stdout.getvalue()

        assert "\033[92mHello\033[0m" in printed_output
        assert "\033[92m World\033[0m" in printed_output
    finally:
        # reestore stdout
        sys.stdout = original_stdout


@pytest.mark.asyncio
async def test_suffix_removal_at_end():
    """Test that suffix is removed at the end of streaming."""

    handler = StreamingHandler()
    consumer = StreamingConsumer(handler)

    try:
        handler.set_pattern(suffix="END")

        await handler.push_chunk("This is a test E")
        await handler.push_chunk("N")

        # should be buffered in current_chunk, not streamed yet
        chunks = await consumer.get_chunks()
        assert len(chunks) == 0

        await handler.push_chunk("D")
        await handler.push_chunk(END_OF_STREAM)

        # Check that suffix was removed
        chunks = await consumer.get_chunks()
        assert len(chunks) >= 1
        assert chunks[0] == "This is a test "
    finally:
        await consumer.cancel()


@pytest.mark.asyncio
async def test_anext_with_none_element():
    """Test __anext__ method with None element (now END_OF_STREAM sentinel)."""

    streaming_handler = StreamingHandler()

    # put END_OF_STREAM into the queue (signal to stop streaming)
    await streaming_handler.queue.put(END_OF_STREAM)

    # call __anext__ directly
    with pytest.raises(StopAsyncIteration):
        await streaming_handler.__anext__()


@pytest.mark.asyncio
async def test_anext_with_end_of_stream_sentinel():
    """Test __anext__ method explicitly with END_OF_STREAM sentinel."""
    streaming_handler = StreamingHandler()

    # Put END_OF_STREAM into the queue
    await streaming_handler.queue.put(END_OF_STREAM)

    # Call __anext__ and expect StopAsyncIteration
    with pytest.raises(StopAsyncIteration):
        await streaming_handler.__anext__()


@pytest.mark.asyncio
async def test_anext_with_empty_string():
    """Test __anext__ method with empty string."""
    streaming_handler = StreamingHandler()

    # NOTE: azure openai issue
    # put empty string into the queue
    await streaming_handler.queue.put("")

    result = await streaming_handler.__anext__()
    assert result == ""


@pytest.mark.asyncio
async def test_anext_with_dict_empty_text():
    """Test __anext__ method with dict containing empty text."""
    streaming_handler = StreamingHandler()
    test_val = {"text": "", "metadata": {}}

    # put dict with empty text into the queue
    await streaming_handler.queue.put(test_val)

    result = await streaming_handler.__anext__()
    assert result == test_val


@pytest.mark.asyncio
async def test_anext_with_dict_none_text():
    """Test __anext__ method with dict containing None text."""
    streaming_handler = StreamingHandler()
    test_val = {"text": None, "metadata": {}}

    # NOTE: azure openai issue
    # put dict with None text into the queue
    await streaming_handler.queue.put(test_val)

    result = await streaming_handler.__anext__()
    assert result == test_val


@pytest.mark.asyncio
async def test_anext_with_normal_text():
    """Test __anext__ method with normal text."""
    streaming_handler = StreamingHandler()

    test_text = "test text"
    await streaming_handler.queue.put(test_text)

    result = await streaming_handler.__anext__()
    assert result == test_text


@pytest.mark.asyncio
async def test_anext_with_event_loop_closed():
    """Test __anext__ method with RuntimeError 'Event loop is closed'."""

    streaming_handler = StreamingHandler()

    # mock queue.get to raise RuntimeError
    with mock.patch.object(streaming_handler.queue, "get", side_effect=RuntimeError("Event loop is closed")):
        result = await streaming_handler.__anext__()
        assert result is None


@pytest.mark.asyncio
async def test_anext_with_other_runtime_error():
    """Test __anext__ method with other RuntimeError."""
    streaming_handler = StreamingHandler()

    # mock queue.get to raise other RuntimeError
    with mock.patch.object(streaming_handler.queue, "get", side_effect=RuntimeError("Some other error")):
        # should propagate the error
        with pytest.raises(RuntimeError, match="Some other error"):
            await streaming_handler.__anext__()


@pytest.mark.asyncio
async def test_include_metadata():
    """Test push_chunk with metadata when include_metadata is True."""
    streaming_handler = StreamingHandler(include_metadata=True)
    streaming_consumer = StreamingConsumer(streaming_handler)

    try:
        test_text = "test text"
        test_metadata = {"temperature": 0.7, "top_p": 0.95}

        await streaming_handler.push_chunk(test_text, metadata=test_metadata)
        await streaming_handler.push_chunk(END_OF_STREAM)

        chunks = await streaming_consumer.get_chunks()
        assert len(chunks) >= 1
        assert chunks[0]["text"] == test_text
        assert chunks[0]["metadata"] == test_metadata
    finally:
        await streaming_consumer.cancel()


@pytest.mark.asyncio
async def test_processing_metadata():
    """Test that metadata is properly passed through the processing chain."""
    streaming_handler = StreamingHandler(include_metadata=True)
    streaming_consumer = StreamingConsumer(streaming_handler)

    try:
        streaming_handler.set_pattern(prefix="PREFIX: ", suffix="SUFFIX")

        test_text = "PREFIX: This is a test message SUFFIX"
        test_metadata = {"temperature": 0.7, "top_p": 0.95}

        await streaming_handler.push_chunk(test_text, metadata=test_metadata)
        await streaming_handler.push_chunk(END_OF_STREAM)

        chunks = await streaming_consumer.get_chunks()
        assert len(chunks) >= 1
        # NOTE: The suffix is only removed at the end of generation
        assert "This is a test message" in chunks[0]["text"]
        assert chunks[0]["metadata"] == test_metadata
    finally:
        await streaming_consumer.cancel()

    streaming_handler = StreamingHandler(include_metadata=True)
    streaming_consumer = StreamingConsumer(streaming_handler)
    try:
        streaming_handler.set_pattern(prefix="PREFIX: ", suffix="SUFFIX")

        await streaming_handler.push_chunk("PRE", metadata={"part": 1})
        await streaming_handler.push_chunk("FIX: ", metadata={"part": 2})
        await streaming_handler.push_chunk("Test ", metadata={"part": 3})
        await streaming_handler.push_chunk("message", metadata={"part": 4})
        await streaming_handler.push_chunk(" SUFF", metadata={"part": 5})
        await streaming_handler.push_chunk("IX", metadata={"part": 6})
        await streaming_handler.push_chunk(END_OF_STREAM)

        chunks = await streaming_consumer.get_chunks()
        # the prefix removal should happen first, then streaming happens
        # verify the text chunks are delivered correctly
        assert len(chunks) >= 2
        for i, expected in enumerate(
            [
                {"text": "Test ", "part": 3},
                {"text": "message", "part": 4},
            ]
        ):
            if i < len(chunks) and "text" in chunks[i]:
                assert chunks[i]["text"] == expected["text"]
                assert chunks[i]["metadata"]["part"] == expected["part"]
    finally:
        await streaming_consumer.cancel()


@pytest.mark.asyncio
async def test_metadata_accumulation_across_chunks():
    """Test that metadata from separate chunks (e.g. response_metadata and usage_metadata) accumulates."""
    streaming_handler = StreamingHandler(include_metadata=True)
    streaming_consumer = StreamingConsumer(streaming_handler)

    try:
        await streaming_handler.push_chunk("Hello")
        await streaming_handler.push_chunk(
            "!", metadata={"response_metadata": {"finish_reason": "stop", "model_name": "gpt-4o"}}
        )
        await streaming_handler.push_chunk(
            "", metadata={"usage_metadata": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
        )
        await streaming_handler.push_chunk(None)

        chunks = await streaming_consumer.get_chunks()

        content_chunks = [c for c in chunks if c["text"] != ""]
        assert len(content_chunks) == 2
        assert content_chunks[0]["text"] == "Hello"
        assert "metadata" not in content_chunks[0]

        metadata_chunks = [c for c in chunks if "metadata" in c]
        assert len(metadata_chunks) >= 1
        final_metadata = metadata_chunks[-1]["metadata"]
        assert "response_metadata" in final_metadata
        assert final_metadata["response_metadata"]["finish_reason"] == "stop"
        assert "usage_metadata" in final_metadata
        assert final_metadata["usage_metadata"]["total_tokens"] == 15
    finally:
        await streaming_consumer.cancel()


@pytest.mark.asyncio
async def test_raw_usage_metadata_chunk_is_not_repeated_on_final_chunk():
    streaming_handler = StreamingHandler(include_metadata=True)
    streaming_consumer = StreamingConsumer(streaming_handler)
    usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    try:
        await streaming_handler.push_chunk("Hello")
        await streaming_handler.push_chunk("", metadata={"usage": usage})
        await streaming_handler.push_chunk(None)

        chunks = await streaming_consumer.get_chunks()
        usage_chunks = [chunk for chunk in chunks if chunk.get("metadata", {}).get("usage")]

        assert len(usage_chunks) == 1
        assert usage_chunks[0]["metadata"]["usage"] == usage
        assert "usage" not in chunks[-1]["metadata"]
        assert chunks[-1]["metadata"]["response_metadata"] is None
        assert chunks[-1]["metadata"]["usage_metadata"] is None
    finally:
        await streaming_consumer.cancel()


@pytest.mark.asyncio
async def test_metadata_defaults_when_provider_returns_no_metadata():
    """Test that the final chunk includes metadata with None values when include_metadata=True
    but the provider does not return any metadata."""
    streaming_handler = StreamingHandler(include_metadata=True)
    streaming_consumer = StreamingConsumer(streaming_handler)

    try:
        await streaming_handler.push_chunk("Hello")
        await streaming_handler.push_chunk(" world")
        await streaming_handler.push_chunk(None)

        chunks = await streaming_consumer.get_chunks()

        final_chunk = chunks[-1]
        assert "metadata" in final_chunk
        assert final_chunk["metadata"]["response_metadata"] is None
        assert final_chunk["metadata"]["usage_metadata"] is None
    finally:
        await streaming_consumer.cancel()


@pytest.mark.asyncio
async def test_anext_with_dict_end_of_stream_sentinel():
    """Test __anext__ with a dict-wrapped END_OF_STREAM sentinel."""

    streaming_handler = StreamingHandler(include_metadata=True)
    await streaming_handler.queue.put({"text": END_OF_STREAM, "metadata": {}})
    with pytest.raises(StopAsyncIteration):
        await streaming_handler.__anext__()


@pytest.mark.asyncio
async def test_push_chunk_unsupported_type():
    """Test push_chunk with an unsupported data type."""

    streaming_handler = StreamingHandler()
    with pytest.raises(
        TypeError, match="StreamingHandler.push_chunk\\(\\) expects str, ToolCallChunk, or ToolCallDeltaChunk, got int"
    ):
        await streaming_handler.push_chunk(123)
    with pytest.raises(
        TypeError, match="StreamingHandler.push_chunk\\(\\) expects str, ToolCallChunk, or ToolCallDeltaChunk, got list"
    ):
        await streaming_handler.push_chunk([1, 2])


@pytest.mark.asyncio
async def test_finish_method():
    """Test the finish() method signals end of stream correctly."""
    handler = StreamingHandler()
    consumer = StreamingConsumer(handler)

    try:
        await handler.push_chunk("Hello")
        await handler.push_chunk(" World")
        await handler.finish()

        chunks = await consumer.get_chunks()
        assert chunks == ["Hello", " World"]
        assert handler.completion == "Hello World"
        assert handler.streaming_finished_event.is_set()
    finally:
        await consumer.cancel()


@pytest.mark.asyncio
async def test_finish_with_suffix():
    """Test finish() removes suffix correctly."""
    handler = StreamingHandler()
    consumer = StreamingConsumer(handler)

    try:
        handler.set_pattern(suffix="END")
        await handler.push_chunk("Hello WorldE")
        await handler.push_chunk("N")
        await handler.push_chunk("D")
        await handler.finish()

        chunks = await consumer.get_chunks()
        assert chunks == ["Hello World"]
    finally:
        await consumer.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("include_metadata", [False, True])
@pytest.mark.parametrize("piped", [False, True])
async def test_tool_events_follow_pending_text_without_text_filters(include_metadata, piped):
    from nemoguardrails.types import ToolCall, ToolCallChunk, ToolCallFunction

    handler = StreamingHandler(include_metadata=include_metadata)
    consumer = StreamingHandler(include_metadata=include_metadata) if piped else handler
    if piped:
        handler.set_pipe_to(consumer)
    handler.stop = ["User:"]
    handler.suffix = '"'
    event = ToolCallChunk([ToolCall(id="call_1", function=ToolCallFunction(name="User:", arguments={"text": '"'}))])
    await handler.push_chunk('Hello"')
    await handler.push_chunk(event)
    await handler.finish()
    chunks = [chunk async for chunk in consumer]
    assert chunks[:2] == ([{"text": "Hello"}, event] if include_metadata else ["Hello", event])
    assert handler.completion == consumer.completion == "Hello"


@pytest.mark.asyncio
async def test_stop_suppresses_subsequent_tool_events():
    from nemoguardrails.types import ToolCall, ToolCallChunk

    handler = StreamingHandler()
    handler.stop = ["STOP"]
    await handler.push_chunk("STOP")
    await handler.push_chunk(ToolCallChunk([ToolCall(id="call_1")]))
    await handler.finish()
    assert [chunk async for chunk in handler] == []


@pytest.mark.asyncio
async def test_buffered_dialog_rejects_tool_events():
    from nemoguardrails.exceptions import StreamingNotSupportedError
    from nemoguardrails.types import ToolCall, ToolCallChunk

    handler = StreamingHandler(enable_buffer=True)
    with pytest.raises(StreamingNotSupportedError, match="buffered dialog"):
        await handler.push_chunk(ToolCallChunk([ToolCall(id="call_1")]))
    assert handler.queue.empty()
    assert handler.buffer == handler.completion == ""


@pytest.mark.asyncio
async def test_anext_rejects_tool_calls_when_disabled():
    from nemoguardrails.exceptions import StreamingNotSupportedError
    from nemoguardrails.types import ToolCall, ToolCallChunk

    handler = StreamingHandler(allow_tool_calls=False)
    await handler.push_chunk(ToolCallChunk([ToolCall(id="call_1")]))
    with pytest.raises(StreamingNotSupportedError, match="Tool calls.*output rails.*during streaming"):
        await anext(handler)


@pytest.mark.asyncio
async def test_pipe_cannot_bypass_disabled_tool_calls():
    from nemoguardrails.exceptions import StreamingNotSupportedError
    from nemoguardrails.types import ToolCall, ToolCallChunk

    handler = StreamingHandler(allow_tool_calls=False)
    target = StreamingHandler()
    handler.set_pipe_to(target)
    with pytest.raises(StreamingNotSupportedError, match="Tool calls.*output rails.*during streaming"):
        await handler.push_chunk(ToolCallChunk([ToolCall(id="call_1")]))
    assert target.queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_prefix", [False, True])
@pytest.mark.parametrize("producer_first", [False, True])
async def test_buffered_wait_observes_producer_failure(after_prefix, producer_first):
    from nemoguardrails.exceptions import StreamingNotSupportedError

    handler = StreamingHandler(enable_buffer=True)
    release = asyncio.Event()
    error = StreamingNotSupportedError("buffered tool calls are unsupported")
    prefix_ready = asyncio.Event()

    async def produce():
        if after_prefix:
            await handler.push_chunk('user greeting\nbot greeting\n"Hello')
        prefix_ready.set()
        await release.wait()
        raise error

    producer = asyncio.create_task(produce())
    handler.set_producer_task(producer)
    try:
        if producer_first:
            await asyncio.wait_for(prefix_ready.wait(), 2)
        if after_prefix:
            assert await asyncio.wait_for(handler.wait_top_k_nonempty_lines(2), 2) == "user greeting\nbot greeting"
            assert not producer.done()
            await handler.disable_buffering()
            waiting = handler.wait()
        else:
            waiting = handler.wait_top_k_nonempty_lines(2)
        release.set()
        with pytest.raises(StreamingNotSupportedError) as raised:
            await asyncio.wait_for(waiting, 2)
        assert raised.value is error
    finally:
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)


@pytest.mark.asyncio
async def test_completed_buffered_producer_preserves_short_response():
    handler = StreamingHandler(enable_buffer=True)

    async def produce():
        await handler.push_chunk("short response")
        await handler.finish()

    producer = asyncio.create_task(produce())
    handler.set_producer_task(producer)
    assert await asyncio.wait_for(handler.wait_top_k_nonempty_lines(2), 2) == "short response"
    assert producer.done()


@pytest.mark.asyncio
async def test_cancelled_buffer_wait_cancels_producer():
    handler = StreamingHandler(enable_buffer=True)
    started = asyncio.Event()
    closed = asyncio.Event()

    async def produce():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    producer = asyncio.create_task(produce())
    handler.set_producer_task(producer)
    waiter = asyncio.create_task(handler.wait_top_k_nonempty_lines(2))
    await asyncio.wait_for(started.wait(), 2)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert producer.cancelled()
    assert closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("closing", ['"', "”"])
@pytest.mark.parametrize("trailing", ["\n", " \t\r\n"])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("finish", [False, True])
async def test_closing_quote_before_trailing_whitespace(closing, trailing, split, buffered, finish):
    handler = StreamingHandler(enable_buffer=buffered)
    handler.set_pattern(prefix=('  "', "  “"), suffix=('"', "”"))
    text = f"  “Hello!{closing}{trailing}"
    for chunk in list(text) if split else [text]:
        await handler.push_chunk(chunk)
    if finish:
        await handler.finish()
    else:
        await handler.push_chunk(END_OF_STREAM)
    if buffered:
        await handler.disable_buffering()

    assert "".join(await asyncio.wait_for(_collect_handler(handler), timeout=1)) == "Hello!"
    assert handler.completion == "Hello!"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,expected",
    [
        ("Hello!  ”\n", "Hello!  "),
        ("Hello! \t\n", "Hello! \t\n"),
        ("She said “hello” \t\nand then waved.”\n", "She said “hello” \t\nand then waved."),
    ],
)
async def test_suffix_whitespace_preserves_message_content(body, expected):
    handler = StreamingHandler()
    handler.set_pattern(suffix=('"', "”"))
    for chunk in body:
        await handler.push_chunk(chunk)
    await handler.finish()

    assert "".join(await asyncio.wait_for(_collect_handler(handler), timeout=1)) == expected

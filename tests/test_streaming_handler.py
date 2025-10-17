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

import asyncio
import io
import sys
import unittest.mock as mock
from typing import List, Optional, Union
from uuid import UUID

import pytest
from langchain.schema.messages import AIMessageChunk
from langchain.schema.output import ChatGenerationChunk, GenerationChunk

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
        await main_handler.push_chunk(END_OF_STREAM)  # Signal end of streaming

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
            await handler.push_chunk(
                END_OF_STREAM
            )  # NOTE: signal end of streaming will get changed soon

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
        top_k_lines = await asyncio.wait_for(
            handler.wait_top_k_nonempty_lines(2), timeout=2.0
        )

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
        await handler.push_chunk(
            END_OF_STREAM
        )  # NOTE: Signal end of streaming we are going to change this

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
        await handler.push_chunk(
            END_OF_STREAM
        )  # NOTE: Signal end of streaming we are going to change this

        chunks = await consumer.get_chunks()
        assert len(chunks) >= 1
        assert chunks[0] == "Different text with "
    finally:
        await consumer.cancel()


@pytest.mark.asyncio
async def test_enable_print_functionality():
    """Test the enable_print functionality."""

    original_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        handler = StreamingHandler(enable_print=True)

        await handler.push_chunk("Hello")
        await handler.push_chunk(" World")

        # end streaming to trigger newline print
        # NOTE: None signals the end of streaming also ""
        await handler.on_llm_end(
            response=None, run_id=UUID("00000000-0000-0000-0000-000000000000")
        )

        printed_output = sys.stdout.getvalue()

        assert "\033[92mHello\033[0m" in printed_output
        assert "\033[92m World\033[0m" in printed_output
    finally:
        # reestore stdout
        sys.stdout = original_stdout


@pytest.mark.asyncio
async def test_first_token_handling():
    """Test the first_token flag behavior directly."""
    handler = StreamingHandler()
    assert handler.first_token is True

    # Mock push_chunk to verify it's not called for empty first token
    original_push_chunk = handler.push_chunk
    push_chunk_called = False

    async def mock_push_chunk(chunk, *args, **kwargs):
        nonlocal push_chunk_called
        push_chunk_called = True

    # replace the method temporarily
    handler.push_chunk = mock_push_chunk

    try:
        # call on_llm_new_token with empty first token
        await handler.on_llm_new_token(
            token="", run_id=UUID("00000000-0000-0000-0000-000000000000")
        )

        # first_token is now False
        assert handler.first_token is False
        # push_chunk was not called (empty first token is skipped)
        assert push_chunk_called is True

        # reset the mock state
        push_chunk_called = False

        # NOTE: this is not the root cause of streaming bug with Azure OpenAI
        # call on_llm_new_token with empty token again (not first)
        await handler.on_llm_new_token(
            token="", run_id=UUID("00000000-0000-0000-0000-000000000000")
        )

        # push_chunk should be called (empty non-first token is not skipped)
        assert push_chunk_called is True

        await handler.on_llm_new_token(
            token="This is a test", run_id=UUID("00000000-0000-0000-0000-000000000000")
        )

        # NOTE: THIS IS A BUG
        assert push_chunk_called is True

        # TODO:
        # asssert that streaming has ended when we are here
    finally:
        # restore the original method
        handler.push_chunk = original_push_chunk
        # Clean up the queue if any items were added by mock or direct calls
        # This ensures no pending tasks from this handler interfere elsewhere.
        if hasattr(handler, "queue") and handler.queue is not None:
            while not handler.queue.empty():
                try:
                    handler.queue.get_nowait()
                    handler.queue.task_done()
                except asyncio.QueueEmpty:
                    break
            # Signal end if push_chunk was mocked and might not have done it
            await handler.queue.put(END_OF_STREAM)


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
        await handler.push_chunk(END_OF_STREAM)  # NOTE: will get changed to SENTINEL

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
    test_val = {"text": "", "generation_info": {}}

    # put dict with empty text into the queue
    await streaming_handler.queue.put(test_val)

    result = await streaming_handler.__anext__()
    assert result == test_val


@pytest.mark.asyncio
async def test_anext_with_dict_none_text():
    """Test __anext__ method with dict containing None text."""
    streaming_handler = StreamingHandler()
    test_val = {"text": None, "generation_info": {}}

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
    with mock.patch.object(
        streaming_handler.queue, "get", side_effect=RuntimeError("Event loop is closed")
    ):
        result = await streaming_handler.__anext__()
        assert result is None


@pytest.mark.asyncio
async def test_anext_with_other_runtime_error():
    """Test __anext__ method with other RuntimeError."""
    streaming_handler = StreamingHandler()

    # mock queue.get to raise other RuntimeError
    with mock.patch.object(
        streaming_handler.queue, "get", side_effect=RuntimeError("Some other error")
    ):
        # should propagate the error
        with pytest.raises(RuntimeError, match="Some other error"):
            await streaming_handler.__anext__()


@pytest.mark.asyncio
async def test_include_generation_metadata():
    """Test push_chunk with generation_info when include_generation_metadata is True."""
    streaming_handler = StreamingHandler(include_generation_metadata=True)
    streaming_consumer = StreamingConsumer(streaming_handler)

    try:
        test_text = "test text"
        test_generation_info = {"temperature": 0.7, "top_p": 0.95}

        await streaming_handler.push_chunk(
            test_text, generation_info=test_generation_info
        )
        await streaming_handler.push_chunk(
            END_OF_STREAM
        )  # NOTE: sjignal end of streaming using "" will get changed soon

        chunks = await streaming_consumer.get_chunks()
        assert len(chunks) >= 1
        assert chunks[0]["text"] == test_text
        assert chunks[0]["generation_info"] == test_generation_info
    finally:
        await streaming_consumer.cancel()


@pytest.mark.asyncio
async def test_include_generation_metadata_with_different_chunk_types():
    """Test push_chunk with different chunk types when include_generation_metadata is True."""

    streaming_handler = StreamingHandler(include_generation_metadata=True)
    streaming_consumer = StreamingConsumer(streaming_handler)

    try:
        test_text = "test text"
        test_generation_info = {"temperature": 0.7, "top_p": 0.95}

        generation_chunk = GenerationChunk(
            text=test_text, generation_info=test_generation_info
        )
        await streaming_handler.push_chunk(
            generation_chunk, generation_info=test_generation_info
        )
        await streaming_handler.push_chunk(
            END_OF_STREAM
        )  # NOTE: sjignal end of streaming using "" will get changed soon

        chunks = await streaming_consumer.get_chunks()
        assert len(chunks) >= 1
        assert chunks[0]["text"] == test_text
        assert chunks[0]["generation_info"] == test_generation_info
    finally:
        await streaming_consumer.cancel()

    # reset handler and consumer for a clean test
    streaming_handler = StreamingHandler(include_generation_metadata=True)
    streaming_consumer = StreamingConsumer(streaming_handler)

    try:
        ai_message_chunk = AIMessageChunk(content=test_text)
        await streaming_handler.push_chunk(
            ai_message_chunk, generation_info=test_generation_info
        )
        await streaming_handler.push_chunk(
            END_OF_STREAM
        )  # NOTE: sjignal end of streaming using "" will get changed soon

        chunks = await streaming_consumer.get_chunks()
        assert len(chunks) >= 1
        assert chunks[0]["text"] == test_text
        assert chunks[0]["generation_info"] == test_generation_info
    finally:
        await streaming_consumer.cancel()


@pytest.mark.asyncio
async def test_on_chat_model_start():
    """Test on_chat_model_start method."""
    streaming_handler = StreamingHandler()

    streaming_handler.current_chunk = "existing chunk"

    await streaming_handler.on_chat_model_start(
        serialized={},
        messages=[[]],
        run_id=UUID("00000000-0000-0000-0000-000000000000"),
    )

    # current_chunk is reset
    assert streaming_handler.current_chunk == ""


@pytest.mark.asyncio
async def test_on_llm_new_token_empty_then_nonempty():
    """Test on_llm_new_token method with empty token followed by non-empty token."""
    streaming_handler = StreamingHandler()
    streaming_consumer = StreamingConsumer(streaming_handler)

    try:
        # first token is empty,  this will be skipped based on implementation
        # NOTE: not azure openai bug
        await streaming_handler.on_llm_new_token(
            token="",
            run_id=UUID("00000000-0000-0000-0000-000000000000"),
        )

        # second token is not empty, this should be processed
        await streaming_handler.push_chunk("second")

        # NOTE: will chnage to sentinel soon to explicitly end the streaming
        await streaming_handler.push_chunk(END_OF_STREAM)

        # wait for the chunks to be processed
        await asyncio.sleep(0.1)

        chunks = await streaming_consumer.get_chunks()
        assert len(chunks) == 2
        assert chunks[0] == ""
        assert chunks[1] == "second"
    finally:
        await streaming_consumer.cancel()


@pytest.mark.asyncio
async def test_on_llm_new_token_with_generation_info():
    """Test on_llm_new_token method with chunk that has generation_info."""
    streaming_handler = StreamingHandler(include_generation_metadata=True)
    streaming_consumer = StreamingConsumer(streaming_handler)

    try:
        test_text = "test token"
        test_generation_info = {"temperature": 0.7, "top_p": 0.95}
        chunk = GenerationChunk(text=test_text, generation_info=test_generation_info)

        await streaming_handler.on_llm_new_token(
            token=test_text,
            chunk=chunk,
            run_id=UUID("00000000-0000-0000-0000-000000000000"),
        )

        # NOTE: end streaming with None
        await streaming_handler.on_llm_end(
            response=None, run_id=UUID("00000000-0000-0000-0000-000000000000")
        )

        chunks = await streaming_consumer.get_chunks()
        assert len(chunks) == 2
        assert chunks[0]["text"] == test_text
        assert chunks[0]["generation_info"] == test_generation_info
        assert chunks[1]["text"] is END_OF_STREAM
        assert chunks[1]["generation_info"] == test_generation_info
    finally:
        await streaming_consumer.cancel()


@pytest.mark.asyncio
async def test_processing_metadata():
    """Test that metadata is properly passed through the processing chain."""
    streaming_handler = StreamingHandler(include_generation_metadata=True)
    streaming_consumer = StreamingConsumer(streaming_handler)

    try:
        streaming_handler.set_pattern(prefix="PREFIX: ", suffix="SUFFIX")

        test_text = "PREFIX: This is a test message SUFFIX"
        test_generation_info = {"temperature": 0.7, "top_p": 0.95}

        await streaming_handler.push_chunk(
            test_text, generation_info=test_generation_info
        )
        await streaming_handler.push_chunk(END_OF_STREAM)  # Signal end of streaming

        chunks = await streaming_consumer.get_chunks()
        assert len(chunks) >= 1
        # NOTE: The suffix is only removed at the end of generation
        assert "This is a test message" in chunks[0]["text"]
        assert chunks[0]["generation_info"] == test_generation_info
    finally:
        await streaming_consumer.cancel()

    streaming_handler = StreamingHandler(include_generation_metadata=True)
    streaming_consumer = StreamingConsumer(streaming_handler)
    try:
        streaming_handler.set_pattern(prefix="PREFIX: ", suffix="SUFFIX")

        await streaming_handler.push_chunk("PRE", generation_info={"part": 1})
        await streaming_handler.push_chunk("FIX: ", generation_info={"part": 2})
        await streaming_handler.push_chunk("Test ", generation_info={"part": 3})
        await streaming_handler.push_chunk("message", generation_info={"part": 4})
        await streaming_handler.push_chunk(" SUFF", generation_info={"part": 5})
        await streaming_handler.push_chunk("IX", generation_info={"part": 6})
        await streaming_handler.push_chunk(END_OF_STREAM)  # End of streaming

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
                assert chunks[i]["generation_info"]["part"] == expected["part"]
    finally:
        await streaming_consumer.cancel()


@pytest.mark.asyncio
async def test_anext_with_dict_end_of_stream_sentinel():
    """Test __anext__ with a dict-wrapped END_OF_STREAM sentinel."""

    streaming_handler = StreamingHandler(include_generation_metadata=True)
    await streaming_handler.queue.put({"text": END_OF_STREAM, "generation_info": {}})
    with pytest.raises(StopAsyncIteration):
        await streaming_handler.__anext__()


@pytest.mark.asyncio
async def test_push_chunk_with_chat_generation_chunk():
    """Test push_chunk with a ChatGenerationChunk."""

    streaming_handler = StreamingHandler()
    consumer = StreamingConsumer(streaming_handler)
    try:
        chat_chunk = ChatGenerationChunk(message=AIMessageChunk(content="chat text"))
        await streaming_handler.push_chunk(chat_chunk)
        await streaming_handler.push_chunk(END_OF_STREAM)
        chunks = await consumer.get_chunks()
        assert chunks == ["chat text"]
    finally:
        await consumer.cancel()


@pytest.mark.asyncio
async def test_push_chunk_with_chat_generation_chunk_with_metadata():
    """Test push_chunk with a ChatGenerationChunk when metadata is included."""

    streaming_handler = StreamingHandler(include_generation_metadata=True)
    consumer = StreamingConsumer(streaming_handler)
    try:
        message_chunk = AIMessageChunk(content="chat text")
        chat_chunk = ChatGenerationChunk(
            message=message_chunk, generation_info={"details": "some details"}
        )
        await streaming_handler.push_chunk(chat_chunk)
        await streaming_handler.push_chunk(END_OF_STREAM)
        chunks = await consumer.get_chunks()
        assert len(chunks) == 2
        assert chunks[0]["text"] == "chat text"
        assert chunks[0]["generation_info"] == {"details": "some details"}
        assert chunks[1]["text"] is END_OF_STREAM
        assert chunks[1]["generation_info"] == {"details": "some details"}
    finally:
        await consumer.cancel()


@pytest.mark.asyncio
async def test_push_chunk_unsupported_type():
    """Test push_chunk with an unsupported data type."""

    streaming_handler = StreamingHandler()
    with pytest.raises(Exception, match="Unsupported chunk type: int"):
        await streaming_handler.push_chunk(123)
    with pytest.raises(Exception, match="Unsupported chunk type: list"):
        await streaming_handler.push_chunk([1, 2])


@pytest.mark.asyncio
async def test_on_llm_new_token_with_chunk_having_none_generation_info():
    """Test on_llm_new_token when chunk.generation_info is None."""
    streaming_handler = StreamingHandler(include_generation_metadata=True)
    consumer = StreamingConsumer(streaming_handler)
    try:
        mock_chunk = GenerationChunk(text="test text", generation_info=None)
        await streaming_handler.on_llm_new_token(
            token="test text",
            chunk=mock_chunk,
            run_id=UUID("00000000-0000-0000-0000-000000000000"),
        )
        await streaming_handler.on_llm_end(
            response=None, run_id=UUID("00000000-0000-0000-0000-000000000000")
        )
        chunks = await consumer.get_chunks()
        assert len(chunks) == 2
        assert chunks[0]["text"] == "test text"
        assert chunks[0]["generation_info"] == {}
        assert chunks[1]["text"] is END_OF_STREAM
        assert chunks[1]["generation_info"] == {}
    finally:
        await consumer.cancel()

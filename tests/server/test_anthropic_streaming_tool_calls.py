# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Verify that tool calls propagate through the streaming path to Anthropic SSE.

Tests two layers:
1. _format_anthropic_sse correctly converts a tool_calls JSON chunk into
   Anthropic-format SSE events (content_block_start/delta/stop for tool_use).
2. The streaming handler receives tool calls extracted from Colang pipeline
   events (BotToolCalls) before END_OF_STREAM, matching the code path in
   LLMRails.generate_async.
"""

import json

import pytest

from nemoguardrails.server.messages import _format_anthropic_sse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TOOL_CALLS = [
    {
        "id": "toolu_abc123",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": json.dumps({"location": "Seattle"}),
        },
    }
]


async def _collect_sse_events(async_gen):
    """Collect all SSE events from an async generator, parsed into dicts."""
    events = []
    async for raw in async_gen:
        for line in raw.strip().split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


async def _async_iter_from_list(items):
    """Turn a plain list into an async iterator."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# 1. _format_anthropic_sse: tool call chunk handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_formatter_text_only():
    """Text-only stream produces text content block and end_turn stop reason."""
    chunks = ["Hello", " world"]
    events = await _collect_sse_events(_format_anthropic_sse(_async_iter_from_list(chunks), model="test-model"))

    text_deltas = [
        e for e in events if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta"
    ]
    assert len(text_deltas) == 2
    assert text_deltas[0]["delta"]["text"] == "Hello"
    assert text_deltas[1]["delta"]["text"] == " world"

    msg_delta = [e for e in events if e.get("type") == "message_delta"][0]
    assert msg_delta["delta"]["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_sse_formatter_tool_calls_from_json_string():
    """A JSON-encoded tool_calls chunk produces Anthropic tool_use SSE blocks."""
    tool_calls_chunk = json.dumps({"tool_calls": SAMPLE_TOOL_CALLS})
    chunks = ["Some text", tool_calls_chunk]

    events = await _collect_sse_events(_format_anthropic_sse(_async_iter_from_list(chunks), model="test-model"))

    tool_starts = [
        e
        for e in events
        if e.get("type") == "content_block_start" and e.get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_starts) == 1
    assert tool_starts[0]["content_block"]["name"] == "get_weather"
    assert tool_starts[0]["content_block"]["id"] == "toolu_abc123"

    tool_deltas = [
        e
        for e in events
        if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert len(tool_deltas) == 1
    parsed_args = json.loads(tool_deltas[0]["delta"]["partial_json"])
    assert parsed_args == {"location": "Seattle"}

    msg_delta = [e for e in events if e.get("type") == "message_delta"][0]
    assert msg_delta["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_sse_formatter_tool_calls_from_dict():
    """A dict chunk with tool_calls key produces Anthropic tool_use SSE blocks."""
    chunks = [{"tool_calls": SAMPLE_TOOL_CALLS}]

    events = await _collect_sse_events(_format_anthropic_sse(_async_iter_from_list(chunks), model="test-model"))

    tool_starts = [
        e
        for e in events
        if e.get("type") == "content_block_start" and e.get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_starts) == 1
    assert tool_starts[0]["content_block"]["name"] == "get_weather"

    msg_delta = [e for e in events if e.get("type") == "message_delta"][0]
    assert msg_delta["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_sse_formatter_multiple_tool_calls():
    """Multiple tool calls get separate content blocks with correct indices."""
    multi_tool_calls = [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "NYC"})},
        },
        {
            "id": "toolu_2",
            "type": "function",
            "function": {"name": "get_time", "arguments": json.dumps({"timezone": "EST"})},
        },
    ]
    chunks = ["Hi", json.dumps({"tool_calls": multi_tool_calls})]

    events = await _collect_sse_events(_format_anthropic_sse(_async_iter_from_list(chunks), model="test-model"))

    tool_starts = [
        e
        for e in events
        if e.get("type") == "content_block_start" and e.get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_starts) == 2
    assert tool_starts[0]["index"] == 1
    assert tool_starts[0]["content_block"]["name"] == "get_weather"
    assert tool_starts[1]["index"] == 2
    assert tool_starts[1]["content_block"]["name"] == "get_time"


# ---------------------------------------------------------------------------
# 2. _stream_llm_call: tool call propagation through streaming handler
#
# In the real flow, _stream_llm_call (actions/llm/utils.py) accumulates
# ToolCall objects from streaming chunks, serializes them as a JSON chunk
# via handler.push_chunk(), then calls handler.finish() which pushes
# END_OF_STREAM. These tests replicate that exact sequence.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_handler_receives_tool_calls():
    """Simulates _stream_llm_call pushing tool calls before finish().

    This mirrors the exact code path at utils.py:135-141.
    """
    from nemoguardrails.streaming import StreamingHandler
    from nemoguardrails.types import ToolCall, ToolCallFunction

    handler = StreamingHandler()

    tool_calls = [
        ToolCall(
            id="call_123",
            type="function",
            function=ToolCallFunction(name="get_weather", arguments={"location": "Seattle"}),
        )
    ]

    await handler.push_chunk("Hello from LLM")

    payload = json.dumps({"tool_calls": [tc.to_dict() for tc in tool_calls]})
    await handler.push_chunk(payload)
    await handler.finish()

    chunks = []
    async for chunk in handler:
        chunks.append(chunk)

    text_chunks = [c for c in chunks if isinstance(c, str) and not c.startswith("{")]
    assert any("Hello from LLM" in c for c in text_chunks)

    tool_call_chunks = []
    for c in chunks:
        if isinstance(c, str):
            try:
                parsed = json.loads(c)
                if isinstance(parsed, dict) and "tool_calls" in parsed:
                    tool_call_chunks.append(parsed)
            except (json.JSONDecodeError, ValueError):
                pass

    assert len(tool_call_chunks) == 1, f"Expected 1 tool_calls chunk, got {len(tool_call_chunks)}. All chunks: {chunks}"
    assert tool_call_chunks[0]["tool_calls"][0]["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_streaming_handler_no_tool_calls_when_none():
    """When the LLM returns no tool calls, no tool_calls chunk appears."""
    from nemoguardrails.streaming import StreamingHandler

    handler = StreamingHandler()

    await handler.push_chunk("Just text")
    await handler.finish()

    chunks = []
    async for chunk in handler:
        chunks.append(chunk)

    for c in chunks:
        if isinstance(c, str):
            try:
                parsed = json.loads(c)
                assert "tool_calls" not in parsed, "Unexpected tool_calls chunk in text-only stream"
            except (json.JSONDecodeError, ValueError):
                pass


@pytest.mark.asyncio
async def test_end_to_end_sse_with_tool_calls():
    """Full pipeline: _stream_llm_call pushes tool calls to streaming handler,
    then _format_anthropic_sse converts to complete Anthropic SSE sequence."""
    from nemoguardrails.streaming import StreamingHandler
    from nemoguardrails.types import ToolCall, ToolCallFunction

    handler = StreamingHandler()

    tool_calls = [
        ToolCall(
            id="toolu_weather1",
            type="function",
            function=ToolCallFunction(
                name="get_weather",
                arguments={"location": "Seattle", "units": "celsius"},
            ),
        )
    ]

    await handler.push_chunk("Here's the weather")
    payload = json.dumps({"tool_calls": [tc.to_dict() for tc in tool_calls]})
    await handler.push_chunk(payload)
    await handler.finish()

    sse_events = await _collect_sse_events(_format_anthropic_sse(handler, model="test-model"))

    msg_start = [e for e in sse_events if e.get("type") == "message_start"]
    assert len(msg_start) == 1
    assert msg_start[0]["message"]["model"] == "test-model"

    text_deltas = [
        e
        for e in sse_events
        if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta"
    ]
    assert len(text_deltas) >= 1
    full_text = "".join(d["delta"]["text"] for d in text_deltas)
    assert "weather" in full_text

    tool_starts = [
        e
        for e in sse_events
        if e.get("type") == "content_block_start" and e.get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_starts) == 1
    assert tool_starts[0]["content_block"]["name"] == "get_weather"
    assert tool_starts[0]["content_block"]["id"] == "toolu_weather1"

    tool_deltas = [
        e
        for e in sse_events
        if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert len(tool_deltas) == 1
    parsed_args = json.loads(tool_deltas[0]["delta"]["partial_json"])
    assert parsed_args["location"] == "Seattle"
    assert parsed_args["units"] == "celsius"

    msg_delta = [e for e in sse_events if e.get("type") == "message_delta"][0]
    assert msg_delta["delta"]["stop_reason"] == "tool_use"

    msg_stop = [e for e in sse_events if e.get("type") == "message_stop"]
    assert len(msg_stop) == 1

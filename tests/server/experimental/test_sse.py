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

import asyncio

import pytest
from openai._streaming import SSEDecoder

from nemoguardrails.server.experimental.provider.sse import (
    AmbiguousServerSentEvent,
    ServerSentEvent,
    ServerSentEventTooLarge,
    TruncatedServerSentEvent,
    iter_sse_events,
)


async def _chunks(*values):
    for value in values:
        yield value


async def _byte_chunks(raw):
    for byte in raw:
        yield bytes([byte])


def test_sse_event_exposes_joined_data_and_last_event_type_without_changing_bytes():
    raw = b"event: first\r\nevent: final\r\ndata: one\r\ndata:two\r\n\r\n"
    event = ServerSentEvent.from_bytes(raw)

    assert event.raw == raw
    assert event.event_type == b"final"
    assert event.data == b"one\ntwo"


@pytest.mark.asyncio
async def test_sse_parser_preserves_events_split_across_arbitrary_chunks():
    raw = b": metadata\n\ndata: first\n\ndata: second\n\n"

    events = [event async for event in iter_sse_events(_chunks(raw[:7], raw[7:19], raw[19:]), max_event_bytes=30)]

    assert [event.raw for event in events] == [b": metadata\n\n", b"data: first\n\n", b"data: second\n\n"]


@pytest.mark.asyncio
async def test_sse_parser_bounds_complete_and_incomplete_events():
    with pytest.raises(ServerSentEventTooLarge):
        _ = [event async for event in iter_sse_events(_chunks(b"data: oversized\n\n"), max_event_bytes=8)]
    with pytest.raises(ServerSentEventTooLarge):
        _ = [event async for event in iter_sse_events(_chunks(b"data: unfinished"), max_event_bytes=8)]


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", [b"\n\n", b"\n\r", b"\n\r\n", b"\r\r", b"\r\r\n", b"\r\n\n", b"\r\n\r", b"\r\n\r\n"])
async def test_all_event_line_endings_at_every_chunk_split(ending):
    blocks = [b"data: one" + ending, b"data: two" + ending]
    raw = b"".join(blocks)

    for split in range(len(raw) + 1):
        events = [event async for event in iter_sse_events(_chunks(raw[:split], b"", raw[split:]))]
        assert [event.raw for event in events] == blocks
        assert [event.data for event in events] == [b"one", b"two"]
        assert ServerSentEvent.from_bytes(blocks[0]) == events[0]


@pytest.mark.asyncio
async def test_bom_is_ignored_only_at_stream_start_and_preserved_in_raw():
    blocks = [b"\xef\xbb\xbfdata: one\n\n", b"data: \xef\xbb\xbfvalue\n\n"]
    raw = b"".join(blocks)

    for split in range(len(raw) + 1):
        events = [event async for event in iter_sse_events(_chunks(raw[:split], raw[split:]))]
        assert [event.raw for event in events] == blocks
        assert [event.data for event in events] == [b"one", b"\xef\xbb\xbfvalue"]
    with pytest.raises(AmbiguousServerSentEvent):
        ServerSentEvent.from_bytes(blocks[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        b"data: one\n\n\xef\xbb\xbfdata: two\n\n",
        b"data: one\n\xef\xbb\xbf\ndata: two\n\n",
        b"\xef\xbb\xbf\xef\xbb\xbfdata: one\n\n",
        b"data: one\n\n\xef\xbb\xbf: comment\n\n",
        b"data: one\r\r\xef\xbb\xbfdata: two\r\r",
        b"data: one\r\n\r\n\xef\xbb\xbfdata: two\r\n\r\n",
        b"data: one\n\n\xef\xbb\xbfdata: two",
    ],
)
async def test_line_leading_bom_after_stream_start_is_rejected_at_every_split(raw):
    """Clients disagree on such lines, so the parser never interprets them."""
    for split in range(len(raw) + 1):
        with pytest.raises(AmbiguousServerSentEvent):
            _ = [event async for event in iter_sse_events(_chunks(raw[:split], raw[split:]))]


@pytest.mark.asyncio
async def test_comments_empty_blocks_and_unknown_fields_preserve_exact_bytes():
    blocks = [b"\n", b": keepalive\r\n\r", b"unknown: value\ndata\n\n"]
    raw = b"".join(blocks)

    events = [event async for event in iter_sse_events(_chunks(*(bytes([byte]) for byte in raw)))]

    assert [event.raw for event in events] == blocks
    assert [event.data for event in events] == [None, None, b""]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tail",
    [
        b"data: partial",
        b"data: partial\n",
        b"data: partial\r",
        b"event: future",
        b"id: 1",
        b"retry: 1",
        b"unknown",
        b" ",
        b": comment\ndata: partial",
    ],
)
async def test_unfinished_field_blocks_are_rejected(tail):
    with pytest.raises(TruncatedServerSentEvent):
        _ = [event async for event in iter_sse_events(_chunks(tail))]
    with pytest.raises(ValueError):
        ServerSentEvent.from_bytes(tail)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tail", [b"", b": comment", b": comment\n", b": comment\r", b": one\n: two", b"\xef\xbb\xbf: comment"]
)
async def test_unfinished_comment_tails_are_discarded(tail):
    events = [event async for event in iter_sse_events(_chunks(tail))]

    assert events == []


@pytest.mark.asyncio
async def test_event_limit_counts_terminators_bom_and_unfinished_lines():
    block = b"\xef\xbb\xbfdata: one\r\n\r\n"
    events = [event async for event in iter_sse_events(_chunks(block + block[3:] * 2), max_event_bytes=len(block))]
    assert len(events) == 3
    for raw in (block, b"x" * len(block)):
        with pytest.raises(ServerSentEventTooLarge):
            _ = [
                event
                async for event in iter_sse_events(
                    _chunks(*(bytes([byte]) for byte in raw)), max_event_bytes=len(block) - 1
                )
            ]


def test_from_bytes_rejects_multiple_blocks():
    with pytest.raises(ValueError, match="exactly one"):
        ServerSentEvent.from_bytes(b"data: one\n\ndata: two\n\n")


@pytest.mark.asyncio
async def test_many_small_events_in_one_large_chunk():
    block = b"data: x\n\n"
    count = 1024 * 1024 // len(block)
    observed = 0

    async for event in iter_sse_events(_chunks(block * count), max_event_bytes=len(block)):
        assert event.raw == block
        observed += 1

    assert observed == count


@pytest.mark.asyncio
async def test_large_event_arriving_one_byte_per_chunk():
    raw = b"data: " + b"x" * (1024 * 1024 - 8) + b"\n\n"

    events = [event async for event in iter_sse_events(_byte_chunks(raw), max_event_bytes=len(raw))]

    assert len(events) == 1
    assert events[0].raw == raw


@pytest.mark.parametrize(
    "raw",
    [
        b"\xef\xbb\xbfdata: one\r\n\r\ndata: two\n\n",
        b"data: a\ndata: b\r\rdata: c\r\n\r\n",
        b": comment\n\nid: 1\nevent: delta\ndata: x\n\nretry: 5\n\n",
        b"data:no-space\n\ndata: \xef\xbb\xbfvalue\n\nunknown: field\ndata\n\n",
        b"Data: wrong-case\n\ndata: right\n\n",
    ],
)
def test_python_sdk_never_sees_data_this_parser_did_not_parse(raw):
    """Compare Python SDK parsing, separately from the Node BOM regression."""

    async def parse():
        return [event.data for event in [event async for event in iter_sse_events(_chunks(raw))]]

    parsed = [data for data in asyncio.run(parse()) if data is not None]
    client = [event.data.encode() for event in SSEDecoder().iter_bytes(iter([raw])) if event.data]
    remaining = iter(parsed)
    assert all(any(data == candidate for candidate in remaining) for data in client)


def _node_like_events(raw: bytes) -> list[bytes]:
    """Model openai-node 7.31.0, which strips a BOM at the start of every line.

    The preserved review evidence shows Node reading a later BOM-prefixed data
    line that a stream-start-only parser ignores. This is a model, not Node.
    """
    events, data = [], []
    for line in raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n"):
        line = line.removeprefix(b"\xef\xbb\xbf")
        if not line:
            if data:
                events.append(b"\n".join(data))
            data = []
            continue
        name, _, value = line.partition(b":")
        if name == b"data":
            data.append(value.removeprefix(b" "))
    return events


@pytest.mark.parametrize(
    "raw",
    [
        b"data: safe\n\n\xef\xbb\xbfdata: EVIL\n\n",
        b"data: safe\r\r\xef\xbb\xbfdata: EVIL\r\r",
        b"data: safe\n\xef\xbb\xbfdata: EVIL\n\n",
        b": note\n\xef\xbb\xbfdata: EVIL\n\n",
        b"\xef\xbb\xbfdata: one\n\ndata: two\n\n",
    ],
)
@pytest.mark.parametrize("split", range(4))
def test_node_like_client_never_sees_data_this_parser_did_not_parse(raw, split):
    """Guard the Node BOM differential: accepted streams expose no extra data."""

    async def parse():
        return [event.data async for event in iter_sse_events(_chunks(raw[:split], raw[split:]))]

    try:
        parsed = [data for data in asyncio.run(parse()) if data is not None]
    except AmbiguousServerSentEvent:
        return
    remaining = iter(parsed)
    assert all(any(data == candidate for candidate in remaining) for data in _node_like_events(raw))


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_event_limit_must_be_a_positive_integer(limit):
    with pytest.raises(ValueError, match="max_event_bytes must be positive"):
        asyncio.run(anext(iter_sse_events(_chunks(b"data: x\n\n"), max_event_bytes=limit)))

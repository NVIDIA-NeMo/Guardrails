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

import pytest

from nemoguardrails.server.experimental.provider.sse import ServerSentEvent, ServerSentEventTooLarge, iter_sse_events


async def _chunks(*values):
    for value in values:
        yield value


def test_sse_event_exposes_joined_data_and_last_event_type_without_changing_bytes():
    raw = b"event: first\r\nevent: final\r\ndata: one\r\ndata:two\r\n\r\n"
    event = ServerSentEvent.from_bytes(raw)

    assert event.raw == raw
    assert event.event_type == b"final"
    assert event.data == b"one\ntwo"


@pytest.mark.asyncio
async def test_sse_parser_preserves_events_split_across_arbitrary_chunks():
    raw = b": metadata\n\ndata: first\n\ndata: second"

    events = [event async for event in iter_sse_events(_chunks(raw[:7], raw[7:19], raw[19:]), max_event_bytes=30)]

    assert [event.raw for event in events] == [b": metadata\n\n", b"data: first\n\n", b"data: second"]


@pytest.mark.asyncio
async def test_sse_parser_bounds_complete_and_incomplete_events():
    with pytest.raises(ServerSentEventTooLarge):
        _ = [event async for event in iter_sse_events(_chunks(b"data: oversized\n\n"), max_event_bytes=8)]
    with pytest.raises(ServerSentEventTooLarge):
        _ = [event async for event in iter_sse_events(_chunks(b"data: unfinished"), max_event_bytes=8)]

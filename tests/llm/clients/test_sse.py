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

import json

import pytest

from nemoguardrails.llm.clients._sse import ServerSentEvent, SSEDecoder


def _decode_lines(lines: list[str]) -> list[ServerSentEvent]:
    decoder = SSEDecoder()
    events = []
    for line in lines:
        sse = decoder.decode(line)
        if sse is not None:
            events.append(sse)
    return events


class TestBasic:
    def test_event_and_data(self):
        events = _decode_lines(["event: completion", 'data: {"foo":true}', ""])
        assert len(events) == 1
        assert events[0].event == "completion"
        assert events[0].json() == {"foo": True}

    def test_data_missing_event(self):
        events = _decode_lines(['data: {"foo":true}', ""])
        assert len(events) == 1
        assert events[0].event == "message"
        assert events[0].json() == {"foo": True}

    def test_event_missing_data(self):
        events = _decode_lines(["event: ping", ""])
        assert len(events) == 1
        assert events[0].event == "ping"
        assert events[0].data == ""


class TestMultipleEvents:
    def test_two_events_no_data(self):
        events = _decode_lines(["event: ping", "", "event: completion", ""])
        assert len(events) == 2
        assert events[0].event == "ping"
        assert events[0].data == ""
        assert events[1].event == "completion"
        assert events[1].data == ""

    def test_two_events_with_data(self):
        events = _decode_lines(
            [
                "event: ping",
                'data: {"foo":true}',
                "",
                "event: completion",
                'data: {"bar":false}',
                "",
            ]
        )
        assert len(events) == 2
        assert events[0].event == "ping"
        assert events[0].json() == {"foo": True}
        assert events[1].event == "completion"
        assert events[1].json() == {"bar": False}


class TestMultiLineData:
    def test_multiple_data_lines(self):
        events = _decode_lines(["event: ping", "data: {", 'data: "foo":', "data: true}", ""])
        assert len(events) == 1
        assert events[0].event == "ping"
        assert events[0].json() == {"foo": True}

    def test_multiple_data_lines_with_empty_data(self):
        events = _decode_lines(
            [
                "event: ping",
                "data: {",
                'data: "foo":',
                "data: ",
                "data:",
                "data: true}",
                "",
            ]
        )
        assert len(events) == 1
        assert events[0].event == "ping"
        assert events[0].data == '{\n"foo":\n\n\ntrue}'
        assert events[0].json() == {"foo": True}

    def test_json_escaped_double_newline(self):
        events = _decode_lines(["event: ping", 'data: {"foo": "my long\\n\\ncontent"}', ""])
        assert len(events) == 1
        assert events[0].event == "ping"
        assert events[0].json() == {"foo": "my long\n\ncontent"}


class TestSSEFields:
    def test_comment_ignored(self):
        events = _decode_lines([": this is a comment", "data: hello", ""])
        assert len(events) == 1
        assert events[0].data == "hello"

    def test_id_field(self):
        events = _decode_lines(["id: evt-42", "data: x", ""])
        assert len(events) == 1
        assert events[0].id == "evt-42"

    def test_id_with_null_byte_rejected(self):
        events = _decode_lines(["id: has\0null", "data: x", ""])
        assert len(events) == 1
        assert events[0].id == ""

    def test_id_persists_across_events(self):
        events = _decode_lines(["id: persistent", "data: first", "", "data: second", ""])
        assert len(events) == 2
        assert events[0].id == "persistent"
        assert events[1].id == "persistent"

    def test_retry_field(self):
        events = _decode_lines(["retry: 5000", "data: x", ""])
        assert len(events) == 1
        assert events[0].retry == 5000

    def test_retry_non_integer_ignored(self):
        events = _decode_lines(["retry: abc", "data: x", ""])
        assert len(events) == 1
        assert events[0].retry is None

    def test_unknown_field_ignored(self):
        events = _decode_lines(["unknown: value", "data: x", ""])
        assert len(events) == 1
        assert events[0].data == "x"

    def test_event_resets_after_dispatch(self):
        events = _decode_lines(["event: custom", "data: first", "", "data: second", ""])
        assert len(events) == 2
        assert events[0].event == "custom"
        assert events[1].event == "message"


class TestEdgeCases:
    def test_consecutive_blank_lines(self):
        events = _decode_lines(["", "", ""])
        assert len(events) == 0

    def test_leading_space_stripped(self):
        events = _decode_lines(["data: hello", ""])
        assert events[0].data == "hello"

    def test_only_first_space_stripped(self):
        events = _decode_lines(["data:  two spaces", ""])
        assert events[0].data == " two spaces"

    def test_no_space_after_colon(self):
        events = _decode_lines(["data:hello", ""])
        assert events[0].data == "hello"

    def test_flush_dispatches_buffered(self):
        decoder = SSEDecoder()
        decoder.decode("data: buffered")
        sse = decoder.decode("")
        assert sse is not None
        assert sse.data == "buffered"

    def test_field_no_value(self):
        events = _decode_lines(["data", ""])
        assert len(events) == 1
        assert events[0].data == ""

    def test_id_only_does_not_dispatch(self):
        events = _decode_lines(["id: foo", "", "", ""])
        assert events == []

    def test_id_only_then_data_dispatches_once(self):
        events = _decode_lines(["id: foo", "", "", "data: real", ""])
        assert len(events) == 1
        assert events[0].data == "real"
        assert events[0].id == "foo"

    def test_id_after_dispatch_does_not_re_dispatch(self):
        events = _decode_lines(["id: persistent", "data: first", "", "", ""])
        assert len(events) == 1
        assert events[0].data == "first"
        assert events[0].id == "persistent"


class TestServerSentEvent:
    def test_defaults(self):
        sse = ServerSentEvent()
        assert sse.event == "message"
        assert sse.data == ""
        assert sse.id == ""
        assert sse.retry is None

    def test_json(self):
        sse = ServerSentEvent(data='{"key": "value"}')
        assert sse.json() == {"key": "value"}

    def test_json_raises_on_invalid(self):
        sse = ServerSentEvent(data="not json")
        with pytest.raises(json.JSONDecodeError):
            sse.json()

    def test_repr(self):
        sse = ServerSentEvent()
        assert repr(sse) == "ServerSentEvent(event='message')"

        sse = ServerSentEvent(data="data", retry=3, id="id", event="event")
        assert repr(sse) == "ServerSentEvent(event='event', data='data', id='id', retry=3)"

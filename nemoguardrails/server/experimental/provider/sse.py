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

"""Parse Server-Sent Events without provider-specific knowledge.

The parser preserves complete upstream events byte-for-byte while exposing the
event type and joined data fields needed by provider adapters.
"""

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass


class ServerSentEventTooLarge(ValueError):
    """Report an SSE event that exceeds the configured buffering limit."""


class TruncatedServerSentEvent(ValueError):
    """Report an unfinished SSE block containing fields at end of stream."""


@dataclass(frozen=True, slots=True)
class ServerSentEvent:
    """Represent one complete SSE event.

    Attributes:
        raw: Original event bytes, including its terminating blank line when
            supplied by the upstream server.
        lines: Event field and comment lines without line endings.
    """

    raw: bytes
    lines: tuple[bytes, ...]

    @classmethod
    def from_bytes(cls, raw: bytes) -> "ServerSentEvent":
        """Create an event while retaining its original representation.

        Args:
            raw: Exactly one complete SSE block, without stream-start BOM handling.

        Returns:
            The parsed event representation.
        """

        framer = _SSEFramer(max_event_bytes=max(1, len(raw)), stream_start=False)
        events = [*framer.feed(raw), *framer.finish()]
        if len(events) != 1 or events[0].raw != raw:
            raise ValueError("Expected exactly one complete SSE block.")
        return events[0]

    @property
    def data(self) -> bytes | None:
        """Return joined SSE data fields, or ``None`` when none are present."""

        values = []
        for line in self.lines:
            name, separator, value = line.partition(b":")
            if separator and name == b"data":
                values.append(value[1:] if value.startswith(b" ") else value)
            elif not separator and name == b"data":
                values.append(b"")
        return b"\n".join(values) if values else None

    @property
    def event_type(self) -> bytes | None:
        """Return the last SSE event field, or ``None`` when it is absent."""

        value = None
        for line in self.lines:
            name, separator, field_value = line.partition(b":")
            if name == b"event":
                value = field_value[1:] if separator and field_value.startswith(b" ") else field_value
        return value


class _SSEFramer:
    def __init__(self, *, max_event_bytes: int, stream_start: bool = True):
        self.max_event_bytes = max_event_bytes
        self.raw = bytearray()
        self.lines: list[bytes] = []
        self.line_start = 0
        self.pending_cr = False
        self.stream_start = stream_start

    def _append(self, value: int) -> None:
        if len(self.raw) >= self.max_event_bytes:
            raise ServerSentEventTooLarge
        self.raw.append(value)

    def _field_line(self, line: bytes) -> bytes:
        if self.stream_start:
            self.stream_start = False
            return line.removeprefix(b"\xef\xbb\xbf")
        return line

    def _end_line(self, ending_length: int) -> ServerSentEvent | None:
        line = self._field_line(bytes(self.raw[self.line_start : -ending_length]))
        if line:
            self.lines.append(line)
            self.line_start = len(self.raw)
            return None
        event = ServerSentEvent(bytes(self.raw), tuple(self.lines))
        self.raw.clear()
        self.lines.clear()
        self.line_start = 0
        return event

    def feed(self, chunk: bytes) -> Iterator[ServerSentEvent]:
        for value in chunk:
            if self.pending_cr:
                self.pending_cr = False
                if value == 10:
                    self._append(value)
                    event = self._end_line(2)
                    if event is not None:
                        yield event
                    continue
                event = self._end_line(1)
                if event is not None:
                    yield event
            self._append(value)
            if value == 13:
                self.pending_cr = True
            elif value == 10:
                event = self._end_line(1)
                if event is not None:
                    yield event

    def finish(self) -> Iterator[ServerSentEvent]:
        if self.pending_cr:
            self.pending_cr = False
            event = self._end_line(1)
            if event is not None:
                yield event
        if self.raw:
            tail = self._field_line(bytes(self.raw[self.line_start :]))
            if any(line and not line.startswith(b":") for line in (*self.lines, tail)):
                raise TruncatedServerSentEvent("The provider stream ends with an unfinished SSE field block.")
            self.raw.clear()
            self.lines.clear()
            self.line_start = 0


async def iter_sse_events(
    source: AsyncIterator[bytes],
    *,
    max_event_bytes: int = 1024 * 1024,
) -> AsyncIterator[ServerSentEvent]:
    """Yield complete SSE blocks from arbitrary HTTP byte chunks.

    Preserve raw bytes, including an initial BOM ignored only for field parsing.
    Drop unfinished comment tails; reject all other unfinished field blocks.

    Args:
        source: Raw upstream response byte stream.
        max_event_bytes: Maximum bytes buffered for one event.

    Yields:
        Complete SSE events in upstream order.

    Raises:
        ValueError: If ``max_event_bytes`` is not positive.
        ServerSentEventTooLarge: If one event exceeds the limit.
        TruncatedServerSentEvent: If the stream ends in an unfinished field block.
    """

    if type(max_event_bytes) is not int or max_event_bytes <= 0:
        raise ValueError("max_event_bytes must be positive.")

    framer = _SSEFramer(max_event_bytes=max_event_bytes)
    async for chunk in source:
        for event in framer.feed(chunk):
            yield event
    for event in framer.finish():
        yield event

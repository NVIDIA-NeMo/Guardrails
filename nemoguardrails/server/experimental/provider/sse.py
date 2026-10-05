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

from collections.abc import AsyncIterator
from dataclasses import dataclass


class ServerSentEventTooLarge(ValueError):
    """Report an SSE event that exceeds the configured buffering limit."""


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
            raw: Complete SSE event bytes.

        Returns:
            The parsed event representation.
        """

        body = raw
        for separator in (b"\r\n\r\n", b"\n\n", b"\r\r"):
            if body.endswith(separator):
                body = body[: -len(separator)]
                break
        return cls(raw=raw, lines=tuple(body.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")))

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


def _event_boundary(buffer: bytes) -> tuple[int, int] | None:
    boundaries = []
    for separator in (b"\r\n\r\n", b"\n\n", b"\r\r"):
        index = buffer.find(separator)
        if index >= 0:
            boundaries.append((index, len(separator)))
    return min(boundaries) if boundaries else None


async def iter_sse_events(
    source: AsyncIterator[bytes],
    *,
    max_event_bytes: int = 1024 * 1024,
) -> AsyncIterator[ServerSentEvent]:
    """Yield complete SSE events from arbitrary HTTP byte chunks.

    Args:
        source: Raw upstream response byte stream.
        max_event_bytes: Maximum bytes buffered for one event.

    Yields:
        Complete SSE events in upstream order.

    Raises:
        ValueError: If ``max_event_bytes`` is not positive.
        ServerSentEventTooLarge: If one event exceeds the limit.
    """

    if max_event_bytes <= 0:
        raise ValueError("max_event_bytes must be positive.")

    buffer = b""
    async for chunk in source:
        buffer += chunk
        while boundary := _event_boundary(buffer):
            index, separator_length = boundary
            event_length = index + separator_length
            if event_length > max_event_bytes:
                raise ServerSentEventTooLarge
            raw, buffer = buffer[:event_length], buffer[event_length:]
            yield ServerSentEvent.from_bytes(raw)
        if len(buffer) > max_event_bytes:
            raise ServerSentEventTooLarge

    if buffer:
        yield ServerSentEvent.from_bytes(buffer)

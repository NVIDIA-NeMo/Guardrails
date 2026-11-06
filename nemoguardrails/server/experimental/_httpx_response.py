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

"""Own retryable HTTPX response cleanup independently of its closed flag."""

from collections.abc import AsyncIterator

import anyio
import httpx

from nemoguardrails.server.experimental._stream_cleanup import StreamClose, close_stream


class _OwnedHttpxStream(httpx.AsyncByteStream):
    """Delegate raw reads while routing HTTPX's close through the retained owner."""

    def __init__(self, resource: httpx.AsyncByteStream, close: StreamClose):
        self._resource = resource
        self._close = close

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._resource.__aiter__()

    async def aclose(self) -> None:
        """Attempt the owner's close with the shared shielded cleanup budget."""

        await close_stream(None, close=self._close)


class _HttpxResponseBody(AsyncIterator[bytes]):
    """Bound explicit and automatic EOF cleanup while keeping failures retryable.

    HTTPX marks its response closed before awaiting the stream. Retaining the
    resource lets later attempts finish even if that await timed out or failed.
    The stream wrapper also bounds HTTPX's automatic close on raw-iterator EOF,
    preserving its byte accounting without shielding provider reads.
    """

    def __init__(self, response: httpx.Response):
        """Retain the original resource and wrap the response stream before raw reads."""

        if not isinstance(response.stream, httpx.AsyncByteStream):
            raise TypeError("The provider response must have an asynchronous byte stream.")
        self._response = response
        self._resource = response.stream
        self._closed = False
        self._close_lock = anyio.Lock()
        response.stream = _OwnedHttpxStream(self._resource, self._close_resource)
        self._source = response.aiter_raw().__aiter__()

    def __aiter__(self) -> "_HttpxResponseBody":
        return self

    async def __anext__(self) -> bytes:
        """Relay raw bytes and attempt bounded cleanup on EOF or a read failure."""

        try:
            return await self._source.__anext__()
        except BaseException:
            await self.aclose()
            raise

    async def _close_resource(self) -> None:
        """Serialize resource closes and record completion only after success."""

        async with self._close_lock:
            if not self._closed:
                await self._resource.aclose()
                self._closed = True

    async def aclose(self) -> None:
        """Attempt bounded response cleanup, retaining unfinished resource closure for retry."""

        await close_stream(self._response, close=self._close_resource)

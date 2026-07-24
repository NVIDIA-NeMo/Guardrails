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

"""Deterministic HTTP test doubles for NeMo Guardrails tests."""

import json
from collections import deque
from collections.abc import Iterable
from typing import Any, Mapping

from nemoguardrails.http.types import HTTPRequest, HTTPResponse


class RecordingHTTPClient:
    """Record requests and return queued responses or exceptions in order."""

    def __init__(self, responses: Iterable[HTTPResponse | BaseException] = ()):
        """Initialize the client with an optional response sequence."""

        self.requests: list[HTTPRequest] = []
        self._responses = deque(responses)
        self.close_calls = 0

    def add_response(self, response: HTTPResponse | BaseException) -> None:
        self._responses.append(response)

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        content: bytes | str | None = None,
        timeout: float | None = None,
    ) -> HTTPResponse:
        """Record a request and consume the next queued result.

        Raises:
            RuntimeError: If no queued result is available.
            BaseException: The queued exception, when the next result is an
                exception instance.
        """

        self.requests.append(
            HTTPRequest(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json,
                content=content,
                timeout=timeout,
            )
        )
        if not self._responses:
            raise RuntimeError("No HTTP responses available")
        response = self._responses.popleft()
        if isinstance(response, BaseException):
            raise response
        return response

    async def close(self) -> None:
        """Record a close call without discarding captured requests."""

        self.close_calls += 1


class RecordedHTTPResponses:
    """Register response payloads and verify that each registered URL is requested."""

    def __init__(self):
        self.client = RecordingHTTPClient()
        self.urls: list[str] = []

    def post(self, url: str, *, payload: Any = None, status: int = 200, body: str | None = None) -> None:
        content = body.encode() if body is not None else json.dumps(payload).encode()
        self.client.add_response(HTTPResponse(status_code=status, content=content))
        self.urls.append(url)

    def __enter__(self) -> "RecordedHTTPResponses":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if exc_type is None:
            assert set(self.urls) == {request.url for request in self.client.requests}


__all__ = ["RecordedHTTPResponses", "RecordingHTTPClient"]

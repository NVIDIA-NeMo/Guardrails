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
from typing import Any

from nemoguardrails.http import HTTPResponse
from nemoguardrails.testing import RecordingHTTPClient


class RecordedHTTPResponses:
    """Register responses and verify the exact requests made by a test."""

    def __init__(self):
        self.client = RecordingHTTPClient()
        self.expected_requests: list[tuple[str, str]] = []

    def post(
        self,
        url: str,
        *,
        payload: Any = None,
        status: int = 200,
        body: str | None = None,
        headers: dict[str, str] | None = None,
        content_type: str | None = None,
        exception: BaseException | None = None,
        times: int = 1,
    ) -> None:
        if exception is not None:
            response = exception
        else:
            response_headers = dict(headers or {})
            if content_type is not None:
                response_headers["Content-Type"] = content_type
            content = body.encode() if body is not None else json.dumps(payload).encode()
            response = HTTPResponse(status_code=status, headers=response_headers, content=content)

        for _ in range(times):
            self.client.add_response(response)
            self.expected_requests.append(("POST", url))

    def __enter__(self) -> "RecordedHTTPResponses":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if exc_type is None:
            actual_requests = [(request.method, request.url) for request in self.client.requests]
            assert self.expected_requests == actual_requests, (self.expected_requests, actual_requests)

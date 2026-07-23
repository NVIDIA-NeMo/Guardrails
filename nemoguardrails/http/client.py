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

"""Protocols implemented by transport-neutral asynchronous HTTP clients."""

from typing import Any, Mapping, Protocol, runtime_checkable

from nemoguardrails.http.types import HTTPResponse


@runtime_checkable
class HTTPClient(Protocol):
    """Send asynchronous HTTP requests without exposing transport-specific types."""

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
        """Send one request and return a response whose content is fully owned.

        Args:
            method: HTTP method, case-insensitive.
            url: Absolute request URL.
            headers: Optional request headers.
            params: Optional query parameters.
            json: Optional JSON-serializable request body.
            content: Optional raw request body.
            timeout: Optional total request timeout in seconds.

        Returns:
            A transport-neutral response containing materialized body bytes.
        """

        ...


@runtime_checkable
class ManagedHTTPClient(HTTPClient, Protocol):
    """An HTTP client that owns resources requiring asynchronous cleanup."""

    async def close(self) -> None:
        """Release resources owned by the client.

        Implementations must make repeated calls safe.
        """

        ...

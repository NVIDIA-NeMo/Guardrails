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

"""Transport-neutral exceptions raised by the outbound HTTP subsystem."""

from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from nemoguardrails.http.types import HTTPRequest, HTTPResponse


def _safe_url(url: str) -> str:
    parts = urlsplit(url)
    hostname = parts.hostname
    if hostname is None:
        netloc = parts.netloc.rsplit("@", 1)[-1]
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    host = f"[{hostname}]" if ":" in hostname else hostname
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


class HTTPClientError(Exception):
    """Base class for outbound HTTP failures.

    ``retry_count`` records how many retries completed before the error was
    returned to the caller.
    """

    def __init__(self, *args: object):
        super().__init__(*args)
        self.retry_count = 0


class HTTPConnectionError(HTTPClientError):
    """Raised when the transport cannot establish or maintain a connection."""


class HTTPTimeoutError(HTTPClientError):
    """Raised when an outbound request exceeds its total timeout."""


class HTTPStatusError(HTTPClientError):
    """Raised for an unsuccessful HTTP status.

    Attributes:
        response: The materialized response that produced the error.
        request: Request metadata when supplied by the caller.
    """

    def __init__(self, response: "HTTPResponse", request: "HTTPRequest | None" = None):
        message = f"HTTP request failed with status {response.status_code}"
        if request is not None:
            message = f"{message}: {request.method.upper()} {_safe_url(request.url)}"
        super().__init__(message)
        self.response = response
        self.request = request


class HTTPResponseDecodeError(HTTPClientError):
    """Raised when a response body cannot be decoded as JSON."""

    def __init__(self, response: "HTTPResponse"):
        super().__init__(f"HTTP response body is not valid JSON (status {response.status_code})")
        self.response = response

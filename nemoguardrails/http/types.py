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

"""Dependency-free request and response values for outbound HTTP calls."""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from nemoguardrails.http.errors import HTTPResponseDecodeError, HTTPStatusError


@dataclass(frozen=True)
class HTTPTLSConfig:
    """Configure TLS verification and optional mutual authentication.

    A custom CA bundle applies only when verification is enabled. Client
    certificate and key paths must be supplied together.
    """

    verify: bool = True
    ca_bundle: str | None = None
    client_certificate: str | None = None
    client_key: str | None = None

    def __post_init__(self) -> None:
        if bool(self.client_certificate) != bool(self.client_key):
            raise ValueError("client_certificate and client_key must be configured together")


@dataclass(frozen=True)
class HTTPRequest:
    """Describe an outbound HTTP request without transport-specific objects.

    The value is suitable for status-error context and deterministic test
    assertions. It does not perform a request itself.
    """

    method: str
    url: str
    headers: Mapping[str, str] | None = None
    params: Mapping[str, Any] | None = None
    json: Any = None
    content: bytes | str | None = None
    timeout: float | None = None


@dataclass(frozen=True)
class HTTPResponse:
    """A materialized, transport-neutral HTTP response.

    ``content`` owns the response bytes, so callers can decode the body after
    the transport or client context has closed. ``extensions`` carries optional
    non-portable metadata without changing the core contract.
    """

    status_code: int
    headers: Mapping[str, str] = field(default_factory=dict)
    content: bytes = b""
    extensions: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Return whether the status code is in the 2xx range."""

        return 200 <= self.status_code < 300

    @property
    def text(self) -> str:
        """Decode the response body using its declared charset or UTF-8."""

        content_type = next(
            (value for name, value in self.headers.items() if name.lower() == "content-type"),
            "",
        )
        charset_match = re.search(r"charset\s*=\s*[\"']?([^;\s\"']+)", content_type, re.IGNORECASE)
        encoding = charset_match.group(1) if charset_match is not None else "utf-8"
        try:
            return self.content.decode(encoding, errors="replace")
        except LookupError:
            return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        """Decode the response body as JSON.

        Raises:
            HTTPResponseDecodeError: If the body is not valid JSON.
        """

        try:
            return json.loads(self.content)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise HTTPResponseDecodeError(self) from error

    def raise_for_status(self, request: HTTPRequest | None = None) -> None:
        """Raise :class:`HTTPStatusError` for status codes of 400 or greater."""

        if self.status_code >= 400:
            raise HTTPStatusError(self, request)

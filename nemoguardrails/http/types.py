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
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from nemoguardrails.http.errors import HTTPResponseDecodeError, HTTPStatusError


@dataclass(frozen=True)
class HTTPRequest:
    method: str
    url: str
    headers: Mapping[str, str] | None = None
    params: Mapping[str, Any] | None = None
    json: Any = None
    content: bytes | str | None = None
    timeout: float | None = None


@dataclass(frozen=True)
class HTTPResponse:
    status_code: int
    headers: Mapping[str, str] = field(default_factory=dict)
    content: bytes = b""
    extensions: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def text(self) -> str:
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
        try:
            return json.loads(self.content)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise HTTPResponseDecodeError(self) from error

    def raise_for_status(self, request: HTTPRequest | None = None) -> None:
        if self.status_code >= 400:
            raise HTTPStatusError(self, request)

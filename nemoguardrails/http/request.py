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

from collections.abc import Callable
from typing import Any, Mapping

from nemoguardrails.http.client import HTTPClient
from nemoguardrails.http.runtime import _resolve_http_client, create_http_client
from nemoguardrails.http.types import HTTPRequest, HTTPResponse


async def http_call(
    client: HTTPClient | None,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, Any] | None = None,
    json: Any = None,
    content: bytes | str | None = None,
    timeout: float | None = None,
    raise_for_status: bool = True,
    factory: Callable[[], HTTPClient] = create_http_client,
) -> HTTPResponse:
    async with _resolve_http_client(client, factory=factory) as resolved:
        response = await resolved.request(
            method,
            url,
            headers=headers,
            params=params,
            json=json,
            content=content,
            timeout=timeout,
        )
    if raise_for_status:
        response.raise_for_status(
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
    return response

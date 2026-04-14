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
import logging
import re
from typing import Any, AsyncIterator, Dict, Optional

from nemoguardrails.exceptions import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMClientError,
    LLMContextWindowError,
    LLMRateLimitError,
    LLMServerError,
    LLMUnsupportedParamsError,
)

log = logging.getLogger(__name__)

_CONTEXT_WINDOW_KEYWORDS = [
    "context length",
    "context_length",
    "context window",
    "maximum token",
    "max_tokens",
    "too many tokens",
    "token limit",
]

_UNSUPPORTED_PARAMS_KEYWORDS = [
    "unsupported parameter",
    "is not supported",
    "not allowed",
    "unknown parameter",
    "unrecognized parameter",
]

_SECRET_PATTERN = re.compile(r"(sk-|nvapi-|key-|bearer\s+)\S+", re.IGNORECASE)


def _redact_secrets(text: str) -> str:
    return _SECRET_PATTERN.sub(lambda m: m.group(1) + "***", text)


class BaseClient:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        try:
            import httpx  # noqa: F401
        except ImportError:
            raise ImportError("httpx is required for the default framework. Install it with: pip install httpx")

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client = None

    async def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _apost(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = await self._get_client()
        response = await client.post(
            f"{self._base_url}{path}",
            json=payload,
            headers=self._build_headers(),
        )
        if response.status_code >= 400:
            self._raise_for_status(response.status_code, response.text, response.headers)
        return response.json()

    async def _apost_stream(self, path: str, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        client = await self._get_client()
        async with client.stream(
            "POST",
            f"{self._base_url}{path}",
            json=payload,
            headers=self._build_headers(),
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                self._raise_for_status(response.status_code, response.text, response.headers)
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    log.warning("Failed to parse SSE chunk: %s", data)

    @staticmethod
    def _raise_for_status(status_code: int, body: str, headers: Any) -> None:
        error_message = ""
        error_type = None
        error_code = None

        try:
            data = json.loads(body)
            error_obj = data.get("error", {})
            if isinstance(error_obj, dict):
                error_message = error_obj.get("message", "")
                error_type = error_obj.get("type")
                error_code = error_obj.get("code")
            elif isinstance(error_obj, str):
                error_message = error_obj
            if not error_message:
                error_message = data.get("message") or data.get("detail") or ""
        except (json.JSONDecodeError, AttributeError):
            error_message = body or ""

        if not error_message:
            error_message = f"HTTP {status_code}"

        error_message = _redact_secrets(error_message)
        msg_lower = error_message.lower()

        if status_code in (401, 403):
            raise LLMAuthenticationError(status_code, error_message, error_type, error_code)

        if status_code == 429:
            retry_after = None
            raw = headers.get("retry-after")
            if raw:
                try:
                    retry_after = float(raw)
                except ValueError:
                    pass
            raise LLMRateLimitError(status_code, error_message, error_type, error_code, retry_after)

        if status_code == 400 or status_code == 422:
            if any(kw in msg_lower for kw in _CONTEXT_WINDOW_KEYWORDS):
                raise LLMContextWindowError(status_code, error_message, error_type, error_code)
            if any(kw in msg_lower for kw in _UNSUPPORTED_PARAMS_KEYWORDS):
                raise LLMUnsupportedParamsError(status_code, error_message, error_type, error_code)
            raise LLMBadRequestError(status_code, error_message, error_type, error_code)

        if status_code >= 500:
            raise LLMServerError(status_code, error_message, error_type, error_code)

        raise LLMClientError(status_code, error_message, error_type, error_code)

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

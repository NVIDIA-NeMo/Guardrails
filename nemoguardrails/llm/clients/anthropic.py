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
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from nemoguardrails.exceptions import (
    LLMConnectionError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from nemoguardrails.llm.clients._errors import raise_for_status
from nemoguardrails.llm.clients._sse import SSEDecoder
from nemoguardrails.llm.clients.base import BaseClient, HTTPResponse

log = logging.getLogger(__name__)


class AnthropicClient(BaseClient):
    _ROUTE = "/messages"

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        *,
        anthropic_version: str = "2023-06-01",
        **kwargs: Any,
    ):
        custom_headers = dict(kwargs.pop("custom_headers", None) or {})
        if api_key:
            custom_headers.setdefault("x-api-key", api_key)
        custom_headers.setdefault("anthropic-version", anthropic_version)

        super().__init__(base_url=base_url, api_key=None, custom_headers=custom_headers, **kwargs)

    @property
    def provider_url(self) -> Optional[str]:
        return self._base_url

    def _build_payload(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system
        if stop_sequences:
            payload["stop_sequences"] = stop_sequences
        if stream:
            payload["stream"] = True
        payload.update(kwargs)
        return payload

    async def create_message(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> HTTPResponse:
        payload = self._build_payload(
            model, messages, max_tokens=max_tokens, system=system, stop_sequences=stop_sequences, **kwargs
        )
        return await self._apost(self._ROUTE, payload)

    async def stream_message(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[HTTPResponse, None]:
        payload = self._build_payload(
            model,
            messages,
            max_tokens=max_tokens,
            system=system,
            stop_sequences=stop_sequences,
            stream=True,
            **kwargs,
        )
        gen = self._apost_stream_anthropic(self._ROUTE, payload)
        try:
            async for chunk in gen:
                yield chunk
        finally:
            await gen.aclose()

    async def _apost_stream_anthropic(self, path: str, payload: Dict[str, Any]) -> AsyncGenerator[HTTPResponse, None]:
        retries_remaining = self._max_retries
        retries_attempted = 0
        ctx = self._error_context()

        while True:
            first_yielded = False
            try:
                async with self._client.stream(
                    "POST",
                    f"{self._base_url}{path}",
                    json=payload,
                    headers=self._build_headers(),
                    params=self._custom_query or None,
                ) as response:
                    if self._should_retry(response.status_code, response.headers) and retries_remaining > 0:
                        await response.aread()
                        await self._sleep_for_retry(retries_attempted, response.headers)
                        retries_remaining -= 1
                        retries_attempted += 1
                        continue

                    if response.status_code >= 400:
                        await response.aread()
                        raise_for_status(response.status_code, response.text, response.headers, ctx)

                    response_headers = dict(response.headers)
                    response_status = response.status_code
                    decoder = SSEDecoder()
                    async for line in response.aiter_lines():
                        sse = decoder.decode(line)
                        if sse is None:
                            continue

                        if sse.event == "message_stop":
                            return

                        if sse.event == "ping":
                            continue

                        try:
                            parsed = sse.json()
                        except json.JSONDecodeError as err:
                            raise LLMResponseValidationError(
                                f"Malformed SSE chunk: {sse.data[:200]!r}: {err}",
                                response_data=None,
                                **ctx.as_kwargs(),
                            ) from err

                        if isinstance(parsed, dict) and parsed.get("type") == "error":
                            error_msg = parsed.get("error", {})
                            raise LLMResponseValidationError(
                                f"Anthropic stream error: {error_msg}",
                                response_data=parsed,
                                **ctx.as_kwargs(),
                            )

                        parsed["_event_type"] = sse.event
                        first_yielded = True
                        yield HTTPResponse(body=parsed, headers=response_headers, status_code=response_status)

                    sse = decoder.decode("")
                    if sse is not None and sse.event != "message_stop":
                        try:
                            parsed = sse.json()
                        except json.JSONDecodeError as err:
                            raise LLMResponseValidationError(
                                f"Malformed trailing SSE chunk: {sse.data[:200]!r}: {err}",
                                response_data=None,
                                **ctx.as_kwargs(),
                            ) from err
                        parsed["_event_type"] = sse.event
                        yield HTTPResponse(body=parsed, headers=response_headers, status_code=response_status)
                    return

            except httpx.TimeoutException as err:
                if first_yielded or retries_remaining <= 0:
                    raise LLMTimeoutError(0, f"Request timed out: {err}", **ctx.as_kwargs()) from err
                await self._sleep_for_retry(retries_attempted)
                retries_remaining -= 1
                retries_attempted += 1
                continue
            except httpx.NetworkError as err:
                if first_yielded or retries_remaining <= 0:
                    raise LLMConnectionError(0, f"Connection error: {err}", **ctx.as_kwargs()) from err
                await self._sleep_for_retry(retries_attempted)
                retries_remaining -= 1
                retries_attempted += 1
                continue
            except RuntimeError as err:
                from nemoguardrails.llm.clients.base import _is_stale_loop_error

                if not _is_stale_loop_error(err):
                    raise
                if first_yielded or retries_remaining <= 0:
                    raise LLMConnectionError(0, f"Stale event loop: {err}", **ctx.as_kwargs()) from err
                log.warning("Retrying after stale event loop binding: %s", err)
                await self._sleep_for_retry(retries_attempted)
                retries_remaining -= 1
                retries_attempted += 1
                continue

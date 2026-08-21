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

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from nemoguardrails.exceptions import LLMResponseValidationError
from nemoguardrails.llm.clients._errors import ErrorContext
from nemoguardrails.llm.clients._sse import ServerSentEvent
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
        gen = self._apost_stream(self._ROUTE, payload)
        try:
            async for chunk in gen:
                yield chunk
        finally:
            await gen.aclose()

    def _is_stream_done(self, sse: ServerSentEvent) -> bool:
        return sse.event == "message_stop"

    def _should_skip_sse(self, sse: ServerSentEvent) -> bool:
        return sse.event == "ping"

    def _transform_sse_chunk(self, parsed: Dict[str, Any], sse: ServerSentEvent) -> Dict[str, Any]:
        parsed["_event_type"] = sse.event
        return parsed

    def _check_sse_error(self, parsed: Any, headers: Any, ctx: Optional[ErrorContext] = None) -> None:
        if isinstance(parsed, dict) and parsed.get("type") == "error":
            error_msg = parsed.get("error", {})
            raise LLMResponseValidationError(
                f"Anthropic stream error: {error_msg}",
                response_data=parsed,
                **(ctx.as_kwargs() if ctx else {}),
            )

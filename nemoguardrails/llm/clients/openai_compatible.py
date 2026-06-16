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

from typing import Any, AsyncGenerator, Dict, List, Optional

from nemoguardrails.llm.clients.base import BaseClient, HTTPResponse


class OpenAICompatibleClient(BaseClient):
    _ROUTE = "/chat/completions"

    @property
    def provider_url(self) -> Optional[str]:
        return self._base_url

    def _build_payload(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        include_usage_in_stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if stop:
            payload["stop"] = stop
        if stream:
            payload["stream"] = True
            if include_usage_in_stream is not False:
                payload["stream_options"] = {"include_usage": True}
        payload.update(kwargs)
        return payload

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> HTTPResponse:
        payload = self._build_payload(model, messages, stop=stop, **kwargs)
        return await self._apost(self._ROUTE, payload)

    async def stream_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[HTTPResponse, None]:
        payload = self._build_payload(model, messages, stop=stop, stream=True, **kwargs)
        gen = self._apost_stream(self._ROUTE, payload)
        try:
            async for chunk in gen:
                yield chunk
        finally:
            await gen.aclose()

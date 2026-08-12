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
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from nemoguardrails.exceptions import LLMClientError, LLMResponseValidationError
from nemoguardrails.llm.anthropic_utils import nemo_to_anthropic_messages
from nemoguardrails.llm.clients.anthropic import AnthropicClient
from nemoguardrails.llm.clients.base import HTTPResponse
from nemoguardrails.types import (
    ChatMessage,
    FinishReason,
    LLMResponse,
    LLMResponseChunk,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
)

_FINISH_REASON_MAP: Dict[str, FinishReason] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
}


class AnthropicChatModel:
    def __init__(
        self,
        client: AnthropicClient,
        model: str,
        *,
        provider_name: Optional[str] = None,
        **kwargs: Any,
    ):
        self._client = client
        self._model = model
        self._provider_name = provider_name or "anthropic"
        self._default_kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def provider_url(self) -> Optional[str]:
        return self._client.provider_url

    def _enrich(self, exc: LLMClientError) -> LLMClientError:
        exc.provider_name = self._provider_name
        exc.model_name = self._model
        exc.base_url = self._client.provider_url
        return exc

    def _prepare_params(self, stop: Optional[List[str]], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        merged = {**self._default_kwargs, **kwargs}
        if stop is not None:
            merged["stop_sequences"] = stop
        return merged

    def _to_messages(self, prompt: Union[str, List[ChatMessage]]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        if isinstance(prompt, str):
            return None, [{"role": "user", "content": prompt}]
        return nemo_to_anthropic_messages(prompt)

    async def generate_async(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        system, messages = self._to_messages(prompt)
        params = self._prepare_params(stop, kwargs)
        try:
            response = await self._client.create_message(self._model, messages, system=system, **params)
        except LLMClientError as exc:
            raise self._enrich(exc)
        return self._parse_response(response)

    async def stream_async(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponseChunk]:
        system, messages = self._to_messages(prompt)
        params = self._prepare_params(stop, kwargs)

        tool_call_acc: Dict[int, Dict[str, Any]] = {}
        current_block_idx = -1
        current_block_type: Optional[str] = None

        gen = self._client.stream_message(self._model, messages, system=system, **params)
        try:
            async for chunk_response in gen:
                body = chunk_response.body
                event_type = body.pop("_event_type", "")

                if event_type == "message_start":
                    continue

                if event_type == "content_block_start":
                    current_block_idx = body.get("index", current_block_idx + 1)
                    block = body.get("content_block", {})
                    current_block_type = block.get("type")
                    if current_block_type == "tool_use":
                        tool_call_acc[current_block_idx] = {
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "arguments_buffer": "",
                        }
                    continue

                if event_type == "content_block_stop":
                    current_block_type = None
                    continue

                if event_type == "content_block_delta":
                    delta = body.get("delta", {})
                    delta_type = delta.get("type")

                    if delta_type == "text_delta":
                        yield LLMResponseChunk(delta_content=delta.get("text"))
                        continue

                    if delta_type == "thinking_delta":
                        yield LLMResponseChunk(delta_reasoning=delta.get("thinking"))
                        continue

                    if delta_type == "input_json_delta":
                        partial_json = delta.get("partial_json", "")
                        if current_block_idx in tool_call_acc and partial_json:
                            tool_call_acc[current_block_idx]["arguments_buffer"] += partial_json
                        continue

                if event_type == "message_delta":
                    delta = body.get("delta", {})
                    raw_finish = delta.get("stop_reason")
                    finish_reason = _FINISH_REASON_MAP.get(raw_finish, "other") if raw_finish else None

                    usage = None
                    raw_usage = body.get("usage")
                    if raw_usage:
                        usage = UsageInfo(
                            input_tokens=raw_usage.get("input_tokens", 0),
                            output_tokens=raw_usage.get("output_tokens", 0),
                            total_tokens=(raw_usage.get("input_tokens", 0) + raw_usage.get("output_tokens", 0)),
                        )

                    chunk = LLMResponseChunk(finish_reason=finish_reason, usage=usage)
                    if finish_reason == "tool_calls" and tool_call_acc:
                        chunk.delta_tool_calls = self._finalize_tool_calls(tool_call_acc)
                    yield chunk
                    continue

        except LLMClientError as exc:
            raise self._enrich(exc)
        finally:
            await gen.aclose()

    @staticmethod
    def _finalize_tool_calls(acc: Dict[int, Dict[str, Any]]) -> List[ToolCall]:
        result = []
        for idx in sorted(acc.keys()):
            entry = acc[idx]
            raw_args = entry["arguments_buffer"]
            try:
                args_dict = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args_dict = {}
            result.append(
                ToolCall(
                    id=entry["id"],
                    type="function",
                    function=ToolCallFunction(
                        name=entry["name"],
                        arguments=args_dict,
                    ),
                )
            )
        return result

    def _parse_response(self, response: HTTPResponse) -> LLMResponse:
        data = response.body
        self._validate_response(data)

        content_blocks = data.get("content", [])
        text_parts = []
        reasoning_parts = []
        thinking_blocks = []
        tool_calls = []

        for block in content_blocks:
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "thinking":
                reasoning_parts.append(block.get("thinking", ""))
                thinking_blocks.append(block)
            elif block_type == "redacted_thinking":
                thinking_blocks.append(block)
            elif block_type == "tool_use":
                args = block.get("input", {})
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        type="function",
                        function=ToolCallFunction(
                            name=block.get("name", ""),
                            arguments=args if isinstance(args, dict) else {},
                        ),
                    )
                )

        raw_finish = data.get("stop_reason")
        finish_reason = _FINISH_REASON_MAP.get(raw_finish, "other") if raw_finish else None

        usage = None
        raw_usage = data.get("usage")
        if raw_usage:
            input_tokens = raw_usage.get("input_tokens", 0)
            output_tokens = raw_usage.get("output_tokens", 0)
            usage = UsageInfo(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cached_tokens=raw_usage.get("cache_read_input_tokens"),
            )

        provider_metadata: Dict[str, Any] = {}
        if response.headers:
            provider_metadata["response_headers"] = dict(response.headers)
        if thinking_blocks:
            provider_metadata["thinking_blocks"] = thinking_blocks

        return LLMResponse(
            content="\n".join(text_parts) if text_parts else "",
            reasoning="\n".join(reasoning_parts) if reasoning_parts else None,
            tool_calls=tool_calls if tool_calls else None,
            model=data.get("model"),
            finish_reason=finish_reason,
            request_id=data.get("id"),
            usage=usage,
            provider_metadata=provider_metadata or None,
        )

    def _validate_response(self, data: Any) -> None:
        ctx = dict(
            model_name=self.model_name,
            provider_name=self.provider_name,
            base_url=self.provider_url,
        )
        if not isinstance(data, dict):
            raise LLMResponseValidationError(
                f"Expected dict response, got {type(data).__name__}", response_data=None, **ctx
            )
        if data.get("type") == "error":
            error = data.get("error", {})
            raise LLMResponseValidationError(
                f"Anthropic error: {error.get('message', str(error))}", response_data=data, **ctx
            )
        content = data.get("content")
        if not isinstance(content, list):
            raise LLMResponseValidationError(
                f"Missing or invalid 'content' in response: {list(data.keys())}", response_data=data, **ctx
            )

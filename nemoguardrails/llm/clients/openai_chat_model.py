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
from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient
from nemoguardrails.types import (
    ChatMessage,
    FinishReason,
    LLMResponse,
    LLMResponseChunk,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
)

_KNOWN_PROVIDER_URLS: Dict[str, str] = {
    "https://api.openai.com/v1": "openai",
    "https://integrate.api.nvidia.com/v1": "nim",
}

_FINISH_REASON_MAP: Dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
}

_STANDARD_RESPONSE_KEYS = frozenset({"model", "choices", "usage", "id", "object", "created", "_response_headers"})


def _is_openai_reasoning_model(model_name: str) -> bool:
    name = model_name.lower()
    if name in ("o1", "o3", "o4") or name.startswith(("o1-", "o3-", "o4-")):
        return True
    if name == "gpt-5" or name.startswith("gpt-5-"):
        return "chat" not in name
    if name.startswith(("gpt-5.", "gpt-6")):
        return True
    return False


class OpenAIChatModel:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        model: str,
        *,
        provider_name: Optional[str] = None,
        **kwargs: Any,
    ):
        self._client = client
        self._model = model
        if provider_name is None:
            provider_name = _KNOWN_PROVIDER_URLS.get(client.provider_url or "", "openai")
        self._provider_name = provider_name
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
            merged["stop"] = stop
        if _is_openai_reasoning_model(self._model):
            merged.pop("temperature", None)
            merged.pop("stop", None)
        return merged

    def _to_messages(self, prompt: Union[str, List[ChatMessage]]) -> List[Dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        result = []
        for msg in prompt:
            d = msg.to_dict()
            d.pop("provider_metadata", None)
            if "tool_calls" in d:
                for tc in d["tool_calls"]:
                    func = tc.get("function", {})
                    if isinstance(func.get("arguments"), dict):
                        func["arguments"] = json.dumps(func["arguments"])
            result.append(d)
        return result

    async def generate_async(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = self._to_messages(prompt)
        params = self._prepare_params(stop, kwargs)
        try:
            data = await self._client.chat_completion(self._model, messages, **params)
        except LLMClientError as exc:
            raise self._enrich(exc)
        return self._parse_response(data)

    async def stream_async(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponseChunk]:
        messages = self._to_messages(prompt)
        params = self._prepare_params(stop, kwargs)

        tool_call_acc: Dict[int, Dict[str, Any]] = {}

        gen = self._client.stream_chat_completion(self._model, messages, **params)
        try:
            async for chunk_data in gen:
                choices = chunk_data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    raw_tool_calls = delta.get("tool_calls")
                    if raw_tool_calls:
                        for tc_delta in raw_tool_calls:
                            idx = tc_delta.get("index", 0)
                            if idx not in tool_call_acc:
                                tool_call_acc[idx] = {
                                    "id": tc_delta.get("id", ""),
                                    "type": tc_delta.get("type", "function"),
                                    "function_name": tc_delta.get("function", {}).get("name", ""),
                                    "arguments_buffer": "",
                                }
                            arg_fragment = tc_delta.get("function", {}).get("arguments", "")
                            if arg_fragment:
                                tool_call_acc[idx]["arguments_buffer"] += arg_fragment

                chunk = self._parse_chunk(chunk_data)
                if chunk is None:
                    continue

                if chunk.finish_reason == "tool_calls" and tool_call_acc:
                    chunk.delta_tool_calls = self._finalize_tool_calls(tool_call_acc)

                yield chunk
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
            # Graceful degrade: truncated (max_tokens) or malformed args => empty dict.
            # The tool will surface the real error when invoked with no arguments.
            try:
                args_dict = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args_dict = {}
            result.append(
                ToolCall(
                    id=entry["id"],
                    type=entry["type"],
                    function=ToolCallFunction(
                        name=entry["function_name"],
                        arguments=args_dict,
                    ),
                )
            )
        return result

    def _validate_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        ctx = dict(
            model_name=self.model_name,
            provider_name=self.provider_name,
            base_url=self.provider_url,
        )
        if not isinstance(data, dict):
            raise LLMResponseValidationError(
                f"Expected dict response, got {type(data).__name__}", response_data=None, **ctx
            )
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseValidationError(
                f"Missing or empty 'choices' in response: {list(data.keys())}", response_data=data, **ctx
            )
        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMResponseValidationError(
                f"Expected dict in choices[0], got {type(choice).__name__}", response_data=data, **ctx
            )
        message = choice.get("message")
        if not isinstance(message, dict):
            raise LLMResponseValidationError("Missing or invalid 'message' in choices[0]", response_data=data, **ctx)
        return message

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        message = self._validate_response(data)
        choice = data["choices"][0]

        content = message.get("content") or ""
        reasoning = message.get("reasoning_content")

        tool_calls = None
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls:
            tool_calls = [self._parse_tool_call(tc) for tc in raw_tool_calls]

        raw_finish = choice.get("finish_reason")
        finish_reason = _FINISH_REASON_MAP.get(raw_finish, "other") if raw_finish else None

        usage = None
        raw_usage = data.get("usage")
        if raw_usage:
            input_tokens = raw_usage.get("prompt_tokens", 0)
            output_tokens = raw_usage.get("completion_tokens", 0)
            usage = UsageInfo(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=raw_usage.get("total_tokens") or (input_tokens + output_tokens),
                reasoning_tokens=(raw_usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                cached_tokens=(raw_usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            )

        provider_metadata = {k: v for k, v in data.items() if k not in _STANDARD_RESPONSE_KEYS and v is not None}

        response_headers = data.get("_response_headers")
        if response_headers:
            provider_metadata["response_headers"] = response_headers

        return LLMResponse(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            model=data.get("model"),
            finish_reason=finish_reason,
            request_id=data.get("id"),
            usage=usage,
            provider_metadata=provider_metadata or None,
        )

    def _parse_chunk(self, data: Dict[str, Any]) -> Optional[LLMResponseChunk]:
        provider_metadata = {k: v for k, v in data.items() if k not in _STANDARD_RESPONSE_KEYS and v is not None}
        response_headers = data.get("_response_headers")
        if response_headers:
            provider_metadata["response_headers"] = response_headers

        choices = data.get("choices", [])
        if not choices:
            raw_usage = data.get("usage")
            if raw_usage:
                input_tokens = raw_usage.get("prompt_tokens", 0)
                output_tokens = raw_usage.get("completion_tokens", 0)
                return LLMResponseChunk(
                    request_id=data.get("id"),
                    usage=UsageInfo(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=raw_usage.get("total_tokens") or (input_tokens + output_tokens),
                        reasoning_tokens=(raw_usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                        cached_tokens=(raw_usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    ),
                    provider_metadata=provider_metadata or None,
                )
            return None

        choice = choices[0]
        delta = choice.get("delta", {})

        content = delta.get("content")
        reasoning = delta.get("reasoning_content")
        raw_finish = choice.get("finish_reason")
        finish_reason = _FINISH_REASON_MAP.get(raw_finish, "other") if raw_finish else None

        return LLMResponseChunk(
            delta_content=content,
            delta_reasoning=reasoning,
            model=data.get("model"),
            finish_reason=finish_reason,
            request_id=data.get("id"),
            provider_metadata=provider_metadata or None,
        )

    @staticmethod
    def _parse_tool_call(tc: Dict[str, Any]) -> ToolCall:
        func = tc.get("function", {})
        raw_args = func.get("arguments", "{}")
        if isinstance(raw_args, str):
            # Graceful degrade: malformed args from provider => empty dict.
            # The tool will surface the real error when invoked with no arguments.
            try:
                args_dict = json.loads(raw_args)
            except json.JSONDecodeError:
                args_dict = {}
        else:
            args_dict = raw_args

        return ToolCall(
            id=tc.get("id", ""),
            type=tc.get("type", "function"),
            function=ToolCallFunction(
                name=func.get("name", ""),
                arguments=args_dict,
            ),
        )

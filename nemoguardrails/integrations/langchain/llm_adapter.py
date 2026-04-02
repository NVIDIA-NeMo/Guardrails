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

import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from nemoguardrails.types import (
    ChatMessage,
    FinishReason,
    LLMModel,
    LLMResponse,
    LLMResponseChunk,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
)


def _infer_model_name(llm: Any):
    """Helper to infer the model name based from an LLM instance.

    Because not all models implement correctly _identifying_params from LangChain, we have to
    try to do this manually.
    """
    for attr in ["model", "model_name"]:
        if hasattr(llm, attr):
            val = getattr(llm, attr)
            if isinstance(val, str):
                return val

    model_kwargs = getattr(llm, "model_kwargs", None)
    if model_kwargs and isinstance(model_kwargs, Dict):
        for attr in ["model", "model_name", "name"]:
            val = model_kwargs.get(attr)
            if isinstance(val, str):
                return val

    # If we still can't figure out, return "unknown".
    return "unknown"


def _infer_provider_from_module(llm: Any) -> Optional[str]:
    """Infer provider name from the LLM's module path.

    This function extracts the provider name from LangChain package naming conventions:
    - langchain_openai -> openai
    - langchain_anthropic -> anthropic
    - langchain_google_genai -> google_genai
    - langchain_nvidia_ai_endpoints -> nvidia_ai_endpoints
    - langchain_community.chat_models.ollama -> ollama

    For patched/wrapped classes, checks base classes as well.

    Args:
        llm: The LLM instance

    Returns:
        The inferred provider name, or None if it cannot be determined
    """
    module = type(llm).__module__

    if module.startswith("langchain_"):
        package = module.split(".")[0]
        provider = package.replace("langchain_", "")

        if provider == "community":
            parts = module.split(".")
            if len(parts) >= 3:
                provider = parts[-1]
                return provider
        else:
            return provider

    for base_class in type(llm).__mro__[1:]:
        base_module = base_class.__module__
        if base_module.startswith("langchain_"):
            package = base_module.split(".")[0]
            provider = package.replace("langchain_", "")

            if provider == "community":
                parts = base_module.split(".")
                if len(parts) >= 3:
                    provider = parts[-1]
                    return provider
            else:
                return provider

    return None


_BASE_URL_ATTRIBUTES = [
    "base_url",
    "endpoint_url",
    "server_url",
    "azure_endpoint",
    "openai_api_base",
    "api_base",
    "api_host",
    "endpoint",
]


class LangChainLLMAdapter:
    def __init__(self, llm):
        self._llm = llm

    @property
    def raw_llm(self) -> Any:
        return self._llm

    @property
    def model_name(self) -> str:
        return _infer_model_name(self._llm)

    @property
    def provider_name(self) -> Optional[str]:
        return _infer_provider_from_module(self._llm)

    @property
    def provider_url(self) -> Optional[str]:
        # temp: uses _BASE_URL_ATTRIBUTES which duplicates utils.py BASE_URL_ATTRIBUTES.
        # utils.py copy will be removed in stack-3 when it switches to model.provider_url.
        for attr in _BASE_URL_ATTRIBUTES:
            value = getattr(self._llm, attr, None)
            if value:
                return str(value)
        client = getattr(self._llm, "client", None)
        if client and hasattr(client, "base_url"):
            return str(client.base_url)
        return None

    def _filter_reasoning_model_params(self, params: Optional[dict]) -> Optional[dict]:
        if not params or "temperature" not in params:
            return params

        model_name = _infer_model_name(self._llm).lower()

        is_openai_reasoning_model = (
            model_name.startswith("o1")
            or model_name.startswith("o3")
            or (model_name.startswith("gpt-5") and "chat" not in model_name)
        )

        if is_openai_reasoning_model:
            filtered = params.copy()
            filtered.pop("temperature", None)
            return filtered

        return params

    def _prepare_llm(self, kwargs: dict):
        kwargs = self._filter_reasoning_model_params(kwargs) or {}
        llm = self._llm
        if kwargs:
            llm = llm.bind(**kwargs)
        return llm

    def _to_langchain_input(self, prompt):
        if isinstance(prompt, list):
            from nemoguardrails.integrations.langchain.message_utils import (
                chatmessages_to_langchain_messages,
            )

            return chatmessages_to_langchain_messages(prompt)
        return prompt

    async def generate(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        llm = self._prepare_llm(kwargs)
        messages = self._to_langchain_input(prompt)
        response = await llm.ainvoke(messages, stop=stop)
        return _langchain_response_to_llm_response(response)

    async def stream(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncIterator[LLMResponseChunk]:
        llm = self._prepare_llm(kwargs)
        messages = self._to_langchain_input(prompt)
        async for chunk in llm.astream(messages, stop=stop):
            yield _langchain_chunk_to_llm_response_chunk(chunk)


class LangChainFramework:
    def create_model(
        self,
        model_name: str,
        provider_name: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> LLMModel:
        from nemoguardrails.llm.models.langchain_initializer import (
            init_langchain_model,
        )

        kwargs = dict(model_kwargs) if model_kwargs else {}
        mode = kwargs.pop("mode", "chat")

        raw_llm = init_langchain_model(
            model_name=model_name,
            provider_name=provider_name,
            mode=mode,
            kwargs=kwargs,
        )
        return LangChainLLMAdapter(raw_llm)


_FINISH_REASON_MAP: Dict[str, FinishReason] = {
    "stop": "stop",
    "end_turn": "stop",
    "length": "length",
    "max_tokens": "length",
    "tool_calls": "tool_calls",
    "tool_use": "tool_calls",
    "content_filter": "content_filter",
}


def _map_finish_reason(raw: Optional[str]) -> Optional[FinishReason]:
    if raw is None:
        return None
    return _FINISH_REASON_MAP.get(raw, "other")


def _build_usage_info(raw: Any) -> Optional[UsageInfo]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        try:
            raw = dict(raw)
        except (TypeError, ValueError):
            return None
    if not raw:
        return None
    return UsageInfo(
        input_tokens=raw.get("input_tokens", raw.get("prompt_tokens", 0)),
        output_tokens=raw.get("output_tokens", raw.get("completion_tokens", 0)),
        total_tokens=raw.get("total_tokens", 0),
        reasoning_tokens=raw.get("reasoning_tokens"),
        cached_tokens=raw.get("cached_tokens", raw.get("cache_read_input_tokens")),
    )


def _extract_reasoning(response) -> Optional[str]:
    content_blocks = getattr(response, "content_blocks", None)
    if content_blocks:
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "reasoning":
                val = block.get("reasoning")
                if val:
                    return val

    additional_kwargs = getattr(response, "additional_kwargs", None)
    if additional_kwargs and isinstance(additional_kwargs, dict):
        val = additional_kwargs.get("reasoning_content")
        if val:
            return val

    return None


def _langchain_response_to_llm_response(response) -> LLMResponse:
    content = getattr(response, "content", None)
    if content is None:
        content = str(response)

    reasoning = _extract_reasoning(response)

    raw_tool_calls = getattr(response, "tool_calls", None)
    tool_calls = None
    if raw_tool_calls:
        tool_calls = []
        for tc in raw_tool_calls:
            if isinstance(tc, dict):
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id") or str(uuid.uuid4()),
                        type="function",
                        function=ToolCallFunction(
                            name=tc.get("name", ""),
                            arguments=tc.get("args", {}),
                        ),
                    )
                )
            else:
                tool_calls.append(
                    ToolCall(
                        id=getattr(tc, "id", None) or str(uuid.uuid4()),
                        type="function",
                        function=ToolCallFunction(
                            name=getattr(tc, "name", ""),
                            arguments=getattr(tc, "args", {}),
                        ),
                    )
                )

    response_metadata = getattr(response, "response_metadata", None) or {}
    additional_kwargs = getattr(response, "additional_kwargs", None) or {}

    usage_metadata = getattr(response, "usage_metadata", None)
    usage = _build_usage_info(usage_metadata)
    if usage is None and response_metadata:
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
        if token_usage:
            usage = _build_usage_info(token_usage)

    model = response_metadata.get("model_name") or response_metadata.get("model")

    raw_finish = response_metadata.get("finish_reason") or response_metadata.get("stop_reason")
    finish_reason = _map_finish_reason(raw_finish)

    stop_sequence = response_metadata.get("stop_sequence")

    request_id = response_metadata.get("id") or response_metadata.get("request_id")

    extracted_keys = {
        "model_name",
        "model",
        "finish_reason",
        "stop_reason",
        "stop_sequence",
        "id",
        "request_id",
        "token_usage",
        "usage",
    }
    reasoning_keys = {"reasoning_content"}
    provider_metadata: Dict[str, Any] = {}
    for k, v in response_metadata.items():
        if k not in extracted_keys:
            provider_metadata[k] = v
    for k, v in additional_kwargs.items():
        if k not in reasoning_keys and k not in provider_metadata:
            provider_metadata[k] = v

    return LLMResponse(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        model=model,
        finish_reason=finish_reason,
        stop_sequence=stop_sequence,
        request_id=request_id,
        usage=usage,
        provider_metadata=provider_metadata if provider_metadata else None,
    )


def _langchain_chunk_to_llm_response_chunk(chunk) -> LLMResponseChunk:
    content = getattr(chunk, "content", None)
    if content is None:
        content = getattr(chunk, "text", None)
    if content is None:
        content = str(chunk)

    response_metadata = getattr(chunk, "response_metadata", None) or {}
    generation_info = getattr(chunk, "generation_info", None) or {}

    usage_metadata = getattr(chunk, "usage_metadata", None)
    usage = _build_usage_info(usage_metadata)
    if usage is None and response_metadata:
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
        if token_usage:
            usage = _build_usage_info(token_usage)
    if usage is None and generation_info:
        token_usage = generation_info.get("token_usage") or generation_info.get("usage")
        if token_usage:
            usage = _build_usage_info(token_usage)

    provider_metadata: Dict[str, Any] = {}
    for k, v in response_metadata.items():
        provider_metadata[k] = v
    for k, v in generation_info.items():
        if k not in provider_metadata:
            provider_metadata[k] = v

    return LLMResponseChunk(
        delta_content=content,
        usage=usage,
        provider_metadata=provider_metadata if provider_metadata else None,
    )

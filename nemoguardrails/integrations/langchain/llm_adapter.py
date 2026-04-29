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
import uuid
from typing import Any, AsyncIterator, Dict, List, NamedTuple, Optional, Union

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

log = logging.getLogger(__name__)


def _flatten_content(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in raw)
    return str(raw)


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
    if model_kwargs and isinstance(model_kwargs, dict):
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
            return parts[-1] if len(parts) >= 3 else "community"
        else:
            return provider

    for base_class in type(llm).__mro__[1:]:
        base_module = base_class.__module__
        if base_module.startswith("langchain_"):
            package = base_module.split(".")[0]
            provider = package.replace("langchain_", "")

            if provider == "community":
                parts = base_module.split(".")
                return parts[-1] if len(parts) >= 3 else "community"
            else:
                return provider

    return None


def _is_openai_reasoning_model(model_name: str) -> bool:
    name = model_name.lower()
    if name in ("o1", "o3", "o4") or name.startswith(("o1-", "o3-", "o4-")):
        return True
    if name == "gpt-5" or name.startswith("gpt-5-"):
        return "chat" not in name
    if name.startswith(("gpt-5.", "gpt-6")):
        return True
    return False


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
        for attr in _BASE_URL_ATTRIBUTES:
            value = getattr(self._llm, attr, None)
            if value:
                return str(value)
        client = getattr(self._llm, "client", None)
        if client and hasattr(client, "base_url"):
            return str(client.base_url)
        return None

    def _prepare_call_params(self, stop: Optional[List[str]], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(kwargs)
        if stop is not None:
            params["stop"] = stop
        if _is_openai_reasoning_model(self.model_name):
            params.pop("temperature", None)
            params.pop("stop", None)
        return params

    def _to_langchain_input(self, prompt):
        if isinstance(prompt, list):
            from nemoguardrails.integrations.langchain.message_utils import (
                chatmessages_to_langchain_messages,
            )

            return chatmessages_to_langchain_messages(prompt)
        return prompt

    async def generate_async(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        params = self._prepare_call_params(stop, kwargs)
        llm = self._llm.bind(**params) if params else self._llm
        messages = self._to_langchain_input(prompt)
        response = await llm.ainvoke(messages)
        return _langchain_response_to_llm_response(response)

    async def stream_async(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncIterator[LLMResponseChunk]:
        params = self._prepare_call_params(stop, kwargs)
        llm = self._llm.bind(**params) if params else self._llm
        messages = self._to_langchain_input(prompt)

        tool_call_acc: Dict[int, Dict[str, Any]] = {}

        async for chunk in llm.astream(messages):
            for tc_chunk in getattr(chunk, "tool_call_chunks", None) or []:
                idx = tc_chunk.get("index", 0)
                if idx not in tool_call_acc:
                    tool_call_acc[idx] = {
                        "id": tc_chunk.get("id") or "",
                        "name": tc_chunk.get("name") or "",
                        "arguments_buffer": "",
                    }
                else:
                    if tc_chunk.get("id"):
                        tool_call_acc[idx]["id"] = tc_chunk["id"]
                    if tc_chunk.get("name"):
                        tool_call_acc[idx]["name"] = tc_chunk["name"]
                arg_fragment = tc_chunk.get("args") or ""
                if arg_fragment:
                    tool_call_acc[idx]["arguments_buffer"] += arg_fragment

            response_chunk = _langchain_chunk_to_llm_response_chunk(chunk)

            if response_chunk.finish_reason == "tool_calls" and tool_call_acc:
                response_chunk.delta_tool_calls = _finalize_tool_call_acc(tool_call_acc)

            yield response_chunk


class LangChainFramework:
    def register_provider(self, name: str, provider_cls: Any) -> None:
        from nemoguardrails.integrations.langchain.providers.providers import (
            register_chat_provider as _register_chat,
        )

        _register_chat(name, provider_cls)

    def register_llm_provider(self, name: str, provider_cls: Any) -> None:
        from nemoguardrails.integrations.langchain.providers.providers import (
            register_llm_provider as _register_llm,
        )

        _register_llm(name, provider_cls)

    def get_provider_names(self) -> List[str]:
        return sorted(set(self.get_chat_provider_names() + self.get_llm_provider_names()))

    def get_chat_provider_names(self) -> List[str]:
        from nemoguardrails.integrations.langchain.providers.providers import (
            get_chat_provider_names as _get_chat,
        )

        return _get_chat()

    def get_llm_provider_names(self) -> List[str]:
        from nemoguardrails.integrations.langchain.providers.providers import (
            get_llm_provider_names as _get_llm,
        )

        return _get_llm()

    async def reset(self) -> None:
        return

    def create_model(
        self,
        model_name: str,
        provider_name: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> LLMModel:
        from nemoguardrails.integrations.langchain.langchain_initializer import (
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
    input_tokens = raw.get("input_tokens", raw.get("prompt_tokens", 0))
    output_tokens = raw.get("output_tokens", raw.get("completion_tokens", 0))
    total_tokens = raw.get("total_tokens") or (input_tokens + output_tokens)
    return UsageInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=raw.get("reasoning_tokens"),
        cached_tokens=raw.get("cached_tokens", raw.get("cache_read_input_tokens")),
    )


_EXTRACTED_METADATA_KEYS = frozenset(
    {
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
)

_REASONING_KEYS = frozenset({"reasoning_content"})


def _extract_reasoning(response: Any) -> Optional[str]:
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


def _extract_tool_calls(response: Any) -> Optional[List[ToolCall]]:
    raw = getattr(response, "tool_calls", None)
    if not raw:
        return None
    return [
        ToolCall(
            id=tc.get("id") or str(uuid.uuid4()),
            type="function",
            function=ToolCallFunction(
                name=tc.get("name", ""),
                arguments=tc.get("args", {}),
            ),
        )
        for tc in raw
    ]


def _finalize_tool_call_acc(acc: Dict[int, Dict[str, Any]]) -> List[ToolCall]:
    result = []
    for idx in sorted(acc.keys()):
        entry = acc[idx]
        raw_args = entry["arguments_buffer"]
        try:
            args_dict = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            log.warning("Failed to parse tool call arguments for '%s' (index %d): %r", entry["name"], idx, raw_args)
            args_dict = {}
        result.append(
            ToolCall(
                id=entry["id"] or str(uuid.uuid4()),
                type="function",
                function=ToolCallFunction(
                    name=entry["name"],
                    arguments=args_dict,
                ),
            )
        )
    return result


def _extract_usage(response: Any) -> Optional[UsageInfo]:
    usage = _build_usage_info(getattr(response, "usage_metadata", None))
    if usage is not None:
        return usage

    for source in (
        getattr(response, "response_metadata", None) or {},
        getattr(response, "generation_info", None) or {},
    ):
        token_usage = source.get("token_usage") or source.get("usage")
        if token_usage:
            usage = _build_usage_info(token_usage)
            if usage is not None:
                return usage

    return None


class _ModelInfo(NamedTuple):
    model: Optional[str]
    finish_reason: Optional[FinishReason]
    stop_sequence: Optional[str]
    request_id: Optional[str]


def _extract_model_info(response_metadata: Dict[str, Any]) -> _ModelInfo:
    model = response_metadata.get("model_name") or response_metadata.get("model")
    raw_finish = response_metadata.get("finish_reason") or response_metadata.get("stop_reason")
    finish_reason = _map_finish_reason(raw_finish)
    stop_sequence = response_metadata.get("stop_sequence")
    request_id = response_metadata.get("id") or response_metadata.get("request_id")
    return _ModelInfo(model, finish_reason, stop_sequence, request_id)


def _build_provider_metadata(
    response_metadata: Dict[str, Any],
    additional_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    result: Dict[str, Any] = {k: v for k, v in response_metadata.items() if k not in _EXTRACTED_METADATA_KEYS}
    if additional_kwargs:
        for k, v in additional_kwargs.items():
            if k not in _REASONING_KEYS and k not in result:
                result[k] = v
    return result or None


def _langchain_response_to_llm_response(response: Any) -> LLMResponse:
    content = _flatten_content(getattr(response, "content", None))
    if content is None:
        content = str(response)

    response_metadata = getattr(response, "response_metadata", None) or {}
    additional_kwargs = getattr(response, "additional_kwargs", None) or {}
    model, finish_reason, stop_sequence, request_id = _extract_model_info(response_metadata)

    return LLMResponse(
        content=content,
        reasoning=_extract_reasoning(response),
        tool_calls=_extract_tool_calls(response),
        model=model,
        finish_reason=finish_reason,
        stop_sequence=stop_sequence,
        request_id=request_id,
        usage=_extract_usage(response),
        provider_metadata=_build_provider_metadata(response_metadata, additional_kwargs),
    )


def _langchain_chunk_to_llm_response_chunk(chunk: Any) -> LLMResponseChunk:
    content = _flatten_content(getattr(chunk, "content", None))
    if content is None:
        content = _flatten_content(getattr(chunk, "text", None))
    if content is None:
        content = str(chunk)

    response_metadata = getattr(chunk, "response_metadata", None) or {}
    generation_info = getattr(chunk, "generation_info", None) or {}
    merged_metadata = {**response_metadata, **generation_info}

    model, finish_reason, stop_sequence, request_id = _extract_model_info(merged_metadata)

    return LLMResponseChunk(
        delta_content=content,
        delta_reasoning=_extract_reasoning(chunk),
        finish_reason=finish_reason,
        model=model,
        request_id=request_id,
        usage=_extract_usage(chunk),
        provider_metadata=_build_provider_metadata(merged_metadata) or None,
    )

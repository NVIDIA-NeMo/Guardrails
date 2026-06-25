# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Utility functions for converting between Guardrails and OpenAI API formats."""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.server.schemas.openai import (
    GuardrailsChatCompletion,
    GuardrailsDataOutput,
    OpenAIModel,
)

log = logging.getLogger(__name__)


def _azure_url() -> str:
    """Build the Azure OpenAI models URL from env vars."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
    version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
    return f"{endpoint}/openai/models?api-version={version}"


def _parse_timestamp(value: Any, default: int) -> int:
    """Return an integer epoch from *value* (int, float, ISO-8601 str, or None)."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except (ValueError, TypeError):
            pass
    return default


def _openai_compatible_url() -> str:
    """Build a ``/v1/models`` URL from ``MAIN_MODEL_BASE_URL``."""
    base = os.environ.get("MAIN_MODEL_BASE_URL", "").rstrip("/")
    if not base:
        raise ValueError("MAIN_MODEL_BASE_URL is not set")
    return f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"


PROVIDERS: Dict[str, dict] = {
    # OpenAI compatible
    "openai": {},
    "vllm": {},
    "nim": {},
    "trt_llm": {},
    "anthropic": {
        "url": lambda: os.environ.get("MAIN_MODEL_BASE_URL", "https://api.anthropic.com").rstrip("/") + "/v1/models",
        "api_key_env": "ANTHROPIC_API_KEY",
        "auth_header": "x-api-key",
        "bearer": False,
        "extra_headers": {"anthropic-version": "2023-06-01"},
        "created_field": "created_at",
        "owned_by": "anthropic",
    },
    "azure": {
        "url": _azure_url,
        "api_key_env": "AZURE_OPENAI_API_KEY",
        "auth_header": "api-key",
        "bearer": False,
        "created_field": "created_at",
        "owned_by": "azure",
    },
    "cohere": {
        "url": lambda: os.environ.get("COHERE_BASE_URL", "https://api.cohere.com").rstrip("/") + "/v2/models",
        "api_key_env": "COHERE_API_KEY",
        "extra_headers": {"Accept": "application/json"},
        "models_key": "models",
        "id_field": "name",
        "owned_by": "cohere",
    },
}
PROVIDERS["azure_openai"] = PROVIDERS["azure"]


async def fetch_models(
    engine: str,
    request_headers: Dict[str, str],
) -> List[OpenAIModel]:
    """Fetch the model list for the specified engine and return OpenAIModel objects."""
    # Look up the provider in the PROVIDERS table
    provider = PROVIDERS.get(engine)

    if provider is None:
        if os.environ.get("MAIN_MODEL_BASE_URL"):
            log.info(
                "Engine '%s' not in provider table; trying OpenAI-compatible endpoint via MAIN_MODEL_BASE_URL",
                engine,
            )
            provider = {}
        else:
            log.warning(
                "Engine '%s' is not supported and MAIN_MODEL_BASE_URL is not set. Returning empty model list.",
                engine,
            )
            return []

    url_or_fn = provider.get("url")
    if url_or_fn is not None:
        processed_url = str(url_or_fn() if callable(url_or_fn) else url_or_fn)
    else:
        processed_url = _openai_compatible_url()

    # Build auth headers
    auth_header_name = provider.get("auth_header", "Authorization")
    use_bearer = provider.get("bearer", True)
    api_key_env = provider.get("api_key_env", "OPENAI_API_KEY")

    headers: Dict[str, str] = {}
    forwarded = request_headers.get("Authorization", "")

    raw_key = os.environ.get(api_key_env, "")
    if not raw_key:
        raw_key = forwarded.removeprefix("Bearer ").strip() if forwarded else ""
    if raw_key:
        headers[auth_header_name] = f"Bearer {raw_key}" if use_bearer else raw_key

    headers.update(provider.get("extra_headers", {}))

    async with httpx.AsyncClient() as client:
        resp = await client.get(processed_url, headers=headers, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()

    models_key = provider.get("models_key", "data")
    model_id = provider.get("id_field", "id")
    created_field = provider.get("created_field", "created")
    static_owned_by = provider.get("owned_by")
    default_owned_by = static_owned_by or os.environ.get("MAIN_MODEL_ENGINE", "system")
    now = int(time.time())

    models = data.get(models_key, []) if isinstance(data, dict) else []

    return [
        OpenAIModel(
            id=m.get(model_id, "unknown"),
            object="model",
            created=_parse_timestamp(m.get(created_field), now),
            owned_by=m.get("owned_by") or default_owned_by,
        )
        for m in models
        if isinstance(m, dict)
    ]


def _parse_tool_call_name_and_arguments(tc: dict) -> tuple[str, str]:
    if "function" in tc:
        func = tc.get("function") or {}
        name = func.get("name", "")
        arguments = func.get("arguments", {})
    else:
        name = tc.get("name", "")
        arguments = tc.get("args", {})

    if isinstance(arguments, dict):
        arguments_str = json.dumps(arguments)
    elif isinstance(arguments, str):
        arguments_str = arguments
    else:
        arguments_str = json.dumps(arguments)
    return name, arguments_str


def _generate_fallback_tool_call_id(tc: dict) -> str:
    fallback_id = f"call_{uuid.uuid4().hex[:8]}"
    func_name = tc.get("name") or (tc.get("function") or {}).get("name", "<unknown>")
    log.warning(
        "Tool call for function %r is missing an 'id'; generated fallback id %r.",
        func_name,
        fallback_id,
    )
    return fallback_id


def normalize_tool_calls_openai(
    tool_calls: List[dict],
) -> List[ChatCompletionMessageToolCall]:
    """Convert internal tool call dicts to OpenAI function tool call objects."""
    openai_tool_calls: List[ChatCompletionMessageToolCall] = []
    for tc in tool_calls:
        name, arguments_str = _parse_tool_call_name_and_arguments(tc)
        openai_tool_calls.append(
            ChatCompletionMessageToolCall(
                id=tc.get("id") or _generate_fallback_tool_call_id(tc),
                type="function",
                function=Function(name=name, arguments=arguments_str),
            )
        )
    return openai_tool_calls


def resolve_tool_calls(bot_message: dict, response_tool_calls: Optional[list] = None) -> Optional[List[dict]]:
    """Collect tool calls from a bot message and/or GenerationResponse.tool_calls."""
    tool_calls = bot_message.get("tool_calls") or response_tool_calls
    return tool_calls or None


def build_chat_completion_message(
    bot_message: dict,
    tool_calls: Optional[List[dict]] = None,
) -> ChatCompletionMessage:
    """Build an OpenAI ChatCompletionMessage from an internal bot message dict."""
    content = bot_message.get("content")
    if content == "" and tool_calls:
        content = None

    if not tool_calls:
        return ChatCompletionMessage(
            role="assistant",
            content=content,
        )

    openai_tool_calls = normalize_tool_calls_openai(tool_calls)

    return ChatCompletionMessage(
        role="assistant",
        content=content,
        tool_calls=openai_tool_calls,  # pyright: ignore[reportArgumentType]
    )


def warn_if_thread_history_invalid_for_tool_use(messages: List[dict]) -> None:
    """Log when persisted thread history is incompatible with OpenAI tool-calling message ordering."""
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        if i == 0 or messages[i - 1].get("role") != "assistant":
            log.warning(
                "Thread message history has a tool message without a preceding assistant "
                "message at index %s. Multi-turn tool use with thread_id is unreliable; "
                "send the full message list (assistant tool_calls + tool results) in each request.",
                i,
            )
            continue
        if not messages[i - 1].get("tool_calls"):
            log.warning(
                "Thread message history has a tool message after an assistant message "
                "without tool_calls at index %s. Prior assistant tool_calls may have been "
                "dropped when the thread was saved; include full history in the request.",
                i,
            )


def extract_bot_message_from_response(
    response: Union[str, dict, GenerationResponse, Tuple[dict, dict]],
) -> Dict[str, Any]:
    """
    Extract the bot message from generate_async response.

    Args:
        response: Response from LLMRails.generate_async() which can be:
            - str: Direct text response
            - dict: Message dict
            - GenerationResponse: Full response object
            - Tuple[dict, dict]: (message, state) tuple

    Returns:
        A dictionary with at least 'role' and 'content' keys
    """
    if isinstance(response, GenerationResponse):
        bot_message_content = response.response[0]
        # Ensure bot_message is always a dict
        if isinstance(bot_message_content, str):
            bot_message = {"role": "assistant", "content": bot_message_content}
        else:
            bot_message = bot_message_content
        return bot_message
    elif isinstance(response, str):
        # Direct string response
        return {"role": "assistant", "content": response}
    elif isinstance(response, tuple):
        # Tuple of (message, state)
        bot_message = response[0]
        if isinstance(bot_message, dict):
            return bot_message
        else:
            return {"role": "assistant", "content": str(bot_message)}
    else:
        # Already a dict
        return response


def generation_response_to_chat_completion(
    response: GenerationResponse,
    model: str,
    config_id: Optional[str] = None,
) -> GuardrailsChatCompletion:
    """
    Convert a GenerationResponse to an OpenAI-compatible GuardrailsChatCompletion.

    Args:
        response: The GenerationResponse from LLMRails.generate_async()
        model: The model name to include in the response
        config_id: Optional guardrails configuration ID

    Returns:
        A GuardrailsChatCompletion instance compatible with OpenAI API format
    """
    bot_message = extract_bot_message_from_response(response)
    tool_calls = resolve_tool_calls(bot_message, response.tool_calls)
    finish_reason = "tool_calls" if tool_calls else "stop"

    # Convert log to dict if present (for JSON serialization)
    log_dict = None
    if response.log:
        if hasattr(response.log, "model_dump"):
            log_dict = response.log.model_dump()
        elif hasattr(response.log, "dict"):
            log_dict = response.log.dict()
        elif isinstance(response.log, dict):
            log_dict = response.log
        else:
            # Fallback: try to convert to dict
            try:
                log_dict = dict(response.log)
            except (TypeError, ValueError):
                # If conversion fails, skip the log
                log_dict = None

    return GuardrailsChatCompletion(
        id=f"chatcmpl-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=build_chat_completion_message(bot_message, tool_calls),
                finish_reason=finish_reason,
                logprobs=None,
            )
        ],
        guardrails=GuardrailsDataOutput(
            config_id=config_id,
            llm_output=response.llm_output,
            output_data=response.output_data,
            log=log_dict,
            state=response.state,
        ),
    )


def bot_message_to_chat_completion(
    bot_message: dict,
    model: str,
    config_id: Optional[str] = None,
) -> GuardrailsChatCompletion:
    """Convert a bot message dict to an OpenAI-compatible GuardrailsChatCompletion."""
    tool_calls = resolve_tool_calls(bot_message)
    finish_reason = "tool_calls" if tool_calls else "stop"

    return GuardrailsChatCompletion(
        id=f"chatcmpl-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=build_chat_completion_message(bot_message, tool_calls),
                finish_reason=finish_reason,
                logprobs=None,
            )
        ],
        guardrails=GuardrailsDataOutput(config_id=config_id) if config_id else None,
    )


def create_error_chat_completion(
    model: str,
    error_message: str,
    config_id: Optional[str] = None,
) -> GuardrailsChatCompletion:
    """
    Create an error response in GuardrailsChatCompletion format.

    Args:
        model: The model name to include in the response
        error_message: The error message to return
        config_id: Optional guardrails configuration ID

    Returns:
        A GuardrailsChatCompletion instance with the error message
    """
    return GuardrailsChatCompletion(
        id=f"chatcmpl-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=error_message,
                ),
                finish_reason="stop",
                logprobs=None,
            )
        ],
        guardrails=GuardrailsDataOutput(config_id=config_id) if config_id else None,
    )


def format_streaming_chunk(
    chunk: Any,
    model: str,
    chunk_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Format a streaming chunk into OpenAI chat completion chunk format.

    Args:
        chunk: The chunk from LLMRails.stream_async() (can be dict, str, or other type)
        model: The model name to include in the chunk
        chunk_id: Optional ID for the chunk (generates UUID if not provided)

    Returns:
        A dictionary in OpenAI streaming chunk format
    """
    if chunk_id is None:
        chunk_id = f"chatcmpl-{uuid.uuid4()}"

    # Determine the payload format based on chunk type
    if isinstance(chunk, dict):
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "delta": chunk,
                    "index": 0,
                    "finish_reason": None,
                }
            ],
        }
    elif isinstance(chunk, str):
        try:
            # Try parsing as JSON - if it parses, it might be a pre-formed payload
            payload = json.loads(chunk)
            # Ensure it has the required fields
            if isinstance(payload, dict):
                if "id" not in payload:
                    payload["id"] = chunk_id
                if "model" not in payload:
                    payload["model"] = model
                return payload
        except (json.JSONDecodeError, ValueError):
            # treat as plain text content token
            pass
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "delta": {"content": chunk},
                    "index": 0,
                    "finish_reason": None,
                }
            ],
        }
    else:
        # For any other type, treat as plain content
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "delta": {"content": str(chunk)},
                    "index": 0,
                    "finish_reason": None,
                }
            ],
        }


def format_streaming_chunk_as_sse(
    chunk: Any,
    model: str,
    chunk_id: Optional[str] = None,
) -> str:
    """
    Format a streaming chunk as a Server-Sent Event (SSE) data line.

    Args:
        chunk: The chunk from StreamingHandler
        model: The model name to include in the chunk
        chunk_id: Optional ID for the chunk

    Returns:
        A formatted SSE string (e.g., "data: {...}\\n\\n")
    """
    payload = format_streaming_chunk(chunk, model, chunk_id)
    data = json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n"

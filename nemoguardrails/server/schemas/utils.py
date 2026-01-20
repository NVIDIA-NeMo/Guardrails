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
import time
import uuid
from typing import Any, Dict, Optional, Tuple, Union

from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.server.schemas.openai import GuardrailsChatCompletion, GuardrailsDataOutput


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
            return {"role": "assistant", "content": bot_message_content}
        else:
            return bot_message_content
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
                message=ChatCompletionMessage(
                    role="assistant",
                    content=bot_message.get("content", ""),
                ),
                finish_reason="stop",
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
        # If chunk is a dict, wrap it in OpenAI chunk format with delta
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

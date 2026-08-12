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

"""Anthropic Messages API endpoints for the NeMo Guardrails server.

Provides /v1/messages (guardrailed inference proxy) and /v1/messages/checks
(guardrail-only analysis without proxying inference).
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse

from nemoguardrails.llm.anthropic_utils import (
    anthropic_content_to_text,
    anthropic_to_nemo_messages,
    anthropic_tool_use_to_openai,
)
from nemoguardrails.rails.llm.options import GenerationResponse, RailStatus
from nemoguardrails.server.api import (
    _get_rails,
    api_request_headers,
    registered_loggers,
)
from nemoguardrails.server.schemas.anthropic import (
    AnthropicMessagesRequest,
    GuardrailsMessagesCheckRequest,
    GuardrailsMessagesRequest,
)
from nemoguardrails.server.schemas.openai import GuardrailCheckResponse

log = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Anthropic format helpers
# =============================================================================


def _build_anthropic_message_response(
    content: str,
    model: str,
    *,
    stop_reason: str = "end_turn",
    input_tokens: int = 0,
    output_tokens: int = 0,
    thinking_blocks: Optional[List[Dict[str, Any]]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    guardrails_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a non-streaming Anthropic Messages API response dict.

    Assembles content blocks in Anthropic's required order:
    thinking blocks first, then text, then tool_use. Sets ``stop_reason``
    to ``"tool_use"`` when tool calls are present.
    """
    content_blocks: List[Dict[str, Any]] = []
    if thinking_blocks:
        content_blocks.extend(thinking_blocks)
    if content:
        content_blocks.append({"type": "text", "text": content})
    if tool_calls:
        for tc in tool_calls:
            func = tc.get("function", {})
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": func.get("name", ""),
                    "input": func.get("arguments", {}),
                }
            )
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    if tool_calls:
        stop_reason = "tool_use"

    response: Dict[str, Any] = {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }
    if guardrails_data is not None:
        response["guardrails"] = guardrails_data
    return response


def _build_error_message_response(
    model: str,
    error_message: str,
) -> Dict[str, Any]:
    """Build an Anthropic-format error response with the error in both content and guardrails metadata."""
    return _build_anthropic_message_response(
        content=error_message,
        model=model,
        stop_reason="end_turn",
        guardrails_data={"error": error_message},
    )


async def _format_anthropic_sse(
    stream_iterator,
    model: str,
) -> Any:
    """Convert a NeMo streaming iterator into Anthropic SSE events.

    Yields the full Anthropic streaming protocol: ``message_start``,
    ``content_block_start/delta/stop`` for text (index 0), then for each
    tool call (indices 1..N), and finally ``message_delta`` and
    ``message_stop``.

    Tool calls arrive as a JSON string chunk ``{"tool_calls": [...]}``,
    pushed by ``_stream_llm_call`` before ``handler.finish()``. They are
    held until the text block closes, then emitted as ``tool_use`` blocks.
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    total_output_tokens = 0
    tool_calls_data: Optional[List[Dict[str, Any]]] = None

    async for chunk in stream_iterator:
        # Intercept tool call payloads injected by _stream_llm_call.
        # These arrive as JSON strings before END_OF_STREAM and must be
        # held until the text block closes, then emitted as tool_use blocks.
        if isinstance(chunk, str):
            try:
                parsed = json.loads(chunk)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if parsed and isinstance(parsed, dict) and "tool_calls" in parsed:
                tool_calls_data = parsed["tool_calls"]
                continue
            text = chunk
        elif isinstance(chunk, dict):
            if "tool_calls" in chunk:
                tool_calls_data = chunk["tool_calls"]
                continue
            text = chunk.get("content", chunk.get("text", ""))
        else:
            text = str(chunk)

        if text:
            total_output_tokens += max(1, len(text) // 4)
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"

    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

    # Emit tool calls as separate content blocks after the text block.
    # Each tool_use block gets its own index (1, 2, ...) following the
    # text block at index 0, matching Anthropic's streaming protocol.
    if tool_calls_data:
        for i, tc in enumerate(tool_calls_data):
            block_index = i + 1
            func = tc.get("function", {})
            tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
            tool_name = func.get("name", "")
            raw_args = func.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    args_dict = json.loads(raw_args)
                except (json.JSONDecodeError, ValueError):
                    args_dict = {}
            else:
                args_dict = raw_args
            args_json = json.dumps(args_dict)

            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': tool_name, 'input': {}}})}\n\n"

            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"

            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"

            total_output_tokens += max(1, len(args_json) // 4)

    stop_reason = "tool_use" if tool_calls_data else "end_turn"
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': total_output_tokens}})}\n\n"

    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


# =============================================================================
# Message conversion for checks
# =============================================================================


def _anthropic_messages_to_check_format(
    system: Optional[Any],
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert Anthropic messages to the OpenAI-style format used by ``check_async``.

    Similar to ``anthropic_to_nemo_messages`` but produces plain dicts
    (not ``ChatMessage`` objects) for the guardrail-only checks endpoint.
    """
    result: List[Dict[str, Any]] = []

    if system:
        text = anthropic_content_to_text(system) if not isinstance(system, str) else system
        result.append({"role": "system", "content": text})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            result.append({"role": "user", "content": anthropic_content_to_text(content)})

        elif role == "assistant":
            if isinstance(content, list):
                tool_calls = anthropic_tool_use_to_openai(content)
                text = anthropic_content_to_text(content)
                entry: Dict[str, Any] = {"role": "assistant", "content": text}
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                result.append(entry)
            else:
                result.append({"role": "assistant", "content": str(content) if content else ""})

        elif role == "tool_result":
            result.append(
                {
                    "role": "tool",
                    "content": anthropic_content_to_text(content),
                    "tool_call_id": msg.get("tool_use_id", ""),
                }
            )

    return result


# =============================================================================
# /v1/messages endpoint
# =============================================================================


@router.post("/v1/messages")
async def messages_endpoint(body: GuardrailsMessagesRequest, request: Request):
    """Anthropic Messages API compatible endpoint with guardrails.

    Accepts Anthropic-format requests, applies guardrails, and returns
    Anthropic-format responses (both streaming and non-streaming).
    """
    log.info("Got messages request for config %s", body.guardrails.config_id)
    for logger in registered_loggers:
        asyncio.get_event_loop().create_task(logger({"endpoint": "/v1/messages", "body": body.model_dump_json()}))

    api_request_headers.set(request.headers)

    config_ids = body.guardrails.config_ids
    if not config_ids:
        if request.app.default_config_id:
            config_ids = [request.app.default_config_id]
        else:
            raise HTTPException(
                status_code=422,
                detail="No guardrails config_id provided and server has no default configuration",
            )

    try:
        llm_rails = await _get_rails(config_ids, model_name=body.model)
    except ValueError as ex:
        log.exception(ex)
        return _build_error_message_response(
            model=body.model,
            error_message=f"Could not load the {config_ids} guardrails configuration.",
        )

    try:
        nemo_messages = anthropic_to_nemo_messages(body.system, body.messages)
        messages_dicts = [m.to_dict() for m in nemo_messages]

        if body.guardrails.context:
            messages_dicts.insert(0, {"role": "context", "content": body.guardrails.context})

        generation_options = body.guardrails.options

        if generation_options.llm_params is None:
            generation_options.llm_params = {}

        # the following fields are all superseded by NeMo Guardrails in some way, and so shouldn't be auto-forwarded
        _FIELDS_NOT_FORWARDED = {"messages", "model", "system", "stream"}
        for field in AnthropicMessagesRequest.model_fields:
            if field in _FIELDS_NOT_FORWARDED:
                continue
            value = getattr(body, field)
            if value is not None:
                generation_options.llm_params[field] = value

        if body.stream:
            stream_iterator = llm_rails.stream_async(
                messages=messages_dicts,
                options=generation_options,
                state=body.guardrails.state,
            )

            return StreamingResponse(
                _format_anthropic_sse(stream_iterator, model=body.model),
                media_type="text/event-stream",
            )
        else:
            res = await llm_rails.generate_async(
                messages=messages_dicts,
                options=generation_options,
                state=body.guardrails.state,
            )

            if isinstance(res, GenerationResponse):
                if isinstance(res.response, str):
                    content = res.response
                elif res.response:
                    content = res.response[0].get("content", "")
                else:
                    content = ""
                guardrails_data = None
                if res.log:
                    guardrails_data = {
                        "log": res.log.model_dump() if hasattr(res.log, "model_dump") else {},
                    }

                thinking_blocks = None
                if res.llm_metadata and "thinking_blocks" in res.llm_metadata:
                    thinking_blocks = res.llm_metadata["thinking_blocks"]
                elif res.reasoning_content:
                    thinking_blocks = [{"type": "thinking", "thinking": res.reasoning_content}]

                return _build_anthropic_message_response(
                    content=content,
                    model=body.model,
                    thinking_blocks=thinking_blocks,
                    tool_calls=res.tool_calls,
                    guardrails_data=guardrails_data,
                )
            else:
                content = ""
                if isinstance(res, dict):
                    content = res.get("content", "")
                elif isinstance(res, str):
                    content = res
                return _build_anthropic_message_response(
                    content=content,
                    model=body.model,
                )

    except HTTPException:
        raise
    except Exception as ex:
        log.exception(ex)
        return _build_error_message_response(
            model=body.model,
            error_message="Internal server error",
        )


# =============================================================================
# /v1/messages/checks endpoint
# =============================================================================


def _map_rail_status(status: RailStatus) -> str:
    return status.value


@router.post(
    "/v1/messages/checks",
    response_model=GuardrailCheckResponse,
    response_model_exclude_none=True,
)
async def messages_check_endpoint(body: GuardrailsMessagesCheckRequest, request: Request):
    """Check Anthropic Messages against guardrails without generating LLM responses.

    Mirrors /v1/checks but accepts Anthropic Messages API format requests.
    Returns the same GuardrailCheckResponse schema.
    """
    api_request_headers.set(request.headers)

    check_messages = _anthropic_messages_to_check_format(body.system, body.messages)

    if not check_messages:
        raise HTTPException(status_code=422, detail="messages must be non-empty")

    config_ids = body.guardrails.config_ids
    if not config_ids:
        if request.app.default_config_id:
            config_ids = [request.app.default_config_id]
        else:
            raise HTTPException(
                status_code=422,
                detail="No guardrails config_id provided and server has no default configuration",
            )
    try:
        llm_rails = await _get_rails(config_ids, model_name=body.model)
    except ValueError as ex:
        log.exception(ex)
        raise HTTPException(status_code=422, detail=str(ex))

    if llm_rails.config.colang_version != "1.0":
        raise HTTPException(
            status_code=422,
            detail="check_async does not support Colang 2.0 configurations.",
        )

    try:
        if body.guardrails.context:
            check_messages.insert(0, {"role": "context", "content": body.guardrails.context})

        result = await llm_rails.check_async(messages=check_messages)

        return GuardrailCheckResponse(
            status=_map_rail_status(result.status),
            content=result.content,
            rail=result.rail,
        )

    except HTTPException:
        raise
    except Exception as ex:
        log.exception(ex)
        raise HTTPException(status_code=500, detail="Internal server error")

# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Buffered output rail streaming helpers."""

import json
import logging
from functools import partial
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Protocol

from nemoguardrails.actions.output_mapping import is_output_blocked
from nemoguardrails.rails.llm.buffer import get_buffer_strategy
from nemoguardrails.rails.llm.config import OutputRailsStreamingConfig
from nemoguardrails.rails.llm.utils import get_action_details_from_flow_id

log = logging.getLogger(__name__)

__all__ = [
    "StreamingOutputActionDispatcher",
    "StreamingOutputRails",
    "StreamingOutputRuntime",
    "run_output_rails_in_streaming",
]


class StreamingOutputActionDispatcher(Protocol):
    async def execute_action(self, action_name: str, params: Dict[str, Any]) -> Any: ...

    def get_action(self, name: str, /) -> Any: ...


class StreamingOutputRuntime(Protocol):
    @property
    def action_dispatcher(self) -> StreamingOutputActionDispatcher: ...

    @property
    def llm_task_manager(self) -> Any: ...

    @property
    def registered_action_params(self) -> Dict[str, Any]: ...


class StreamingOutputRails(Protocol):
    _explain_info: Any

    @property
    def config(self) -> Any: ...

    @property
    def runtime(self) -> StreamingOutputRuntime: ...

    @property
    def llm(self) -> Any: ...

    def _ensure_explain_info(self) -> Any: ...


async def run_output_rails_in_streaming(
    rails: StreamingOutputRails,
    streaming_handler: AsyncIterator[str],
    output_rails_streaming_config: OutputRailsStreamingConfig,
    prompt: Optional[str] = None,
    messages: Optional[List[dict]] = None,
    stream_first: Optional[bool] = None,
) -> AsyncIterator[str]:
    """
    1. Buffers tokens from 'streaming_handler' via BufferStrategy.
    2. Runs sequential (parallel for colang 2.0 in future) flows for each chunk.
    3. Yields the chunk if not blocked, or STOP if blocked.
    """
    buffer_strategy = get_buffer_strategy(output_rails_streaming_config)
    output_rails_flows_id = rails.config.rails.output.flows
    stream_first = stream_first or output_rails_streaming_config.stream_first
    get_action_details = partial(get_action_details_from_flow_id, flows=rails.config.flows)

    parallel_mode = getattr(rails.config.rails.output, "parallel", False)

    async for chunk_batch in buffer_strategy(streaming_handler):
        user_output_chunks = chunk_batch.user_output_chunks
        # format processing_context for output rails processing (needs full context)
        bot_response_chunk = buffer_strategy.format_chunks(chunk_batch.processing_context)

        # check if user_output_chunks is a list of individual chunks
        # or if it's a JSON string, by convention this means an error occurred and the error dict is stored as a JSON
        if not isinstance(user_output_chunks, list):
            try:
                json.loads(user_output_chunks)
                yield user_output_chunks
                return
            except (json.JSONDecodeError, TypeError):
                # if it's not JSON, treat it as empty list
                user_output_chunks = []

        if stream_first:
            # yield the individual chunks directly from the buffer strategy
            for chunk in user_output_chunks:
                yield chunk

        if parallel_mode:
            async for chunk in _run_parallel_output_rails(
                rails=rails,
                output_rails_flows_id=output_rails_flows_id,
                get_action_details=get_action_details,
                bot_response_chunk=bot_response_chunk,
                prompt=prompt,
                messages=messages,
            ):
                yield chunk
                return
        else:
            async for chunk in _run_sequential_output_rails(
                rails=rails,
                output_rails_flows_id=output_rails_flows_id,
                get_action_details=get_action_details,
                bot_response_chunk=bot_response_chunk,
                prompt=prompt,
                messages=messages,
            ):
                yield chunk
                return

        if not stream_first:
            # yield the individual chunks directly from the buffer strategy
            for chunk in user_output_chunks:
                yield chunk


async def _run_parallel_output_rails(
    *,
    rails: StreamingOutputRails,
    output_rails_flows_id: List[str],
    get_action_details: Callable[[str], tuple[str, Dict[str, Any]]],
    bot_response_chunk: str,
    prompt: Optional[str],
    messages: Optional[List[dict]],
) -> AsyncIterator[str]:
    try:
        context = _prepare_context_for_parallel_rails(bot_response_chunk, prompt, messages)
        events = _create_events_for_chunk(bot_response_chunk, context)

        flows_with_params = {}
        for flow_id in output_rails_flows_id:
            action_name, action_params = get_action_details(flow_id)
            params = _prepare_params(
                rails=rails,
                flow_id=flow_id,
                action_name=action_name,
                bot_response_chunk=bot_response_chunk,
                prompt=prompt,
                messages=messages,
                action_params=action_params,
            )
            flows_with_params[flow_id] = {
                "action_name": action_name,
                "params": params,
            }

        result_tuple = await rails.runtime.action_dispatcher.execute_action(
            "run_output_rails_in_parallel_streaming",
            {
                "flows_with_params": flows_with_params,
                "events": events,
            },
        )

        # ActionDispatcher.execute_action always returns (result, status)
        result, status = result_tuple

        if status != "success":
            log.error(f"Parallel rails execution failed with status: {status}")
        else:
            # if there are any stop events, content was blocked or internal error occurred
            result_events = getattr(result, "events", None)
            if result_events:
                yield _parallel_stop_error(result_events[0])
                return

    except Exception as e:
        log.error(f"Error in parallel rail execution: {e}")

    # update explain info for parallel mode
    rails._explain_info = rails._ensure_explain_info()


async def _run_sequential_output_rails(
    *,
    rails: StreamingOutputRails,
    output_rails_flows_id: List[str],
    get_action_details: Callable[[str], tuple[str, Dict[str, Any]]],
    bot_response_chunk: str,
    prompt: Optional[str],
    messages: Optional[List[dict]],
) -> AsyncIterator[str]:
    for flow_id in output_rails_flows_id:
        action_name, action_params = get_action_details(flow_id)

        params = _prepare_params(
            rails=rails,
            flow_id=flow_id,
            action_name=action_name,
            bot_response_chunk=bot_response_chunk,
            prompt=prompt,
            messages=messages,
            action_params=action_params,
        )

        action_result = await rails.runtime.action_dispatcher.execute_action(action_name, params)
        rails._explain_info = rails._ensure_explain_info()

        # Use the mapping to decide if the result indicates blocked content.
        action_func = rails.runtime.action_dispatcher.get_action(action_name)
        if is_output_blocked(action_result, action_func):
            yield _blocked_output_error(flow_id)
            return


def _get_last_context_message(
    messages: Optional[List[dict]] = None,
) -> dict:
    if messages is None:
        return {}

    for message in reversed(messages):
        if message.get("role") == "context":
            return message
    return {}


def _get_latest_user_message(
    messages: Optional[List[dict]] = None,
) -> str:
    if messages is None:
        return ""
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _prepare_context_for_parallel_rails(
    chunk_str: str,
    prompt: Optional[str] = None,
    messages: Optional[List[dict]] = None,
) -> dict:
    """Prepare context for parallel rails execution."""
    context_message = _get_last_context_message(messages)
    user_message = prompt or _get_latest_user_message(messages)

    context = {
        "user_message": user_message,
        "bot_message": chunk_str,
    }

    if context_message:
        context.update(context_message["content"])

    return context


def _create_events_for_chunk(chunk_str: str, context: dict) -> List[dict]:
    """Create events for running output rails on a chunk."""
    return [
        {"type": "ContextUpdate", "data": context},
        {"type": "BotMessage", "text": chunk_str},
    ]


def _prepare_params(
    *,
    rails: StreamingOutputRails,
    flow_id: str,
    action_name: str,
    bot_response_chunk: str,
    prompt: Optional[str] = None,
    messages: Optional[List[dict]] = None,
    action_params: Dict[str, Any],
):
    context_message = _get_last_context_message(messages)
    user_message = prompt or _get_latest_user_message(messages)

    context = {
        "user_message": user_message,
        "bot_message": bot_response_chunk,
    }

    if context_message:
        context.update(context_message["content"])

    model_name = flow_id.split("$")[-1].split("=")[-1].strip('"')

    resolved_params = dict(action_params or {})
    for key, value in resolved_params.items():
        if value == "$bot_message":
            resolved_params[key] = bot_response_chunk
        elif value == "$user_message":
            resolved_params[key] = user_message

    return {
        # TODO:: are there other context variables that need to be passed?
        # passing events to compute context was not successful
        # context var failed due to different context
        "context": context,
        "llm_task_manager": rails.runtime.llm_task_manager,
        "config": rails.config,
        "model_name": model_name,
        "llms": rails.runtime.registered_action_params.get("llms", {}),
        "llm": rails.runtime.registered_action_params.get(f"{action_name}_llm", rails.llm),
        **resolved_params,
    }


def _blocked_output_error(flow_id: str) -> str:
    reason = f"Blocked by {flow_id} rails."

    # return the error as a plain JSON string (not in SSE format)
    # NOTE: When integrating with the OpenAI Python client, the server code should:
    # 1. detect this JSON error object in the stream
    # 2. terminate the stream
    # 3. format the error following OpenAI's SSE format
    # the OpenAI client will then properly raise an APIError with this error message
    error_data = {
        "error": {
            "message": reason,
            "type": "guardrails_violation",
            "param": flow_id,
            "code": "content_blocked",
        }
    }

    # return as plain JSON: the server should detect this JSON and convert it to an HTTP error
    return json.dumps(error_data)


def _internal_rail_error(flow_id: str, error_message: str) -> str:
    error_data = {
        "error": {
            "message": f"Internal error in {flow_id} rail: {error_message}",
            "type": "internal_error",
            "param": flow_id,
            "code": "rail_execution_failure",
        }
    }
    return json.dumps(error_data)


def _parallel_stop_error(stop_event: dict) -> str:
    blocked_flow = stop_event.get("flow_id", "output rails")
    error_type = stop_event.get("error_type")

    if error_type == "internal_error":
        error_message = stop_event.get("error_message", "Unknown error")
        return _internal_rail_error(blocked_flow, error_message)
    else:
        reason = f"Blocked by {blocked_flow} rails."
        error_code = "content_blocked"
        error_type = "guardrails_violation"

    error_data = {
        "error": {
            "message": reason,
            "type": error_type,
            "param": blocked_flow,
            "code": error_code,
        }
    }
    return json.dumps(error_data)

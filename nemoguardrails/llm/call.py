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

"""The shared entry point for invoking an LLM model.

``llm_call`` wraps an :class:`~nemoguardrails.types.LLMModel` with the project's
prompt logging, token accounting, reasoning-trace extraction, tool-call capture, and
error wrapping, so every caller gets the same observability and the same
:class:`~nemoguardrails.exceptions.LLMCallException` contract.

This module deliberately carries no Colang dependency. Built-in rail actions in
``nemoguardrails.library`` call ``llm_call``, and those actions must stay runnable by
engines that have no Colang runtime; ``tests/llm/test_call_import_graph.py`` enforces it.

Neighbouring modules: text helpers for interpreting a completion live in
:mod:`nemoguardrails.llm.completion_parsing`, and the Colang-specific history and event
helpers in :mod:`nemoguardrails.actions.llm.utils`.
"""

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, NoReturn, Optional, Union, cast

from nemoguardrails.context import (
    llm_call_info_var,
    llm_response_metadata_var,
    llm_stats_var,
    reasoning_trace_var,
    tool_calls_var,
)
from nemoguardrails.exceptions import LLMCallException, LLMClientError
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.logging.llm_tracker import track_llm_call
from nemoguardrails.types import ChatMessage, LLMModel, LLMResponse, LLMResponseChunk, UsageInfo

if TYPE_CHECKING:
    from nemoguardrails.streaming import StreamingHandler

logger = logging.getLogger(__name__)


def _ensure_chat_messages(prompt: Union[str, list]) -> Union[str, List[ChatMessage]]:
    if isinstance(prompt, str):
        return prompt
    if not prompt:
        return cast(List[ChatMessage], [])
    if isinstance(prompt[0], ChatMessage):
        return cast(List[ChatMessage], prompt)
    return [ChatMessage.from_dict(d) for d in prompt]


# TODO: we must drop prompt in the codebase completely and use messages everywhere.
@track_llm_call
async def llm_call(
    llm: Optional[Any],
    prompt: Union[str, List[dict]],
    model_name: Optional[str] = None,
    model_provider: Optional[str] = None,
    stop: Optional[List[str]] = None,
    llm_params: Optional[dict] = None,
    streaming_handler: Optional["StreamingHandler"] = None,
) -> LLMResponse:
    if llm is None:
        raise LLMCallException(ValueError("No LLM provided to llm_call()"))

    model: LLMModel
    if isinstance(llm, LLMModel):
        model = llm
    else:
        raise TypeError(
            f"Expected an LLMModel instance, got {type(llm).__name__}. "
            "Wrap your LLM with an appropriate adapter before passing it to llm_call()."
        )

    _setup_llm_call_info(model, model_name, model_provider)
    _log_prompt(prompt)
    chat_prompt = _ensure_chat_messages(prompt)

    if streaming_handler:
        return await _stream_llm_call(model, chat_prompt, streaming_handler, stop, llm_params)

    try:
        response: LLMResponse = await model.generate_async(chat_prompt, stop=stop, **(llm_params or {}))
    except Exception as e:
        _raise_llm_call_exception(e, model)

    _store_reasoning_traces(response)
    _log_completion(response)
    _update_token_stats(response)
    _store_tool_calls(response)
    _store_request_id(response)
    _store_response_metadata(response)
    return response


async def _stream_llm_call(
    model: LLMModel,
    prompt: Union[str, List[ChatMessage]],
    handler: "StreamingHandler",
    stop: Optional[List[str]],
    llm_params: Optional[dict] = None,
) -> LLMResponse:
    handler.stop = stop or []
    streaming_handler_metadata: Dict[str, Any] = {}
    accumulated_provider_metadata: Dict[str, Any] = {}
    accumulated_reasoning: List[str] = []
    tool_calls = None
    model_name: Optional[str] = None
    finish_reason: Optional[str] = None
    request_id: Optional[str] = None
    usage: Optional[UsageInfo] = None

    try:
        async for chunk in model.stream_async(prompt, stop=stop, **(llm_params or {})):
            content = chunk.delta_content or ""

            if chunk.delta_reasoning:
                accumulated_reasoning.append(chunk.delta_reasoning)
            if chunk.delta_tool_calls:
                tool_calls = chunk.delta_tool_calls
            if chunk.model:
                model_name = chunk.model
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.request_id:
                request_id = chunk.request_id
            if chunk.usage:
                usage = chunk.usage
            if chunk.provider_metadata:
                accumulated_provider_metadata.update(chunk.provider_metadata)

            chunk_metadata = _extract_chunk_metadata(chunk)
            if chunk_metadata:
                streaming_handler_metadata.update(chunk_metadata)

            await handler.push_chunk(content, chunk_metadata)

        llm_response_metadata_var.set(accumulated_provider_metadata or None)

        # Push tool calls to the streaming consumer before finish().
        # finish() sends END_OF_STREAM which terminates the consumer, so
        # tool calls must be injected first. The SSE formatter in
        # messages.py detects this JSON payload and emits tool_use blocks.
        if tool_calls:
            payload = json.dumps({"tool_calls": [tc.to_dict() for tc in tool_calls]})
            await handler.push_chunk(payload)

        await handler.finish()

        llm_call_info = llm_call_info_var.get()
        if llm_call_info:
            llm_call_info.completion = handler.completion

        if usage:
            fake_chunk = LLMResponseChunk(usage=usage)
            _update_token_stats_from_chunk(fake_chunk)

        if tool_calls:
            tool_calls_var.set([tc.to_dict() for tc in tool_calls])
        else:
            tool_calls_var.set(None)

        reasoning_content = "".join(accumulated_reasoning) if accumulated_reasoning else None
        # TODO: call _extract_and_remove_think_tags on the completed response
        # to handle models that stream reasoning via <think> tags in content
        # rather than via delta_reasoning. Pre-existing gap, not introduced here.
        reasoning_trace_var.set(reasoning_content)

        response = LLMResponse(
            content=handler.completion,
            reasoning=reasoning_content,
            tool_calls=tool_calls,
            model=model_name,
            finish_reason=finish_reason,
            request_id=request_id,
            usage=usage,
            provider_metadata=accumulated_provider_metadata or None,
        )
        # Streaming returns early from llm_call, so this must happen here too.
        _store_request_id(response)
        return response

    except Exception as e:
        _raise_llm_call_exception(e, model)


def _extract_chunk_metadata(chunk: LLMResponseChunk) -> Optional[Dict[str, Any]]:
    # This feeds handler.push_chunk() for the StreamingHandler consumer path
    # (API responses, output rails). Separate from the field accumulation in
    # _stream_llm_call which builds the returned LLMResponse for the pipeline.
    # TODO(Pouyanpi): consider pushing tool_calls and reasoning through the handler too,
    # so output rails and streaming consumers can see them in real-time.
    metadata: Dict[str, Any] = {}
    if chunk.provider_metadata:
        metadata["provider_metadata"] = chunk.provider_metadata
    if chunk.usage:
        metadata["usage"] = {
            "input_tokens": chunk.usage.input_tokens,
            "output_tokens": chunk.usage.output_tokens,
            "total_tokens": chunk.usage.total_tokens,
        }
    return metadata if metadata else None


def _setup_llm_call_info(model: LLMModel, model_name: Optional[str], model_provider: Optional[str]) -> None:
    """Initialize or update LLM call info in context."""
    llm_call_info = llm_call_info_var.get()
    if llm_call_info is None:
        llm_call_info = LLMCallInfo()
        llm_call_info_var.set(llm_call_info)

    llm_call_info.llm_model_name = model_name or model.model_name
    llm_call_info.llm_provider_name = model_provider or model.provider_name


def _prompt_message_role(message: Union[dict, ChatMessage]) -> str:
    """Return a message's role, accepting either a raw dict or a ``ChatMessage``.

    ``Role`` is a ``str`` subclass, so it is returned as-is: it maps correctly through
    ``type_map`` and supports ``str`` methods. Passing it through ``str()`` would yield
    ``"Role.USER"`` because ``Enum.__str__`` wins over the ``str`` mixin.
    """
    if isinstance(message, ChatMessage):
        return message.role or ""
    return message.get("role") or message.get("type") or ""


def _prompt_message_content(message: Union[dict, ChatMessage]) -> str:
    """Return a message's textual content, or "" when it is absent or non-textual.

    Multimodal payloads carry a list of content parts rather than a string; those are
    omitted from the logged prompt, matching the previous dict-only behavior.
    """
    content = message.content if isinstance(message, ChatMessage) else message.get("content", "")
    return content if isinstance(content, str) else ""


def _log_prompt(prompt: Union[str, List[dict], List[ChatMessage]]) -> None:
    """Log the prompt to LLM call info.

    Accepts the normalized ``ChatMessage`` form as well as raw dicts, because
    ``llm_call`` logs the prompt before ``_ensure_chat_messages`` runs and callers may
    already supply ``ChatMessage`` objects.
    """
    llm_call_info = llm_call_info_var.get()
    if llm_call_info is None:
        return

    if isinstance(prompt, str):
        llm_call_info.prompt = prompt
        logger.info("Prompt :: %s", prompt, extra={"id": llm_call_info.id, "task": llm_call_info.task})
    else:
        type_map = {
            "human": "User",
            "ai": "Bot",
            "tool": "Tool",
            "system": "System",
            "developer": "Developer",
            "user": "User",
            "assistant": "Bot",
        }
        formatted_prompt = "\n" + "\n".join(
            [
                "[cyan]"
                + type_map.get(_prompt_message_role(msg), _prompt_message_role(msg).title())
                + "[/]"
                + "\n"
                + _prompt_message_content(msg)
                for msg in prompt
            ]
        )
        llm_call_info.prompt = formatted_prompt
        logger.info(
            "Prompt Messages :: %s",
            formatted_prompt,
            extra={"id": llm_call_info.id, "task": llm_call_info.task},
        )


def _log_completion(response: LLMResponse) -> None:
    llm_call_info = llm_call_info_var.get()
    if llm_call_info is None:
        return

    completion_text = _extract_content(response)
    llm_call_info.completion = completion_text

    if response.reasoning:
        full_completion = f"{response.reasoning}\n---\n{completion_text}"
    else:
        full_completion = completion_text

    logger.info(
        "Completion :: %s",
        full_completion,
        extra={"id": llm_call_info.id, "task": llm_call_info.task},
    )


def _update_token_stats(response: LLMResponse) -> None:
    llm_call_info = llm_call_info_var.get()
    llm_stats = llm_stats_var.get()

    if llm_call_info is None:
        return

    if not llm_call_info.total_tokens:
        llm_call_info.total_tokens = 0
    if not llm_call_info.prompt_tokens:
        llm_call_info.prompt_tokens = 0
    if not llm_call_info.completion_tokens:
        llm_call_info.completion_tokens = 0

    if response.usage:
        llm_call_info.total_tokens = response.usage.total_tokens
        llm_call_info.prompt_tokens = response.usage.input_tokens
        llm_call_info.completion_tokens = response.usage.output_tokens

        if llm_stats:
            llm_stats.inc("total_tokens", response.usage.total_tokens)
            llm_stats.inc("total_prompt_tokens", response.usage.input_tokens)
            llm_stats.inc("total_completion_tokens", response.usage.output_tokens)
    else:
        logger.info("Token stats in LLM call info cannot be computed for current model!")


def _update_token_stats_from_chunk(chunk: LLMResponseChunk) -> None:
    llm_call_info = llm_call_info_var.get()
    llm_stats = llm_stats_var.get()

    if llm_call_info is None:
        return

    if not llm_call_info.total_tokens:
        llm_call_info.total_tokens = 0
    if not llm_call_info.prompt_tokens:
        llm_call_info.prompt_tokens = 0
    if not llm_call_info.completion_tokens:
        llm_call_info.completion_tokens = 0

    if chunk.usage:
        llm_call_info.total_tokens = chunk.usage.total_tokens
        llm_call_info.prompt_tokens = chunk.usage.input_tokens
        llm_call_info.completion_tokens = chunk.usage.output_tokens

        if llm_stats:
            llm_stats.inc("total_tokens", chunk.usage.total_tokens)
            llm_stats.inc("total_prompt_tokens", chunk.usage.input_tokens)
            llm_stats.inc("total_completion_tokens", chunk.usage.output_tokens)


def _extract_http_status(exception: BaseException) -> Optional[int]:
    """Extract an HTTP status code from a provider exception, if present.

    Checks, in order:
    1. ``LLMClientError.status_code`` (NeMo Guardrails client layer).
    2. ``exception.status_code`` (OpenAI SDK, httpx).
    3. ``exception.response.status_code`` (requests-style wrappers).

    Returns ``None`` when no status can be determined or when the status
    is ``0`` (used by ``LLMTimeoutError`` / ``LLMConnectionError`` for
    client-side failures where no HTTP response was received).
    """
    if isinstance(exception, LLMClientError):
        return exception.status_code if exception.status_code > 0 else None

    status = getattr(exception, "status_code", None)
    if isinstance(status, int) and status > 0:
        return status

    response = getattr(exception, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if isinstance(status, int) and status > 0:
            return status

    return None


def _raise_llm_call_exception(
    exception: Exception,
    model: LLMModel,
) -> NoReturn:
    llm_call_info = llm_call_info_var.get()
    model_name = llm_call_info.llm_model_name if llm_call_info else model.model_name
    provider_name = llm_call_info.llm_provider_name if llm_call_info else model.provider_name
    endpoint_url = model.provider_url

    context_parts = []
    if model_name:
        context_parts.append(f"model={model_name}")
    if provider_name:
        context_parts.append(f"provider={provider_name}")
    if endpoint_url:
        context_parts.append(f"endpoint={endpoint_url}")

    status = _extract_http_status(exception)

    if context_parts:
        detail = f"Error invoking LLM ({', '.join(context_parts)})"
        raise LLMCallException(exception, detail=detail, status=status) from exception

    raise LLMCallException(exception, status=status) from exception


def _store_reasoning_traces(response: LLMResponse) -> None:
    reasoning_content = response.reasoning

    if not reasoning_content:
        reasoning_content = _extract_and_remove_think_tags(response)

    reasoning_trace_var.set(reasoning_content)


def _extract_and_remove_think_tags(response: LLMResponse) -> Optional[str]:
    """Extract reasoning from <think> tags and remove them from `response.content`.

    This function looks for <think>...</think> tags in the response content,
    and if found, extracts the reasoning content inside the tags. It has a side-effect:
    it removes the full reasoning trace and tags from response.content.

    Args:
        response: The LLM response object

    Returns:
        The extracted reasoning content, or None if no <think> tags found
    """
    content = response.content
    has_opening_tag = "<think>" in content
    has_closing_tag = "</think>" in content

    if not has_opening_tag and not has_closing_tag:
        return None

    if has_opening_tag != has_closing_tag:
        logger.warning(
            "Malformed <think> tags detected: missing %s tag. "
            "Skipping reasoning extraction to prevent corrupted content.",
            "closing" if has_opening_tag else "opening",
        )
        return None

    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if match:
        reasoning_content = match.group(1).strip()
        response.content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return reasoning_content
    return None


def _store_tool_calls(response: LLMResponse) -> None:
    if response.tool_calls:
        tool_calls_var.set([tc.to_dict() for tc in response.tool_calls])
    else:
        tool_calls_var.set(None)


def _store_request_id(response: LLMResponse) -> None:
    """Record the provider's response id on the current call, when it returned one.

    Kept separate from ``LLMCallInfo.id``, which ``track_llm_call`` generates client-side:
    only this value can be quoted to a provider, and only this value matches
    ``gen_ai.response.id`` on the OTEL span for the same call, so overloading one field with
    both meanings would make log-to-trace correlation unreliable.
    """
    llm_call_info = llm_call_info_var.get()
    if llm_call_info is None:
        return
    llm_call_info.request_id = response.request_id


def _store_response_metadata(response: LLMResponse) -> None:
    llm_response_metadata_var.set(response.provider_metadata)


def _extract_content(response: LLMResponse) -> str:
    """Extract text content from response."""
    return response.content


def warn_if_truncated(response: LLMResponse, task: str) -> bool:
    """Return True and emit a warning if the LLM produced no visible content because it hit the max_tokens budget.

    Reasoning models (OpenAI o-series, gpt-5, DeepSeek-R1, Gemini 2.5, Qwen QwQ, etc.)
    spend output tokens on internal reasoning before emitting visible text. A small
    max_tokens budget can be fully consumed by the reasoning phase, leaving empty
    content and finish_reason="length". The call succeeds silently and callers that
    only inspect response.content see nothing. Callers whose downstream parser
    does not fail safely on empty input (e.g. self_check_facts, whose parser
    inverts the result) should use the return value to take an explicit
    fail-safe branch.
    """
    truncated = not response.content and response.finish_reason == "length"
    if truncated:
        logger.warning(
            "Task %s: LLM returned empty content with finish_reason='length'. "
            "The max_tokens budget was likely consumed before any visible output. "
            "If using a reasoning model (o1/o3/o4-mini, gpt-5, deepseek-r1, "
            "gemini-2.5, etc.), increase the prompt's max_tokens in your config.",
            task,
        )
    return truncated

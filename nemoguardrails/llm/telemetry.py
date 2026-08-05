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
import os
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any, Generator, Optional

from nemoguardrails.tracing.constants import EventNames, GenAIAttributes, OtelContentCapture

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

    from nemoguardrails.types import UsageInfo
else:
    try:
        from opentelemetry.trace import SpanKind, StatusCode
    except ImportError:
        SpanKind = None
        StatusCode = None


_LEGACY_EVENT_BY_ROLE = {
    "system": EventNames.GEN_AI_SYSTEM_MESSAGE,
    "user": EventNames.GEN_AI_USER_MESSAGE,
    "assistant": EventNames.GEN_AI_ASSISTANT_MESSAGE,
    "tool": EventNames.GEN_AI_TOOL_MESSAGE,
}

_GENAI_REQUEST_PARAMS = {
    "temperature": GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE,
    "max_tokens": GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS,
    "max_completion_tokens": GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS,
    "top_p": GenAIAttributes.GEN_AI_REQUEST_TOP_P,
    "top_k": GenAIAttributes.GEN_AI_REQUEST_TOP_K,
    "frequency_penalty": GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY,
    "presence_penalty": GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY,
}


def _use_json_span_format() -> bool:
    tokens = {token.strip() for token in os.environ.get(OtelContentCapture.STABILITY_OPT_IN_ENV, "").split(",")}
    return OtelContentCapture.STABILITY_OPT_IN_LATEST in tokens


def _system_parts_from_messages(messages: list[dict[str, Any]]) -> list[dict]:
    return [
        {"type": "text", "content": message["content"]}
        for message in messages
        if message.get("role") == "system" and message.get("content") is not None
    ]


def _non_system_input_messages(messages: list[dict[str, Any]]) -> list[dict]:
    return [
        {
            "role": message["role"],
            "parts": [{"type": "text", "content": message["content"]}],
        }
        for message in messages
        if message.get("role") is not None and message.get("role") != "system" and message.get("content") is not None
    ]


def _set_llm_call_content_json(span: "Span", input_messages: list[dict[str, Any]], output_text: Optional[str]) -> None:
    system_parts = _system_parts_from_messages(input_messages)
    if system_parts:
        span.set_attribute(GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS, json.dumps(system_parts))
    non_system = _non_system_input_messages(input_messages)
    if non_system:
        span.set_attribute(GenAIAttributes.GEN_AI_INPUT_MESSAGES, json.dumps(non_system))
    if output_text is not None:
        output_messages = [{"role": "assistant", "parts": [{"type": "text", "content": output_text}]}]
        span.set_attribute(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages))


def _set_llm_call_content_events(
    span: "Span", input_messages: list[dict[str, Any]], output_text: Optional[str]
) -> None:
    for message in input_messages:
        role = message.get("role")
        content = message.get("content")
        event_name = _LEGACY_EVENT_BY_ROLE.get(role)
        if event_name is not None and isinstance(role, str) and isinstance(content, str):
            span.add_event(event_name, attributes={"role": role, "content": content})
    if output_text is not None:
        span.add_event(
            EventNames.GEN_AI_CHOICE,
            attributes={"index": 0, "message.role": "assistant", "message.content": output_text},
        )


def set_llm_call_content(
    span: Optional["Span"],
    input_messages: list[dict[str, Any]],
    output_text: Optional[str] = None,
) -> None:
    """Record LLM input and output content on a span when available.

    The configured OpenTelemetry content format determines whether content is
    stored as JSON attributes or legacy events. Telemetry failures are ignored
    so they cannot affect the model call.
    """
    if span is None:
        return
    with suppress(Exception):
        if _use_json_span_format():
            _set_llm_call_content_json(span, input_messages, output_text)
        else:
            _set_llm_call_content_events(span, input_messages, output_text)


def _stop_sequences(params: dict) -> Optional[list]:
    value = params.get("stop")
    if value is None:
        value = params.get("stop_sequences")
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, list) and value:
        return value
    return None


def set_llm_request_attributes(span: Optional["Span"], params: dict, *, stream: bool = False) -> None:
    """Record supported GenAI request parameters on a span.

    Unknown parameters are ignored, as are telemetry failures.
    """
    if span is None:
        return
    with suppress(Exception):
        for key, attribute in _GENAI_REQUEST_PARAMS.items():
            value = params.get(key)
            if value is not None:
                span.set_attribute(attribute, value)
        stop_sequences = _stop_sequences(params)
        if stop_sequences is not None:
            span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES, stop_sequences)
        if stream:
            span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_STREAM, True)


def set_llm_response_attributes(
    span: Optional["Span"],
    *,
    model: Optional[str] = None,
    response_id: Optional[str] = None,
    finish_reason: Optional[str] = None,
    usage: Optional["UsageInfo"] = None,
) -> None:
    """Record available model response metadata and token usage on a span."""
    if span is None:
        return
    with suppress(Exception):
        if model is not None:
            span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_MODEL, model)
        if response_id is not None:
            span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_ID, response_id)
        if finish_reason is not None:
            span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason])
        if usage is not None:
            span.set_attribute(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, usage.input_tokens)
            span.set_attribute(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, usage.output_tokens)
            if usage.reasoning_tokens is not None:
                span.set_attribute(
                    GenAIAttributes.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
                    usage.reasoning_tokens,
                )


def record_span_error(span: Optional["Span"], exc: BaseException) -> None:
    """Record error status and the exception type on a span.

    The exception message and stack trace are deliberately omitted: a provider
    error can embed request content, which must not reach telemetry unless
    content capture is explicitly enabled. Only the low-cardinality error type
    is recorded.
    """
    if span is None:
        return
    with suppress(Exception):
        span.set_attribute("error.type", type(exc).__name__)
        span.set_status(StatusCode.ERROR)


@contextmanager
def llm_call_span(
    tracer: Optional["Tracer"],
    model_name: str,
    provider_name: str,
    operation_name: str = "chat",
) -> Generator[Optional["Span"], None, None]:
    """Create a GenAI client span for one LLM call.

    Yields ``None`` when no tracer is configured. Exceptions from the wrapped
    call are recorded on the span and re-raised unchanged.
    """
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(
        f"{operation_name} {model_name}",
        kind=SpanKind.CLIENT,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        with suppress(Exception):
            span.set_attribute(GenAIAttributes.GEN_AI_OPERATION_NAME, operation_name)
            span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_MODEL, model_name)
            span.set_attribute(GenAIAttributes.GEN_AI_PROVIDER_NAME, provider_name)
        try:
            yield span
        except BaseException as exc:
            record_span_error(span, exc)
            raise


__all__ = [
    "llm_call_span",
    "record_span_error",
    "set_llm_call_content",
    "set_llm_request_attributes",
    "set_llm_response_attributes",
]

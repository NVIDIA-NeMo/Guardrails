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

import time
import warnings
from collections.abc import AsyncIterator, Mapping
from contextlib import nullcontext
from typing import Any, Optional, Union

from nemoguardrails.llm.telemetry import (
    llm_call_span,
    set_llm_call_content,
    set_llm_request_attributes,
    set_llm_response_attributes,
)
from nemoguardrails.tracing.constants import (
    OperationNames,
    SystemConstants,
    llm_operation_duration,
    record_time_per_output_chunk,
    record_time_to_first_chunk,
    record_token_usage,
)
from nemoguardrails.types import ChatMessage, LLMModel, LLMResponse, LLMResponseChunk, UsageInfo


class InstrumentedLLMModel:
    """Decorate an ``LLMModel`` with tracing, metrics, and content capture.

    Requests and responses are delegated to the wrapped model without changing
    their runtime contract. Wrapping an existing ``InstrumentedLLMModel`` is
    idempotent and returns the existing decorator.

    The tracer controls client-span creation, while metrics and content capture
    are independently opt-in. ``default_request_params`` contributes provider
    defaults to telemetry without changing the parameters sent to the model.
    """

    def __new__(cls, model: LLMModel, *args: Any, **kwargs: Any):
        if isinstance(model, cls):
            return model
        return super().__new__(cls)

    def __init__(
        self,
        model: LLMModel,
        *,
        tracer: Optional[Any] = None,
        metrics_enabled: bool = False,
        content_capture_enabled: bool = False,
        default_request_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if model is self:
            if (
                tracer is not self._tracer
                or metrics_enabled != self._metrics_enabled
                or content_capture_enabled != self._content_capture_enabled
                or dict(default_request_params or {}) != self._default_request_params
            ):
                warnings.warn(
                    "InstrumentedLLMModel is already instrumented; new instrumentation "
                    "settings are ignored. Re-instrument the underlying wrapped_model instead.",
                    stacklevel=2,
                )
            return
        self._model = model
        self._tracer = tracer
        self._metrics_enabled = metrics_enabled
        self._content_capture_enabled = content_capture_enabled
        self._default_request_params = dict(default_request_params or {})

    @property
    def model_name(self) -> str:
        return self._model.model_name

    @property
    def provider_name(self) -> Optional[str]:
        return self._model.provider_name

    @property
    def provider_url(self) -> Optional[str]:
        return self._model.provider_url

    @property
    def api_key(self) -> Optional[str]:
        """The wrapped model's bearer token/API key, if it exposes one.

        Raises ``AttributeError`` for wrapped models without one, so
        ``hasattr(rails.llm, "api_key")`` still reports support correctly
        through this decorator.
        """
        return getattr(self._model, "api_key")

    @api_key.setter
    def api_key(self, value: Optional[str]) -> None:
        setattr(self._model, "api_key", value)

    @property
    def wrapped_model(self) -> LLMModel:
        """Return the underlying model for direct access or re-instrumentation."""
        return self._model

    def _request_params(self, stop: Optional[list[str]], kwargs: dict[str, Any]) -> dict[str, Any]:
        params = {**self._default_request_params, **kwargs}
        if stop is not None:
            params["stop"] = stop
        return params

    @staticmethod
    def _input_messages(prompt: Union[str, list[ChatMessage]]) -> list[dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return [message.to_dict() if isinstance(message, ChatMessage) else message for message in prompt]

    async def generate_async(
        self,
        prompt: Union[str, list[ChatMessage]],
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate one response while recording the LLM call telemetry.

        The prompt, stop sequences, and provider-specific keyword arguments are
        forwarded unchanged to the wrapped model.
        """
        operation_name = OperationNames.CHAT
        provider_name = self.provider_name or SystemConstants.UNKNOWN
        params = self._request_params(stop, kwargs)
        with llm_call_span(self._tracer, self.model_name, provider_name, operation_name) as span:
            set_llm_request_attributes(span, params)
            duration = (
                llm_operation_duration(self.model_name, provider_name, operation_name)
                if self._metrics_enabled
                else nullcontext()
            )
            with duration:
                response = await self._model.generate_async(prompt, stop=stop, **kwargs)
            set_llm_response_attributes(
                span,
                model=response.model,
                response_id=response.request_id,
                finish_reason=response.finish_reason,
                usage=response.usage,
            )
            if self._content_capture_enabled:
                set_llm_call_content(span, self._input_messages(prompt), response.content)
        if self._metrics_enabled:
            record_token_usage(self.model_name, provider_name, operation_name, response.usage)
        return response

    async def stream_async(
        self,
        prompt: Union[str, list[ChatMessage]],
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponseChunk]:
        """Stream response chunks while recording one LLM call lifecycle.

        Metrics and response attributes are finalized when the stream completes
        or closes, and the wrapped stream is always closed when it supports
        ``aclose``.
        """
        operation_name = OperationNames.CHAT
        provider_name = self.provider_name or SystemConstants.UNKNOWN
        params = self._request_params(stop, kwargs)
        captured_usage: Optional[UsageInfo] = None
        captured_model: Optional[str] = None
        captured_response_id: Optional[str] = None
        captured_finish_reason: Optional[str] = None
        content_parts: list[str] = []
        stream = self._model.stream_async(prompt, stop=stop, **kwargs)
        with llm_call_span(self._tracer, self.model_name, provider_name, operation_name) as span:
            set_llm_request_attributes(span, params, stream=True)
            duration = (
                llm_operation_duration(self.model_name, provider_name, operation_name)
                if self._metrics_enabled
                else nullcontext()
            )
            try:
                with duration:
                    started_at = time.monotonic() if self._metrics_enabled else 0.0
                    last_chunk_at: Optional[float] = None
                    async for chunk in stream:
                        if self._metrics_enabled and (chunk.delta_content or chunk.delta_reasoning):
                            now = time.monotonic()
                            if last_chunk_at is None:
                                record_time_to_first_chunk(
                                    self.model_name,
                                    provider_name,
                                    operation_name,
                                    now - started_at,
                                )
                            else:
                                record_time_per_output_chunk(
                                    self.model_name,
                                    provider_name,
                                    operation_name,
                                    now - last_chunk_at,
                                )
                            last_chunk_at = now
                        if chunk.model is not None:
                            captured_model = chunk.model
                        if chunk.request_id is not None:
                            captured_response_id = chunk.request_id
                        if chunk.finish_reason is not None:
                            captured_finish_reason = chunk.finish_reason
                        if chunk.usage is not None:
                            captured_usage = chunk.usage
                        if self._content_capture_enabled and chunk.delta_content:
                            content_parts.append(chunk.delta_content)
                        yield chunk
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()
            set_llm_response_attributes(
                span,
                model=captured_model,
                response_id=captured_response_id,
                finish_reason=captured_finish_reason,
                usage=captured_usage,
            )
            if self._content_capture_enabled:
                output_text = "".join(content_parts) if content_parts else None
                set_llm_call_content(span, self._input_messages(prompt), output_text)
        if self._metrics_enabled:
            record_token_usage(self.model_name, provider_name, operation_name, captured_usage)


def instrument_llm_model(
    model: LLMModel,
    *,
    tracer: Optional[Any] = None,
    metrics_enabled: bool = False,
    content_capture_enabled: bool = False,
    default_request_params: Optional[Mapping[str, Any]] = None,
) -> LLMModel:
    """Return ``model`` decorated with the requested telemetry.

    The original model is returned when neither tracing nor metrics are enabled.
    Existing ``InstrumentedLLMModel`` instances are returned unchanged.
    """
    if isinstance(model, InstrumentedLLMModel):
        return model
    if tracer is None and not metrics_enabled:
        return model
    return InstrumentedLLMModel(
        model,
        tracer=tracer,
        metrics_enabled=metrics_enabled,
        content_capture_enabled=content_capture_enabled,
        default_request_params=default_request_params,
    )


__all__ = ["InstrumentedLLMModel", "instrument_llm_model"]

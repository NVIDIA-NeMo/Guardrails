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

"""Engine registry for IORails engine.

Manages a collection of ModelEngine and APIEngine instances, one per configured
model type. Each engine owns its own RetryClient with per-model settings.
"""

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Optional, TypeVar

from nemoguardrails.guardrails.api_engine import APIEngine
from nemoguardrails.guardrails.base_engine import BaseEngine
from nemoguardrails.guardrails.guardrails_types import get_request_id, truncate
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.guardrails.telemetry import (
    api_call_span,
    llm_call_span,
    set_llm_call_content,
)
from nemoguardrails.rails.llm.config import Model, RailsConfigData
from nemoguardrails.tracing.constants import (
    llm_operation_duration,
    record_time_per_output_chunk,
    record_time_to_first_chunk,
    record_token_usage,
)
from nemoguardrails.types import LLMResponse, LLMResponseChunk, UsageInfo

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

log = logging.getLogger(__name__)

_EngineT = TypeVar("_EngineT", bound=BaseEngine)


class EngineRegistry:
    """Registry of ModelEngine and APIEngine instances for IORails.

    Creates one engine per configured model or API service, keyed by name.
    Each engine owns its own HTTP client with per-model retry and timeout settings.
    """

    def __init__(
        self,
        models: list[Model],
        rails_config_data: RailsConfigData,
        tracer: Optional["Tracer"] = None,
        metrics_enabled: bool = False,
        content_capture_enabled: bool = False,
    ) -> None:
        """Build one engine per configured model and API service.

        When *tracer* is provided, LLM and API calls produce OTEL spans; when
        ``None`` the span helpers become no-ops.

        When *metrics_enabled* is True, LLM calls emit the OTEL GenAI
        client-side metrics (``gen_ai.client.token.usage``,
        ``gen_ai.client.operation.duration``, plus the streaming
        chunk-timing metrics).  Defaults to False so callers that don't
        opt in get no metric emissions even if a MeterProvider is
        configured globally.

        When *content_capture_enabled* is True, LLM call spans carry
        input/output message content per the OTEL GenAI content-capture
        contract.  Defaults to False; should only be True when
        ``tracer`` is also set, since capture on a no-op span is wasted
        work.
        """
        self._engines: dict[str, BaseEngine] = {}
        self._running = False
        self._tracer = tracer
        self._metrics_enabled = metrics_enabled
        self._content_capture_enabled = content_capture_enabled

        for model_config in models:
            engine = ModelEngine(model_config)
            self._engines[model_config.type] = engine
            log.info(
                "Registered model engine: type=%s, model=%s, base_url=%s",
                model_config.type,
                model_config.model,
                engine.base_url,
            )

        jailbreak_config = rails_config_data.jailbreak_detection
        if jailbreak_config and jailbreak_config.nim_base_url:
            if "jailbreak_detection" in self._engines:
                raise ValueError(
                    "Engine name 'jailbreak_detection' is already registered as a model engine. "
                    "Cannot register the jailbreak detection API engine with the same name."
                )
            api_engine = APIEngine.from_jailbreak_config(jailbreak_config)
            self._engines["jailbreak_detection"] = api_engine
            log.info(
                "Registered API engine: name=%s, url=%s",
                "jailbreak_detection",
                api_engine.url,
            )

    async def start(self) -> None:
        """Start all engine clients. Call this during service startup."""
        if self._running:
            return

        started: list[BaseEngine] = []

        for name, engine in self._engines.items():
            try:
                await engine.start()
                started.append(engine)
            except Exception as e:
                log.error("Error starting engine %s: %s", name, e)
                for eng in started:
                    try:
                        await eng.stop()
                    except Exception:
                        pass
                raise RuntimeError(f"Failed to start engine: Engine {name}: exception {e}") from e

        self._running = True

    async def stop(self) -> None:
        """Stop all engine clients. Call this during service shutdown."""
        if not self._running:
            return

        engine_errors: dict[str, Exception] = {}
        try:
            for name, engine in self._engines.items():
                try:
                    await engine.stop()
                except Exception as e:
                    engine_errors[name] = e
                    log.error("Error stopping engine %s: %s", name, e)
        finally:
            self._running = False

        if engine_errors:
            engine_error_string = ", ".join(
                f"Engine {name}: exception {exception}" for name, exception in engine_errors.items()
            )
            raise RuntimeError(f"Failed to stop engines: {engine_error_string}")

    def _get_engine(self, name: str, expected_type: type[_EngineT]) -> _EngineT:
        """Look up an engine by name, verifying its type."""
        if name not in self._engines:
            available = list(self._engines.keys())
            raise KeyError(f"No engine configured with name '{name}'. Available: {available}")
        engine = self._engines[name]
        if not isinstance(engine, expected_type):
            raise TypeError(f"Engine '{name}' is {type(engine).__name__}, expected {expected_type.__name__}")
        return engine

    async def model_call(self, model_type: str, messages: list[dict], **kwargs: Any) -> LLMResponse:
        """Route a chat completion request to the named model engine.

        Returns the structured ``LLMResponse`` from the engine — content,
        reasoning (when the provider exposes it), usage, finish reason.
        Callers that only want the assistant text should access ``.content``.

        When metrics are enabled, emits ``gen_ai.client.operation.duration``
        (with ``error.type`` on exception) and ``gen_ai.client.token.usage``
        (one observation each for ``input`` and ``output`` token types,
        only when ``LLMResponse.usage`` is populated).

        Raises:
            KeyError: If no engine is registered with the given name.
            TypeError: If the named engine is not a ModelEngine.
        """
        req_id = get_request_id()
        log.debug("[%s] Model engine '%s' messages: %s", req_id, model_type, truncate(messages))

        engine = self._get_engine(model_type, ModelEngine)
        # TODO: Replace with LLMModel.provider_name after refactoring
        provider_name = engine.model_config.engine or "unknown"
        operation_name = "chat"

        # Compose: span (always created — no-op when tracer is None) and
        # duration metric (only when metrics enabled).  Token usage is
        # emitted after the call returns since it depends on
        # ``result.usage`` — exception path skips it because control
        # never reaches the line below.
        duration_ctx = (
            llm_operation_duration(engine.model_name, provider_name, operation_name)
            if self._metrics_enabled
            else nullcontext()
        )
        with llm_call_span(self._tracer, engine.model_name, provider_name, operation_name) as span:
            with duration_ctx:
                result = await engine.chat_completion(messages, **kwargs)
            # Capture content inside the span context so the helper sees
            # the live LLM CLIENT span (not None even on the success path)
            # and the attributes/events land before the span closes.
            if self._content_capture_enabled:
                set_llm_call_content(span, messages, result.content)

        if self._metrics_enabled:
            record_token_usage(engine.model_name, provider_name, operation_name, result.usage)

        log.debug("[%s] Model engine '%s' response: %s", req_id, model_type, truncate(result))
        return result

    async def stream_model_call(
        self, model_type: str, messages: list[dict], **kwargs: Any
    ) -> AsyncGenerator[LLMResponseChunk, None]:
        """Stream chat completion chunks from the named model engine.

        Yields ``LLMResponseChunk`` objects. The surrounding
        ``llm_call_span`` wraps the full generator lifetime: it opens
        before the first chunk and closes when the generator exhausts or
        raises.

        When metrics are enabled, emits ``gen_ai.client.operation.duration``
        for the full stream lifetime (with ``error.type`` on exception)
        and ``gen_ai.client.token.usage`` after stream completion using
        the ``UsageInfo`` carried on the terminal SSE chunk (when the
        provider returns one — controlled by ``include_usage_in_stream``,
        defaults to True for OpenAI-compatible engines).  No token
        observation is emitted on early consumer cancellation or on
        provider error mid-stream.

        Raises:
            KeyError: If no engine is registered with the given name.
            TypeError: If the named engine is not a ModelEngine.
        """
        req_id = get_request_id()
        log.debug("[%s] Model engine '%s' stream messages: %s", req_id, model_type, truncate(messages))

        engine = self._get_engine(model_type, ModelEngine)
        # TODO: Change to LLMModel.provider_name after refactor
        provider_name = engine.model_config.engine or "unknown"
        operation_name = "chat"

        # Capture the most recent chunk's ``usage`` field so we can emit
        # token metrics after the stream completes — providers (e.g.
        # OpenAI-compatible) only populate ``usage`` on the terminal
        # chunk when ``stream_options.include_usage=true``.
        captured_usage: Optional["UsageInfo"] = None
        # Accumulate streamed delta_content here when content capture is on;
        # joined and recorded onto the LLM span at stream end.  The list is
        # allocated unconditionally (cost: one empty list per stream); the
        # per-chunk appends are gated on the flag so the disabled path
        # doesn't carry chunk strings in memory.
        content_parts: list[str] = []
        duration_ctx = (
            llm_operation_duration(engine.model_name, provider_name, operation_name)
            if self._metrics_enabled
            else nullcontext()
        )
        with llm_call_span(self._tracer, engine.model_name, provider_name, operation_name) as span:
            with duration_ctx:
                # Gate timing-state setup on ``_metrics_enabled`` so the
                # cold path skips ``time.monotonic()`` and the per-chunk
                # bookkeeping entirely.  ``t0`` defaults to ``0.0`` in
                # the disabled path so the type stays a plain ``float``
                # — it's never read in that branch.
                t0 = time.monotonic() if self._metrics_enabled else 0.0
                last_chunk_time: Optional[float] = None
                async for chunk in engine.stream_chat_completion(messages, **kwargs):
                    if self._metrics_enabled:
                        # Per OTEL semconv, "first chunk" / "output chunk"
                        # mean content-bearing chunks — gate on
                        # ``delta_content`` / ``delta_reasoning`` to skip
                        # the terminal usage frame and any other cosmetic
                        # SSE events that the parser leaves in place.
                        if chunk.delta_content or chunk.delta_reasoning:
                            now = time.monotonic()
                            if last_chunk_time is None:
                                record_time_to_first_chunk(engine.model_name, provider_name, operation_name, now - t0)
                            else:
                                record_time_per_output_chunk(
                                    engine.model_name, provider_name, operation_name, now - last_chunk_time
                                )
                            last_chunk_time = now
                        if chunk.usage is not None:
                            captured_usage = chunk.usage
                    if self._content_capture_enabled and chunk.delta_content:
                        content_parts.append(chunk.delta_content)
                    yield chunk
            # Capture accumulated stream content inside the span context so
            # the helper sees the live LLM CLIENT span before it closes.
            # Reached only on natural exhaustion — consumer cancellation or
            # provider error raises out of the ``with`` blocks above, in
            # which case partial content is intentionally not recorded.
            # Empty ``content_parts`` -> output_text=None so we don't claim
            # an empty assistant response (matches iorails.py's request-span
            # streaming path).
            if self._content_capture_enabled:
                output_text = "".join(content_parts) if content_parts else None
                set_llm_call_content(span, messages, output_text)

        # Reached only on natural exhaustion (not on consumer cancellation
        # or provider error — those raise out of the ``with`` blocks above).
        if self._metrics_enabled:
            record_token_usage(engine.model_name, provider_name, operation_name, captured_usage)

    async def api_call(self, api_name: str, message: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Route an API request to the named API engine.

        Raises:
            KeyError: If no engine is registered with the given name.
            TypeError: If the named engine is not an APIEngine.
        """
        req_id = get_request_id()
        log.debug("[%s] API engine '%s' request: %s", req_id, api_name, truncate(message))

        with api_call_span(self._tracer, api_name):
            api_engine = self._get_engine(api_name, APIEngine)
            response = await api_engine.call(message, **kwargs)

        log.debug("[%s] API engine '%s' response: %s", req_id, api_name, truncate(response))
        return response

    async def __aenter__(self):
        """Async context manager entry: start all engine clients."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit: stop all engine clients."""
        await self.stop()

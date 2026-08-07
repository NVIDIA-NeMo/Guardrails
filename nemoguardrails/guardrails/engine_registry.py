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
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Optional, TypeVar

from nemoguardrails.guardrails.api_engine import APIEngine
from nemoguardrails.guardrails.base_engine import BaseEngine
from nemoguardrails.guardrails.guardrails_types import get_request_id, truncate
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.guardrails.telemetry import api_call_span
from nemoguardrails.guardrails.tool_schema import ToolExchange, ToolResult, Toolset
from nemoguardrails.http.client import ClosableHTTPClient
from nemoguardrails.http.runtime import create_http_client
from nemoguardrails.rails.llm.config import Model, RailsConfigData
from nemoguardrails.types import LLMModel, LLMResponse, LLMResponseChunk

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

        All three telemetry settings are handed to each ``ModelEngine``,
        which owns the instrumentation so that rails reaching the model
        through ``generate_async`` are instrumented identically to main
        generation reaching it through ``model_call``.
        """
        self._engines: dict[str, BaseEngine] = {}
        self._llms: dict[str, LLMModel] = {}
        self._http_client: Optional[ClosableHTTPClient] = None
        self._running = False
        self._tracer = tracer

        for model_config in models:
            engine = ModelEngine(
                model_config,
                tracer=tracer,
                metrics_enabled=metrics_enabled,
                content_capture_enabled=content_capture_enabled,
            )
            self._engines[model_config.type] = engine
            self._llms[model_config.type] = engine
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

    @property
    def llms(self) -> dict[str, LLMModel]:
        """The configured model engines keyed by ``Model.type``.

        This is the ``Dict[str, LLMModel]`` that library rail actions index by
        model type (``llms["content_safety"]``).  API engines are not LLMs and
        are excluded.
        """
        return self._llms

    @property
    def http_client(self) -> ClosableHTTPClient:
        """The process-wide HTTP client shared by vendor rail actions.

        One managed client replaces the per-request client each vendor action
        would otherwise create and close, so connections are pooled across
        requests.

        Raises:
            RuntimeError: If the registry has not been started.
        """
        if self._http_client is None:
            raise RuntimeError("EngineRegistry has not been started. Call start() first.")
        return self._http_client

    async def _close_http_client(self) -> None:
        """Close the managed HTTP client, if one is open.

        The reference is cleared before the await so a failing ``close()``
        still leaves the registry without a half-closed client.
        """
        client, self._http_client = self._http_client, None
        if client is not None:
            await client.close()

    async def start(self) -> None:
        """Start all engine clients and the managed HTTP client.

        Call this during service startup.  A failure part-way through rolls
        everything already started back, so a failed start leaks nothing.
        """
        if self._running:
            return

        started: list[tuple[str, BaseEngine]] = []
        self._http_client = create_http_client()

        for name, engine in self._engines.items():
            try:
                await engine.start()
                started.append((name, engine))
            except Exception as e:
                log.error("Error starting engine %s: %s", name, e)
                await self._rollback_start(started)
                raise RuntimeError(f"Failed to start engine: Engine {name}: exception {e}") from e

        self._running = True

    async def _rollback_start(self, started: list[tuple[str, BaseEngine]]) -> None:
        """Release everything a partial ``start`` brought up."""
        for name, engine in started:
            try:
                await engine.stop()
            except Exception as stop_error:
                log.warning("Error stopping engine %s during start rollback: %s", name, stop_error)
        try:
            await self._close_http_client()
        except Exception as close_error:
            log.warning("Error closing the managed HTTP client during start rollback: %s", close_error)

    async def stop(self) -> None:
        """Stop all engine clients and the managed HTTP client.

        Call this during service shutdown.  Every component is stopped even if
        an earlier one fails; the failures are reported together afterwards.
        """
        if not self._running:
            return

        errors: dict[str, Exception] = {}
        try:
            for name, engine in self._engines.items():
                try:
                    await engine.stop()
                except Exception as e:
                    errors[f"Engine {name}"] = e
                    log.error("Error stopping engine %s: %s", name, e)
            try:
                await self._close_http_client()
            except Exception as e:
                errors["HTTP client"] = e
                log.error("Error closing the managed HTTP client: %s", e)
        finally:
            self._running = False

        if errors:
            error_string = ", ".join(f"{component}: exception {exception}" for component, exception in errors.items())
            raise RuntimeError(f"Failed to stop engines: {error_string}")

    def _get_engine(self, name: str, expected_type: type[_EngineT]) -> _EngineT:
        """Look up an engine by name, verifying its type."""
        if name not in self._engines:
            available = list(self._engines.keys())
            raise KeyError(f"No engine configured with name '{name}'. Available: {available}")
        engine = self._engines[name]
        if not isinstance(engine, expected_type):
            raise TypeError(f"Engine '{name}' is {type(engine).__name__}, expected {expected_type.__name__}")
        return engine

    def provider_name(self, model_type: str) -> str:
        """Return the provider/engine name (e.g. 'nim', 'openai') for a model engine."""
        return self._get_engine(model_type, ModelEngine).provider_name

    async def model_call(self, model_type: str, messages: list[dict], **kwargs: Any) -> LLMResponse:
        """Route a chat completion request to the named model engine.

        Returns the structured ``LLMResponse`` from the engine — content,
        reasoning (when the provider exposes it), usage, finish reason.
        Callers that only want the assistant text should access ``.content``.

        Parameter merging and OTEL instrumentation live in
        ``ModelEngine.generate_from_messages`` so that rails, which reach the
        model through ``llm_call`` rather than through this method, emit the
        same spans and metrics.  *messages* is already in wire form here — every
        IORails entry point normalizes through ``IORails._convert_to_messages``
        — so this skips the ``generate_async`` protocol adapter.

        Raises:
            KeyError: If no engine is registered with the given name.
            TypeError: If the named engine is not a ModelEngine.
        """
        req_id = get_request_id()
        log.debug("[%s] Model engine '%s' messages: %s", req_id, model_type, truncate(messages))

        engine = self._get_engine(model_type, ModelEngine)
        result = await engine.generate_from_messages(messages, **kwargs)

        log.debug("[%s] Model engine '%s' response: %s", req_id, model_type, truncate(result))
        return result

    async def stream_model_call(
        self, model_type: str, messages: list[dict], **kwargs: Any
    ) -> AsyncGenerator[LLMResponseChunk, None]:
        """Stream chat completion chunks from the named model engine.

        Yields ``LLMResponseChunk`` objects.  Parameter merging, the LLM
        CLIENT span, and the metrics live in
        ``ModelEngine.stream_from_messages`` — see that method for the span and
        metric contract.  As in ``model_call``, *messages* is already in wire
        form, so this skips the ``stream_async`` protocol adapter.

        Raises:
            KeyError: If no engine is registered with the given name.
            TypeError: If the named engine is not a ModelEngine.
        """
        req_id = get_request_id()
        log.debug("[%s] Model engine '%s' stream messages: %s", req_id, model_type, truncate(messages))

        engine = self._get_engine(model_type, ModelEngine)
        stream = engine.stream_from_messages(messages, **kwargs)
        try:
            async for chunk in stream:
                yield chunk
        finally:
            # Close the delegate explicitly.  A consumer that abandons this
            # generator only unwinds *this* one; without the aclose the
            # instrumented core stays suspended at its yield until garbage
            # collection, so the duration metric's `finally` never runs.
            await stream.aclose()

    def parse_tools(self, model_type: str, llm_params: Optional[dict]) -> Toolset:
        """Parse the tool block in ``llm_params`` for the named model engine.

        Delegates to the engine's ``parse_tools`` so the provider-specific shape
        (keyed on the engine) is normalized into a ``Toolset`` for the tool rails.

        Raises:
            KeyError: If no engine is registered with the given name.
            TypeError: If the named engine is not a ModelEngine.
        """
        engine = self._get_engine(model_type, ModelEngine)
        return engine.parse_tools({**engine.body_param_defaults, **(llm_params or {})})

    def extract_tool_results(self, model_type: str, messages: list[dict]) -> list[ToolResult]:
        """Extract incoming tool results from ``messages`` for the named model engine.

        Delegates to the engine's ``extract_tool_results`` so the provider's
        tool-result messages are normalized into the ``ToolResult`` list the
        ToolResultRail consumes.

        Raises:
            KeyError: If no engine is registered with the given name.
            TypeError: If the named engine is not a ModelEngine.
        """
        engine = self._get_engine(model_type, ModelEngine)
        return engine.extract_tool_results(messages)

    def extract_tool_exchanges(self, model_type: str, messages: list[dict]) -> list[ToolExchange]:
        """Group ``messages`` into per-turn ``(tool_calls, tool_results)`` exchanges.

        Delegates to the engine's ``extract_tool_exchanges`` so each tool result is
        validated against its own turn's calls. This keeps ``call_id`` linkage
        turn-local, which ``RailsManager.are_tool_results_safe`` relies on so that ids
        reused across turns (spec-allowed) are not flagged as ambiguous duplicates.

        Raises:
            KeyError: If no engine is registered with the given name.
            TypeError: If the named engine is not a ModelEngine.
        """
        engine = self._get_engine(model_type, ModelEngine)
        return engine.extract_tool_exchanges(messages)

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

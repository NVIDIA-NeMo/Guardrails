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
from nemoguardrails.guardrails.telemetry import api_call_span, llm_call_span
from nemoguardrails.rails.llm.config import Model, RailsConfigData

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
    ) -> None:
        """Build one engine per configured model and API service.

        When *tracer* is provided, LLM and API calls produce OTEL spans; when
        ``None`` the span helpers become no-ops.
        """
        self._engines: dict[str, BaseEngine] = {}
        self._running = False
        self._tracer = tracer

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

    async def model_call(self, model_type: str, messages: list[dict], **kwargs: Any) -> str:
        """Route a chat completion request to the named model engine.

        Raises:
            KeyError: If no engine is registered with the given name.
            TypeError: If the named engine is not a ModelEngine.
        """
        req_id = get_request_id()
        log.debug("[%s] Model engine '%s' messages: %s", req_id, model_type, truncate(messages))

        engine = self._get_engine(model_type, ModelEngine)
        with llm_call_span(self._tracer, engine.model_name, engine.model_config.engine or "unknown"):
            result = await engine.chat_completion(messages, **kwargs)

        log.debug("[%s] Model engine '%s' response: %s", req_id, model_type, truncate(result))
        return result

    async def stream_model_call(
        self, model_type: str, messages: list[dict], **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion chunks from the named model engine.

        Raises:
            KeyError: If no engine is registered with the given name.
            TypeError: If the named engine is not a ModelEngine.
        """
        # TODO Streaming instrumentation handled in follow-on PR

        req_id = get_request_id()
        log.debug("[%s] Model engine '%s' stream messages: %s", req_id, model_type, truncate(messages))

        engine = self._get_engine(model_type, ModelEngine)
        async for chunk in engine.stream_chat_completion(messages, **kwargs):
            yield chunk

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

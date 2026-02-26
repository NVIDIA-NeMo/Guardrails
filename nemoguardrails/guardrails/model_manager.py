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

"""Model manager for IORails engine.

Manages a collection of ModelEngine instances, one per configured model type.
Each ModelEngine owns its own RetryClient with per-model settings.
"""

import logging
from typing import Any

from nemoguardrails.guardrails.api_engine import APIEngine
from nemoguardrails.guardrails.guardrails_types import get_request_id, truncate
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.rails.llm.config import RailsConfig

log = logging.getLogger(__name__)


class ModelManager:
    """Manages ModelEngine instances for IORails.

    Creates one ModelEngine per configured model, keyed by model type
    (e.g. "main", "content_safety", "jailbreak_detection").
    Each engine owns its own HTTP client with per-model retry and timeout settings.
    """

    def __init__(self, config: RailsConfig) -> None:
        self._engines: dict[str, ModelEngine] = {}
        self._api_engines: dict[str, APIEngine] = {}
        self._running = False

        for model_config in config.models:
            self._engines[model_config.type] = ModelEngine(model_config)
            log.info(
                "Registered model engine: type=%s, model=%s, base_url=%s",
                model_config.type,
                model_config.model,
                self._engines[model_config.type].base_url,
            )

        self._init_jailbreak_detection_engine(config)

    def _init_jailbreak_detection_engine(self, config: RailsConfig) -> None:
        """Initialize APIEngine instances from rails configuration."""

        jailbreak_config = config.rails.config.jailbreak_detection
        if jailbreak_config and jailbreak_config.nim_base_url:
            self._api_engines["jailbreak_detection"] = APIEngine.from_jailbreak_config(jailbreak_config)
            log.info(
                "Registered API engine: name=%s, url=%s",
                "jailbreak_detection",
                self._api_engines["jailbreak_detection"].url,
            )

    async def start(self) -> None:
        """Start all engine clients. Call this during service startup."""
        if self._running:
            return

        started = []
        engine_errors = {}

        for engine_type, engine in self._engines.items():
            try:
                await engine.start()
                started.append(engine)
            except Exception as e:
                engine_errors[engine_type] = e
                log.error("Error starting model engine type %s: %s", engine_type, e)

        for api_name, api_engine in self._api_engines.items():
            try:
                await api_engine.start()
                started.append(api_engine)
            except Exception as e:
                engine_errors[api_name] = e
                log.error("Error starting API engine %s: %s", api_name, e)

        if engine_errors:
            # Roll back engines that started successfully to avoid leaked clients
            for engine in started:
                try:
                    await engine.stop()
                except Exception:
                    pass
            engine_error_string = ", ".join(
                f"Engine {name}: exception {exception}" for name, exception in engine_errors.items()
            )
            raise RuntimeError(f"Failed to start engines: {engine_error_string}")

        self._running = True

    async def stop(self) -> None:
        """Stop all engine clients. Call this during service shutdown."""
        if not self._running:
            return

        engine_errors = {}
        try:
            for engine_type, engine in self._engines.items():
                try:
                    await engine.stop()
                except Exception as e:
                    engine_errors[engine_type] = e
                    log.error("Error stopping model engine type %s: %s", engine_type, e)

            for api_name, api_engine in self._api_engines.items():
                try:
                    await api_engine.stop()
                except Exception as e:
                    engine_errors[api_name] = e
                    log.error("Error stopping API engine %s: %s", api_name, e)
        finally:
            self._running = False

        if engine_errors:
            engine_error_string = ", ".join(
                f"Engine {name}: exception {exception}" for name, exception in engine_errors.items()
            )
            raise RuntimeError(f"Failed to stop engines: {engine_error_string}")

    def _get_model_engine(self, model_type: str) -> ModelEngine:
        """Look up a ModelEngine by its model type."""
        if model_type not in self._engines:
            available = list(self._engines.keys())
            raise KeyError(f"No model configured with type '{model_type}'. Available types: {available}")
        return self._engines[model_type]

    def _get_api_engine(self, api_name: str) -> APIEngine:
        """Look up an APIEngine by its name."""
        if api_name not in self._api_engines:
            available = list(self._api_engines.keys())
            raise KeyError(f"No API engine configured with name '{api_name}'. Available: {available}")
        return self._api_engines[api_name]

    async def generate_async(self, model_type: str, messages: list[dict], **kwargs: Any) -> str:
        """Generate a chat completion response from the named model engine."""
        req_id = get_request_id()
        log.debug("[%s] Model engine '%s' messages: %s", req_id, model_type, truncate(messages))

        engine = self._get_model_engine(model_type)
        response = await engine.call(messages, **kwargs)
        result = response["choices"][0]["message"]["content"]

        log.debug("[%s] Model engine '%s' response: %s", req_id, model_type, truncate(result))
        return result

    async def api_call(self, api_name: str, message: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        req_id = get_request_id()
        log.debug("[%s] API engine '%s' request: %s", req_id, api_name, truncate(message))
        api_engine = self._get_api_engine(api_name)
        response = await api_engine.call(message, **kwargs)

        log.debug("[%s] API engine '%s' response: %s", req_id, api_name, truncate(response))
        return response

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

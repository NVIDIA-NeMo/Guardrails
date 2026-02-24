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

from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.rails.llm.config import Model

log = logging.getLogger(__name__)


class ModelManager:
    """Manages ModelEngine instances for IORails.

    Creates one ModelEngine per configured model, keyed by model type
    (e.g. "main", "content_safety", "jailbreak_detection").
    Each engine owns its own HTTP client with per-model retry and timeout settings.
    """

    def __init__(self, models: list[Model]) -> None:
        self._engines: dict[str, ModelEngine] = {}

        for model_config in models:
            self._engines[model_config.type] = ModelEngine(model_config)
            log.info(
                "Registered model engine: type=%s, model=%s, base_url=%s",
                model_config.type,
                model_config.model,
                self._engines[model_config.type].base_url,
            )

    async def start(self) -> None:
        """Start all model engine clients."""
        for engine in self._engines.values():
            await engine.start()

    async def stop(self) -> None:
        """Stop all model engine clients."""
        for engine in self._engines.values():
            await engine.stop()

    def get_engine(self, model_type: str) -> ModelEngine:
        """Look up a ModelEngine by its model type.

        Raises:
            KeyError: If no model with the given type is configured.
        """
        if model_type not in self._engines:
            available = list(self._engines.keys())
            raise KeyError(f"No model configured with type '{model_type}'. Available types: {available}")
        return self._engines[model_type]

    async def generate_async(self, model_type: str, messages: list[dict], **kwargs: Any) -> str:
        """Generate a response from the model of the given type.

        Args:
            model_type: The model type key (e.g. "main", "content_safety").
            messages: List of message dicts in OpenAI format.
            **kwargs: Additional LLM parameters.

        Returns:
            The content string from the model's response.
        """
        engine = self.get_engine(model_type)
        response = await engine.call(messages, **kwargs)
        return response["choices"][0]["message"]["content"]

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

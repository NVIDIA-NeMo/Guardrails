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

"""IORails bindings for shared model resources and invocation.

This adapter owns no provider transport or parsing implementation. Its lifecycle
releases only the private client pool owned by this runtime.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Optional

from pydantic import SecretStr

from nemoguardrails.guardrails.tool_parsing import (
    parse_tools,
)
from nemoguardrails.llm.call import llm_call
from nemoguardrails.llm.constants import AZURE_PROVIDERS
from nemoguardrails.llm.frameworks import get_default_framework, get_framework
from nemoguardrails.llm.frameworks.default import DefaultFramework
from nemoguardrails.llm.models.instrumented import InstrumentedLLMModel
from nemoguardrails.llm.models.resources import build_model_resources, prepare_model_kwargs, resolve_model_resources
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.types import LLMModel, LLMResponse, LLMResponseChunk

log = logging.getLogger(__name__)


def _canonical_model_config(model: Model) -> Model:
    """Translate the established IORails transport settings on a private copy."""
    params = dict(model.parameters or {})
    params.pop("stream", None)
    params.pop("stream_options", None)
    api_key = params.get("api_key")
    if isinstance(api_key, SecretStr):
        params["api_key"] = api_key.get_secret_value()
    if "timeout_connect" in params:
        value = params.pop("timeout_connect")
        if "connect_timeout" in params and params["connect_timeout"] != value:
            raise ValueError("Specify only one of timeout_connect and connect_timeout.")
        params["connect_timeout"] = value
    if "max_attempts" in params:
        attempts = int(params.pop("max_attempts"))
        if attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        retries = attempts - 1
        if "max_retries" in params and params["max_retries"] != retries:
            raise ValueError("max_attempts and max_retries specify different retry counts.")
        params["max_retries"] = retries
    if params.get("default_headers") is not None:
        params["default_headers"] = {str(key): str(value) for key, value in params["default_headers"].items()}
    query = params.get("default_query")
    if query is not None:
        if not isinstance(query, Mapping):
            raise ValueError("default_query must be a mapping of parameter names to values.")
        for value in query.values():
            values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else (value,)
            if any(isinstance(element, Mapping) for element in values):
                raise ValueError("default_query values must be scalars or lists of scalars, not nested objects.")
    if model.engine not in AZURE_PROVIDERS and params.get("base_url"):
        base_url = params["base_url"].rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        params["base_url"] = base_url
    return model.model_copy(update={"parameters": params})


def _default_parameters(model: LLMModel) -> dict[str, Any]:
    defaults = getattr(model, "default_parameters", {})
    return dict(defaults) if isinstance(defaults, Mapping) else {}


class ModelRuntime:
    """Bind resource models, model instrumentation, and canonical tool contracts."""

    def __init__(
        self,
        models: Sequence[Model],
        *,
        main_model: Optional[LLMModel] = None,
        tracer: Any = None,
        metrics_enabled: bool = False,
        content_capture_enabled: bool = False,
    ) -> None:
        # Resolve overrides before normalizing parameters on ignored main entries.
        if main_model is not None and any(model.type == "main" for model in models):
            log.warning(
                "Both an LLM was provided via constructor and a main LLM is specified in the config. "
                "The LLM provided via constructor will be used and the main LLM from config will be ignored."
            )
        selected = resolve_model_resources(models, main_model=main_model)
        configs = ([selected.main_config] if selected.main_config is not None else []) + list(
            selected.specialized_configs
        )
        framework = get_framework(get_default_framework())

        def normalize(model: Model) -> Model:
            if isinstance(framework, DefaultFramework) and framework.uses_builtin_provider(model.engine):
                normalized = _canonical_model_config(model)
                framework.prepare_builtin_parameters(model.engine, prepare_model_kwargs(normalized))
                return normalized
            return model

        plan = replace(
            selected,
            main_config=normalize(selected.main_config) if selected.main_config else None,
            specialized_configs=tuple(normalize(model) for model in selected.specialized_configs),
        )
        self.resources = build_model_resources(
            plan,
            managed=True,
            client_defaults={"timeout": 30.0, "connect_timeout": 5.0, "max_retries": 2},
        )
        self.main_model = self.resources.main_model
        self.model_caches = self.resources.caches
        all_models = dict(self.resources.specialized_models)
        if self.main_model is not None:
            all_models["main"] = self.main_model
        self.llms = {
            name: model
            if isinstance(model, InstrumentedLLMModel)
            else InstrumentedLLMModel(
                model,
                tracer=tracer,
                metrics_enabled=metrics_enabled,
                content_capture_enabled=content_capture_enabled,
                default_request_params=_default_parameters(model),
            )
            for name, model in all_models.items()
        }
        self._tool_defaults = {}
        for config in configs:
            self._tool_defaults.setdefault(config.type, {"tools": (config.parameters or {}).get("tools")})
        self._lifecycle_lock = asyncio.Lock()

    async def start(self) -> None:
        """Finish pending cleanup before a runtime is reused."""
        async with self._lifecycle_lock:
            await self.resources.start()

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            await self.resources.aclose()

    def provider_name(self, model_type: str) -> str:
        return self.llms[model_type].provider_name or "unknown"

    async def model_call(self, model_type: str, messages: list[dict], **kwargs: Any) -> LLMResponse:
        return await llm_call(self.llms[model_type], messages, llm_params=kwargs)

    async def stream_model_call(
        self,
        model_type: str,
        messages: list[dict],
        *,
        chunk_consumer: Callable[[LLMResponseChunk], Awaitable[None]],
        **kwargs: Any,
    ) -> LLMResponse:
        return await llm_call(self.llms[model_type], messages, llm_params=kwargs, chunk_consumer=chunk_consumer)

    def parse_tools(self, model_type: str, llm_params: Optional[dict]):
        model = self.llms[model_type]
        return parse_tools(
            {**self._tool_defaults.get(model_type, {}), **_default_parameters(model), **(llm_params or {})},
        )

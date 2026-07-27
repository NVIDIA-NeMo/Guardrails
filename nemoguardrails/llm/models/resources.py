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

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from nemoguardrails._compat.langchain_kwargs import check_langchain_kwargs
from nemoguardrails.exceptions import InvalidModelConfigurationError
from nemoguardrails.llm.cache import CacheInterface, LFUCache
from nemoguardrails.llm.frameworks import get_default_framework
from nemoguardrails.llm.models.initializer import ModelInitializationError, init_llm_model
from nemoguardrails.types import LLMModel

if TYPE_CHECKING:
    from nemoguardrails.rails.llm.config import Model

log = logging.getLogger(__name__)

ModelInitializer = Callable[..., LLMModel]
ModelDecorator = Callable[[LLMModel, Mapping[str, Any]], LLMModel]


@dataclass(frozen=True, slots=True)
class LLMModelResources:
    main: LLMModel | None
    by_type: Mapping[str, LLMModel]
    caches: Mapping[str, CacheInterface]

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_type", MappingProxyType(dict(self.by_type)))
        object.__setattr__(self, "caches", MappingProxyType(dict(self.caches)))

    def with_main(self, main: LLMModel) -> LLMModelResources:
        return replace(self, main=main)

    def action_parameters(self) -> dict[str, Any]:
        parameters: dict[str, Any] = {"llms": dict(self.by_type)}
        if self.main is not None:
            parameters["llm"] = self.main
        for model_type, model in self.by_type.items():
            parameters[f"{model_type}_llm"] = model
        if self.caches:
            parameters["model_caches"] = dict(self.caches)
        return parameters


def prepare_model_kwargs(model_config: Model) -> dict[str, Any]:
    kwargs = dict(model_config.parameters or {})
    if model_config.api_key_env_var:
        api_key = os.environ.get(model_config.api_key_env_var)
        if api_key:
            kwargs["api_key"] = api_key
    return kwargs


def _create_model_cache(model_config: Model) -> LFUCache:
    cache_config = model_config.cache
    if cache_config is None:
        raise ValueError(f"Model '{model_config.type}' does not define a cache")
    if cache_config.maxsize <= 0:
        raise ValueError(
            f"Invalid cache maxsize for model '{model_config.type}': {cache_config.maxsize}. "
            "Capacity must be greater than 0. Skipping cache creation."
        )

    stats_logging_interval = None
    if cache_config.stats.enabled and cache_config.stats.log_interval is not None:
        stats_logging_interval = cache_config.stats.log_interval

    cache = LFUCache(
        maxsize=cache_config.maxsize,
        track_stats=cache_config.stats.enabled,
        stats_logging_interval=stats_logging_interval,
    )
    log.info("Created cache for model '%s' with maxsize %s", model_config.type, cache_config.maxsize)
    return cache


def _initialize_model(
    model_config: Model,
    initializer: ModelInitializer,
    decorator: ModelDecorator | None,
    *,
    mode: str | None = None,
) -> LLMModel:
    model_name = model_config.model
    if not model_name:
        raise InvalidModelConfigurationError(
            f"`model` field must be set in model configuration: {model_config.model_dump_json()}"
        )

    kwargs = prepare_model_kwargs(model_config)
    model = initializer(
        model_name=model_name,
        provider_name=model_config.engine,
        mode=mode or model_config.mode,
        kwargs=kwargs,
    )
    return decorator(model, kwargs) if decorator is not None else model


def build_llm_model_resources(
    models: Sequence[Model],
    *,
    main: LLMModel | None = None,
    initializer: ModelInitializer = init_llm_model,
    decorator: ModelDecorator | None = None,
) -> LLMModelResources:
    configured_models = list(models)
    models_to_check = [model for model in configured_models if model.type != "main"] if main else configured_models
    check_langchain_kwargs(models_to_check, get_default_framework())

    resolved_main = decorator(main, {}) if main is not None and decorator is not None else main
    configured_main = next((model for model in configured_models if model.type == "main"), None)
    if resolved_main is not None:
        if configured_main is not None:
            log.warning(
                "Both an LLM was provided via constructor and a main LLM is specified in the config. "
                "The LLM provided via constructor will be used and the main LLM from config will be ignored."
            )
    elif configured_main is not None:
        try:
            resolved_main = _initialize_model(configured_main, initializer, decorator, mode="chat")
        except ModelInitializationError as error:
            log.error("Failed to initialize model: %s", error)
            raise
        except Exception as error:
            log.error("Unexpected error initializing model: %s", error)
            raise
    else:
        log.info("No main LLM specified in the config and no LLM provided via constructor.")

    by_type: dict[str, LLMModel] = {}
    for model_config in configured_models:
        if model_config.type in {"main", "embeddings", "jailbreak_detection"}:
            continue
        try:
            model = _initialize_model(model_config, initializer, decorator)
            by_type.setdefault(model_config.type, model)
        except ModelInitializationError as error:
            log.error("Failed to initialize model: %s", error)
            raise
        except Exception as error:
            log.error("Unexpected error initializing model: %s", error)
            raise

    caches: dict[str, CacheInterface] = {}
    for model_config in configured_models:
        if model_config.type in {"main", "embeddings"}:
            continue
        if model_config.cache and model_config.cache.enabled:
            cache = _create_model_cache(model_config)
            caches[model_config.type] = cache
            log.info("Initialized model '%s' with cache enabled", model_config.type)

    return LLMModelResources(main=resolved_main, by_type=by_type, caches=caches)


__all__ = [
    "LLMModelResources",
    "ModelDecorator",
    "ModelInitializer",
    "build_llm_model_resources",
    "prepare_model_kwargs",
]

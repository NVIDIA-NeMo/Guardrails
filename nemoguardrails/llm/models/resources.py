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

"""Shared model selection and construction for runtime adapters.

Runtimes can own the built-in HTTP clients used by their configured models.
Injected models and custom providers keep their existing ownership.
"""

import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence, Tuple

from nemoguardrails.exceptions import InvalidModelConfigurationError
from nemoguardrails.llm.cache import CacheInterface, LFUCache
from nemoguardrails.llm.frameworks import get_default_framework, get_framework
from nemoguardrails.llm.models.initializer import init_llm_model
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.types import LLMModel

if TYPE_CHECKING:
    from nemoguardrails.llm.frameworks.default import _ClientPool

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelResourcePlan:
    """Validated selection; retain configuration order and duplicate entries."""

    main_config: Optional[Model]
    main_model: Optional[LLMModel]
    specialized_configs: Tuple[Model, ...]
    cache_configs: Tuple[Model, ...]


@dataclass
class ModelResources:
    """Constructed models and fresh caches for one runtime's action bindings."""

    main_model: Optional[LLMModel]
    specialized_models: Dict[str, LLMModel]
    caches: Dict[str, CacheInterface]
    _owned_clients: Optional["_ClientPool"] = field(default=None, repr=False)
    _cleanup_pending: bool = field(default=False, init=False, repr=False)

    async def start(self) -> None:
        """Complete interrupted cleanup before a runtime reuses model resources."""
        if self._cleanup_pending:
            await self.aclose()

    async def aclose(self) -> None:
        """Close owned model transport; borrowed models/frameworks are untouched."""
        if self._owned_clients is not None:
            self._cleanup_pending = True
            await self._owned_clients.aclose()
            self._cleanup_pending = False


def resolve_model_resources(
    models: Sequence[Model],
    *,
    main_model: Optional[LLMModel] = None,
) -> ModelResourcePlan:
    """Select and validate resources without constructing models or reading secrets.

    Embeddings are managed separately. Jailbreak detection can own a rail
    cache without using a generation model.
    The first configured main wins, unless a caller supplies one.
    """
    main_config = None
    specialized_configs = []
    cache_configs = []
    for model in models:
        if model.type == "main":
            if main_model is None and main_config is None:
                main_config = model
        elif model.type not in {"embeddings", "jailbreak_detection"}:
            specialized_configs.append(model)
        if model.type not in {"main", "embeddings"} and model.cache and model.cache.enabled:
            _validate_model_cache(model)
            cache_configs.append(model)

    selected_models = ([main_config] if main_config is not None else []) + specialized_configs
    for model in selected_models:
        if not model.model or not model.model.strip():
            raise InvalidModelConfigurationError(f"`model` field must be set for model type '{model.type}'.")

    return ModelResourcePlan(main_config, main_model, tuple(specialized_configs), tuple(cache_configs))


def prepare_model_kwargs(model: Model) -> Dict[str, Any]:
    """Resolve an optional environment credential without mutating configuration."""
    kwargs = dict(model.parameters or {})
    if model.api_key_env_var:
        api_key = os.environ.get(model.api_key_env_var)
        if api_key:
            kwargs["api_key"] = api_key
    return kwargs


def _validate_model_cache(model: Model) -> None:
    assert model.cache is not None
    if model.cache.maxsize <= 0:
        raise ValueError(
            f"Invalid cache maxsize for model '{model.type}': {model.cache.maxsize}. "
            "Capacity must be greater than 0. Skipping cache creation."
        )


def _create_model_cache(model: Model) -> LFUCache:
    """Create a per-runtime cache using the configured capacity and statistics."""
    assert model.cache is not None
    stats_logging_interval = None
    if model.cache.stats.enabled and model.cache.stats.log_interval is not None:
        stats_logging_interval = model.cache.stats.log_interval
    cache = LFUCache(
        maxsize=model.cache.maxsize,
        track_stats=model.cache.stats.enabled,
        stats_logging_interval=stats_logging_interval,
    )
    log.info("Created cache for model '%s' with maxsize %s", model.type, model.cache.maxsize)
    return cache


def build_model_resources(
    plan: ModelResourcePlan,
    *,
    main_mode: Optional[Literal["chat", "text"]] = None,
    managed: bool = False,
    client_defaults: Optional[Dict[str, Any]] = None,
) -> ModelResources:
    """Construct through the selected framework, preserving legacy duplicate rules.

    Specialized models keep the first binding; caches keep the last enabled
    binding. Adapters may override the configured main mode for compatibility.
    """
    owned_clients = None
    initialize_model = init_llm_model
    if managed:
        from nemoguardrails.llm.frameworks.default import DefaultFramework, _ClientPool

        framework = get_framework(get_default_framework())
        # Subclasses can override model construction and own their transports.
        if type(framework) is DefaultFramework:
            owned_clients = _ClientPool(lazy=True, defaults=client_defaults)
            create_model = partial(framework._create_model_with_pool, pool=owned_clients)
            initialize_model = partial(init_llm_model, create_model=create_model)

    main_model = plan.main_model
    if plan.main_config is not None:
        config = plan.main_config
        assert config.model is not None
        main_model = initialize_model(
            model_name=config.model,
            provider_name=config.engine,
            mode=main_mode if main_mode is not None else config.mode,
            kwargs=prepare_model_kwargs(config),
        )

    specialized_models = {}
    for config in plan.specialized_configs:
        assert config.model is not None
        model = initialize_model(
            model_name=config.model,
            provider_name=config.engine,
            mode=config.mode,
            kwargs=prepare_model_kwargs(config),
        )
        specialized_models.setdefault(config.type, model)

    caches: Dict[str, CacheInterface] = {config.type: _create_model_cache(config) for config in plan.cache_configs}
    return ModelResources(
        main_model=main_model, specialized_models=specialized_models, caches=caches, _owned_clients=owned_clients
    )

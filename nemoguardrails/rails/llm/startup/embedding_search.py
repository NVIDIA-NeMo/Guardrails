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

"""Embedding search provider setup."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type

from nemoguardrails.embeddings.index import EmbeddingsIndex
from nemoguardrails.rails.llm.config import EmbeddingSearchProvider, RailsConfig

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_ENGINE = "FastEmbed"

__all__ = [
    "DEFAULT_EMBEDDING_ENGINE",
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingSearchState",
    "apply_embedding_model_config",
    "get_embedding_search_provider_instance",
]


@dataclass
class EmbeddingSearchState:
    providers: Dict[str, Type[EmbeddingsIndex]]
    default_model: Optional[str]
    default_engine: Optional[str]
    default_params: Dict[str, Any]

    @classmethod
    def default(cls) -> "EmbeddingSearchState":
        return cls(
            providers={},
            default_model=DEFAULT_EMBEDDING_MODEL,
            default_engine=DEFAULT_EMBEDDING_ENGINE,
            default_params={},
        )

    def update_defaults(
        self,
        *,
        default_model: Optional[str],
        default_engine: Optional[str],
        default_params: Dict[str, Any],
    ) -> None:
        self.default_model = default_model
        self.default_engine = default_engine
        self.default_params = default_params

    def register_provider(self, name: str, cls: Type[EmbeddingsIndex]) -> None:
        self.providers[name] = cls

    def get_provider_instance(
        self,
        esp_config: Optional[EmbeddingSearchProvider] = None,
    ) -> EmbeddingsIndex:
        return get_embedding_search_provider_instance(
            embedding_search_providers=self.providers,
            default_embedding_model=self.default_model,
            default_embedding_engine=self.default_engine,
            default_embedding_params=self.default_params,
            esp_config=esp_config,
        )


def apply_embedding_model_config(
    config: RailsConfig,
    default_embedding_model: Optional[str],
    default_embedding_engine: Optional[str],
    default_embedding_params: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """Apply an embeddings model config to the default embedding search settings."""
    for model in config.models:
        if model.type != "embeddings":
            continue

        default_embedding_model = model.model
        default_embedding_engine = model.engine
        default_embedding_params = model.parameters or {}

        for esp in [
            config.core.embedding_search_provider,
            config.knowledge_base.embedding_search_provider,
        ]:
            if esp.name != "default":
                continue
            if "embedding_model" not in esp.parameters and model.model is not None:
                esp.parameters["embedding_model"] = model.model
            if "embedding_engine" not in esp.parameters and model.engine is not None:
                esp.parameters["embedding_engine"] = model.engine

        break

    return default_embedding_model, default_embedding_engine, default_embedding_params


def get_embedding_search_provider_instance(
    embedding_search_providers: Dict[str, Type[EmbeddingsIndex]],
    default_embedding_model: Optional[str],
    default_embedding_engine: Optional[str],
    default_embedding_params: Dict[str, Any],
    esp_config: Optional[EmbeddingSearchProvider] = None,
) -> EmbeddingsIndex:
    """Create the embedding search provider selected by a provider config."""
    if esp_config is None:
        esp_config = EmbeddingSearchProvider()

    if esp_config.name == "default":
        from nemoguardrails.embeddings.basic import BasicEmbeddingsIndex

        return BasicEmbeddingsIndex(
            embedding_model=esp_config.parameters.get("embedding_model", default_embedding_model),
            embedding_engine=esp_config.parameters.get("embedding_engine", default_embedding_engine),
            embedding_params=esp_config.parameters.get("embedding_parameters", default_embedding_params),
            cache_config=esp_config.cache,
            **{
                k: v
                for k, v in esp_config.parameters.items()
                if k
                in [
                    "use_batching",
                    "max_batch_size",
                    "matx_batch_hold",
                    "search_threshold",
                ]
                and v is not None
            },
        )

    if esp_config.name not in embedding_search_providers:
        raise Exception(f"Unknown embedding search provider: {esp_config.name}")

    kwargs = esp_config.parameters
    return embedding_search_providers[esp_config.name](**kwargs)

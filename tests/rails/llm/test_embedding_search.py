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

import pytest

from nemoguardrails.embeddings.basic import BasicEmbeddingsIndex
from nemoguardrails.embeddings.index import EmbeddingsIndex
from nemoguardrails.rails.llm.config import EmbeddingSearchProvider, Model, RailsConfig
from nemoguardrails.rails.llm.startup import embedding_search as startup_embedding_search
from nemoguardrails.rails.llm.startup.embedding_search import (
    DEFAULT_EMBEDDING_ENGINE,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingSearchState,
    apply_embedding_model_config,
    get_embedding_search_provider_instance,
)


def test_embedding_search_startup_module_exports_public_helpers():
    assert startup_embedding_search.__all__ == [
        "DEFAULT_EMBEDDING_ENGINE",
        "DEFAULT_EMBEDDING_MODEL",
        "EmbeddingSearchState",
        "apply_embedding_model_config",
        "get_embedding_search_provider_instance",
    ]


def test_embedding_search_state_owns_providers_defaults_and_provider_creation():
    class CustomEmbeddingsIndex(EmbeddingsIndex):
        def __init__(self, prefix: str):
            self.prefix = prefix

    state = EmbeddingSearchState.default()

    assert state.providers == {}
    assert state.default_model == DEFAULT_EMBEDDING_MODEL
    assert state.default_engine == DEFAULT_EMBEDDING_ENGINE
    assert state.default_params == {}

    state.update_defaults(
        default_model="prepared-model",
        default_engine="PreparedEngine",
        default_params={"device": "cpu"},
    )
    state.register_provider("custom", CustomEmbeddingsIndex)

    index = state.get_provider_instance(EmbeddingSearchProvider(name="custom", parameters={"prefix": "docs"}))

    assert isinstance(index, CustomEmbeddingsIndex)
    assert index.prefix == "docs"
    assert state.default_model == "prepared-model"
    assert state.default_engine == "PreparedEngine"
    assert state.default_params == {"device": "cpu"}


def test_apply_embedding_model_config_backfills_default_search_providers():
    config = RailsConfig(
        models=[
            Model(
                type="embeddings",
                engine="SentenceTransformers",
                model="intfloat/e5-large-v2",
                parameters={"device": "cpu"},
            )
        ],
    )

    default_model, default_engine, default_params = apply_embedding_model_config(
        config=config,
        default_embedding_model=DEFAULT_EMBEDDING_MODEL,
        default_embedding_engine=DEFAULT_EMBEDDING_ENGINE,
        default_embedding_params={},
    )

    assert default_model == "intfloat/e5-large-v2"
    assert default_engine == "SentenceTransformers"
    assert default_params == {"device": "cpu"}
    assert config.core.embedding_search_provider.parameters["embedding_model"] == "intfloat/e5-large-v2"
    assert config.core.embedding_search_provider.parameters["embedding_engine"] == "SentenceTransformers"
    assert config.knowledge_base.embedding_search_provider.parameters["embedding_model"] == "intfloat/e5-large-v2"
    assert config.knowledge_base.embedding_search_provider.parameters["embedding_engine"] == "SentenceTransformers"


def test_get_embedding_search_provider_instance_uses_default_provider_parameters():
    esp_config = EmbeddingSearchProvider(
        parameters={
            "embedding_model": "custom-model",
            "embedding_engine": "CustomEngine",
            "embedding_parameters": {"device": "cpu"},
            "use_batching": True,
            "max_batch_size": 3,
            "search_threshold": 0.7,
        }
    )

    index = get_embedding_search_provider_instance(
        embedding_search_providers={},
        default_embedding_model=DEFAULT_EMBEDDING_MODEL,
        default_embedding_engine=DEFAULT_EMBEDDING_ENGINE,
        default_embedding_params={},
        esp_config=esp_config,
    )

    assert isinstance(index, BasicEmbeddingsIndex)
    assert index.embedding_model == "custom-model"
    assert index.embedding_engine == "CustomEngine"
    assert index.embedding_params == {"device": "cpu"}
    assert index.use_batching is True
    assert index.max_batch_size == 3
    assert index.search_threshold == 0.7


def test_get_embedding_search_provider_instance_uses_registered_custom_provider():
    class CustomEmbeddingsIndex(EmbeddingsIndex):
        def __init__(self, prefix: str):
            self.prefix = prefix

    index = get_embedding_search_provider_instance(
        embedding_search_providers={"custom": CustomEmbeddingsIndex},
        default_embedding_model=DEFAULT_EMBEDDING_MODEL,
        default_embedding_engine=DEFAULT_EMBEDDING_ENGINE,
        default_embedding_params={},
        esp_config=EmbeddingSearchProvider(name="custom", parameters={"prefix": "docs"}),
    )

    assert isinstance(index, CustomEmbeddingsIndex)
    assert index.prefix == "docs"


def test_get_embedding_search_provider_instance_rejects_unknown_provider():
    with pytest.raises(Exception, match="Unknown embedding search provider: missing"):
        get_embedding_search_provider_instance(
            embedding_search_providers={},
            default_embedding_model=DEFAULT_EMBEDDING_MODEL,
            default_embedding_engine=DEFAULT_EMBEDDING_ENGINE,
            default_embedding_params={},
            esp_config=EmbeddingSearchProvider(name="missing"),
        )

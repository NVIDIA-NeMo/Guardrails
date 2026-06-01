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

from nemoguardrails.rails.llm.config import Model, RailsConfig
from nemoguardrails.rails.llm.startup import config_preparation as startup_config_preparation
from nemoguardrails.rails.llm.startup.config_preparation import (
    prepare_llmrails_config,
)
from nemoguardrails.rails.llm.startup.embedding_search import (
    DEFAULT_EMBEDDING_ENGINE,
    DEFAULT_EMBEDDING_MODEL,
)


def test_config_preparation_startup_module_exports_public_helpers():
    assert startup_config_preparation.__all__ == [
        "PreparedLLMRailsConfig",
        "prepare_llmrails_config",
    ]


def test_prepare_llmrails_config_loads_colang_flows_once_in_place():
    config = RailsConfig(models=[])

    first = prepare_llmrails_config(
        config=config,
        default_embedding_model=DEFAULT_EMBEDDING_MODEL,
        default_embedding_engine=DEFAULT_EMBEDDING_ENGINE,
        default_embedding_params={},
    )
    first_flow_count = len(config.flows)

    second = prepare_llmrails_config(
        config=config,
        default_embedding_model=DEFAULT_EMBEDDING_MODEL,
        default_embedding_engine=DEFAULT_EMBEDDING_ENGINE,
        default_embedding_params={},
    )

    assert first.config is config
    assert second.config is config
    assert first_flow_count > 0
    assert len(config.flows) == first_flow_count


def test_prepare_llmrails_config_marks_rail_flows_each_time():
    config = RailsConfig(
        models=[],
        flows=[{"id": "check input", "is_system_flow": False}],
    )

    prepare_llmrails_config(
        config=config,
        default_embedding_model=DEFAULT_EMBEDDING_MODEL,
        default_embedding_engine=DEFAULT_EMBEDDING_ENGINE,
        default_embedding_params={},
    )

    config.rails.input.flows = ["check input"]
    prepare_llmrails_config(
        config=config,
        default_embedding_model=DEFAULT_EMBEDDING_MODEL,
        default_embedding_engine=DEFAULT_EMBEDDING_ENGINE,
        default_embedding_params={},
    )

    assert config.flows[0]["is_system_flow"] is True
    assert config.flows[0]["is_subflow"] is True


def test_prepare_llmrails_config_can_prepare_a_copy():
    config = RailsConfig(models=[])

    prepared = prepare_llmrails_config(
        config=config,
        default_embedding_model=DEFAULT_EMBEDDING_MODEL,
        default_embedding_engine=DEFAULT_EMBEDDING_ENGINE,
        default_embedding_params={},
        in_place=False,
    )

    assert prepared.config is not config
    assert len(config.flows) == 0
    assert len(prepared.config.flows) > 0


def test_prepare_llmrails_config_returns_embedding_defaults():
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

    prepared = prepare_llmrails_config(
        config=config,
        default_embedding_model=DEFAULT_EMBEDDING_MODEL,
        default_embedding_engine=DEFAULT_EMBEDDING_ENGINE,
        default_embedding_params={},
    )

    assert prepared.default_embedding_model == "intfloat/e5-large-v2"
    assert prepared.default_embedding_engine == "SentenceTransformers"
    assert prepared.default_embedding_params == {"device": "cpu"}
    assert config.core.embedding_search_provider.parameters["embedding_model"] == "intfloat/e5-large-v2"

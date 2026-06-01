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

"""LLMRails config preparation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.startup.colang_flows import (
    load_default_colang_1_flows,
    load_guardrails_library_flows_and_bot_messages,
    mark_rail_flows_as_system_subflows,
)
from nemoguardrails.rails.llm.startup.embedding_search import apply_embedding_model_config

_COLANG_FLOWS_PREPARED_ATTR = "_llmrails_colang_flows_prepared"

__all__ = ["PreparedLLMRailsConfig", "prepare_llmrails_config"]


@dataclass
class PreparedLLMRailsConfig:
    config: RailsConfig
    default_embedding_model: Optional[str]
    default_embedding_engine: Optional[str]
    default_embedding_params: Dict[str, Any]


def prepare_llmrails_config(
    *,
    config: RailsConfig,
    default_embedding_model: Optional[str],
    default_embedding_engine: Optional[str],
    default_embedding_params: Dict[str, Any],
    in_place: bool = True,
) -> PreparedLLMRailsConfig:
    """Prepare a RailsConfig for the standard LLMRails runtime."""
    prepared_config = config if in_place else config.model_copy(deep=True)

    if not getattr(prepared_config, _COLANG_FLOWS_PREPARED_ATTR, False):
        load_default_colang_1_flows(prepared_config)
        load_guardrails_library_flows_and_bot_messages(prepared_config)
        setattr(prepared_config, _COLANG_FLOWS_PREPARED_ATTR, True)

    mark_rail_flows_as_system_subflows(prepared_config)

    (
        default_embedding_model,
        default_embedding_engine,
        default_embedding_params,
    ) = apply_embedding_model_config(
        config=prepared_config,
        default_embedding_model=default_embedding_model,
        default_embedding_engine=default_embedding_engine,
        default_embedding_params=default_embedding_params,
    )

    return PreparedLLMRailsConfig(
        config=prepared_config,
        default_embedding_model=default_embedding_model,
        default_embedding_engine=default_embedding_engine,
        default_embedding_params=default_embedding_params,
    )

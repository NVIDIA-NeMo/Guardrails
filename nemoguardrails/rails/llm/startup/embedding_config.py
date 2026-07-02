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

"""Startup-phase derivation of the default embedding-search settings from config."""

from typing import Any, Dict, Optional, Tuple

from nemoguardrails.rails.llm.config import RailsConfig

__all__ = ["apply_embedding_model_config"]


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

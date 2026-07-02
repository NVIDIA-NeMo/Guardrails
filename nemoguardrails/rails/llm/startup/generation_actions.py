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

"""LLM generation action registration."""

from typing import Any, Protocol, Type

from nemoguardrails.actions.llm.generation import LLMGenerationActions
from nemoguardrails.actions.v2_x.generation import LLMGenerationActionsV2dotx

__all__ = [
    "GenerationActionRails",
    "GenerationActionRuntime",
    "LLMGenerationActions",
    "LLMGenerationActionsV2dotx",
    "generation_actions_class_for_colang_version",
    "register_llm_generation_actions",
]


class GenerationActionRuntime(Protocol):
    @property
    def llm_task_manager(self) -> Any: ...

    def register_actions(self, actions_obj: Any, /, override: bool = True) -> None: ...


class GenerationActionRails(Protocol):
    _llm_generation_actions: Any

    @property
    def config(self) -> Any: ...

    @property
    def llm(self) -> Any: ...

    @property
    def runtime(self) -> GenerationActionRuntime: ...

    def _get_embeddings_search_provider_instance(self, esp_config: Any = None) -> Any: ...


def generation_actions_class_for_colang_version(colang_version: str) -> Type[Any]:
    """Return the LLM generation actions class selected by the Colang version."""
    return LLMGenerationActions if colang_version == "1.0" else LLMGenerationActionsV2dotx


def register_llm_generation_actions(rails: GenerationActionRails, verbose: bool) -> None:
    """Create and register the LLM generation actions for an LLMRails instance."""
    llm_generation_actions_class = generation_actions_class_for_colang_version(rails.config.colang_version)
    rails._llm_generation_actions = llm_generation_actions_class(
        config=rails.config,
        llm=rails.llm,
        llm_task_manager=rails.runtime.llm_task_manager,
        get_embedding_search_provider_instance=rails._get_embeddings_search_provider_instance,
        verbose=verbose,
    )

    rails.runtime.register_actions(rails._llm_generation_actions, override=False)

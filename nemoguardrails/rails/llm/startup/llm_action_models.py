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

"""LLM action model loading."""

import logging
import os
from typing import Any, Callable, Dict, Protocol

from nemoguardrails.exceptions import InvalidModelConfigurationError
from nemoguardrails.llm.models.initializer import ModelInitializationError
from nemoguardrails.types import LLMModel

log = logging.getLogger("nemoguardrails.rails.llm.llmrails")

InitLLM = Callable[..., LLMModel]

__all__ = [
    "InitLLM",
    "LLMActionRails",
    "LLMActionRuntime",
    "load_llm_action_models",
    "model_kwargs_from_config",
    "sync_update_llm_bindings",
]


class LLMActionRuntime(Protocol):
    def register_action_param(self, name: str, value: Any) -> None: ...


class LLMActionRails(Protocol):
    llm: Any
    _llm_generation_actions: Any

    @property
    def config(self) -> Any: ...

    @property
    def runtime(self) -> LLMActionRuntime: ...


def model_kwargs_from_config(model_config: Any) -> Dict[str, Any]:
    """Prepare model kwargs, including optional API key environment variables."""
    kwargs = dict(model_config.parameters or {})

    if model_config.api_key_env_var:
        api_key = os.environ.get(model_config.api_key_env_var)
        if api_key:
            kwargs["api_key"] = api_key

    return kwargs


def sync_update_llm_bindings(rails: LLMActionRails, llm: LLMModel) -> None:
    """Synchronize the main LLM bindings after a public update."""
    rails.llm = llm
    rails._llm_generation_actions.llm = llm
    rails.runtime.register_action_param("llm", llm)


def load_llm_action_models(rails: LLMActionRails, init_llm: InitLLM) -> None:
    """Load the main and action LLMs configured for an LLMRails instance."""
    from nemoguardrails._compat.langchain_kwargs import check_langchain_kwargs
    from nemoguardrails.llm.frameworks import get_default_framework

    prepare_model_kwargs = getattr(rails, "_prepare_model_kwargs", model_kwargs_from_config)
    models_to_check = (
        [model for model in rails.config.models if model.type != "main"] if rails.llm else rails.config.models
    )
    check_langchain_kwargs(models_to_check, get_default_framework())

    if rails.llm:
        if any(model.type == "main" for model in rails.config.models):
            log.warning(
                "Both an LLM was provided via constructor and a main LLM is specified in the config. "
                "The LLM provided via constructor will be used and the main LLM from config will be ignored."
            )
        rails.runtime.register_action_param("llm", rails.llm)

    else:
        main_model = next((model for model in rails.config.models if model.type == "main"), None)

        if main_model and main_model.model:
            kwargs = prepare_model_kwargs(main_model)
            rails.llm = init_llm(
                model_name=main_model.model,
                provider_name=main_model.engine,
                mode="chat",
                kwargs=kwargs,
            )
            rails.runtime.register_action_param("llm", rails.llm)

        else:
            log.info("No main LLM specified in the config and no LLM provided via constructor.")

    llms = dict()

    for llm_config in rails.config.models:
        if llm_config.type in ["embeddings", "jailbreak_detection"]:
            continue

        if rails.llm and llm_config.type == "main":
            continue

        try:
            model_name = llm_config.model
            if not model_name:
                raise InvalidModelConfigurationError(
                    f"`model` field must be set in model configuration: {llm_config.model_dump_json()}"
                )

            provider_name = llm_config.engine
            kwargs = prepare_model_kwargs(llm_config)
            mode = llm_config.mode

            llm_model = init_llm(
                model_name=model_name,
                provider_name=provider_name,
                mode=mode,
                kwargs=kwargs,
            )

            if llm_config.type == "main":
                if not rails.llm:
                    rails.llm = llm_model
                    rails.runtime.register_action_param("llm", rails.llm)
            else:
                model_attr = f"{llm_config.type}_llm"
                if not hasattr(rails, model_attr):
                    setattr(rails, model_attr, llm_model)
                rails.runtime.register_action_param(model_attr, getattr(rails, model_attr))
                llms[llm_config.type] = getattr(rails, model_attr)

        except ModelInitializationError as e:
            log.error("Failed to initialize model: %s", str(e))
            raise
        except Exception as e:
            log.error("Unexpected error initializing model: %s", str(e))
            raise

    rails.runtime.register_action_param("llms", llms)

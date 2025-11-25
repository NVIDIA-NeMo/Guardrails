# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Module for initializing LLM models with proper error handling and type checking."""

from typing import Any, Dict, Literal, Union

from langchain_core.language_models import BaseChatModel, BaseLLM

from nemoguardrails.llm.models.langchain_initializer import (
    ModelInitializationError,
    init_langchain_model,
)


# later we can easily convert it to a class
def init_llm_model(
    model_name: str,
    provider_name: str,
    mode: Literal["chat", "text"],
    kwargs: Dict[str, Any],
) -> Union[BaseChatModel, BaseLLM]:
    """Initialize an LLM model with proper error handling.

    Currently, this function only supports LangChain models.
    In the future, it may support other model backends.

    Args:
        model_name: Name of the model to initialize
        provider_name: Name of the provider to use
        mode: Literal taking either "chat" or "text" values
        kwargs: Additional arguments to pass to the model initialization

    Returns:
        An initialized LLM model

    Raises:
        ModelInitializationError: If model initialization fails
    """
    # currently we only support LangChain models
    return init_langchain_model(
        model_name=model_name,
        provider_name=provider_name,
        mode=mode,
        kwargs=kwargs,
    )


__all__ = ["init_llm_model", "ModelInitializationError"]

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

"""Module for initializing LLM models with proper error handling and type checking."""

from typing import Any, Dict, Literal

from nemoguardrails.llm.frameworks import get_default_framework, get_framework
from nemoguardrails.types import LLMModel


class ModelInitializationError(Exception):
    pass


def init_llm_model(
    model_name: str,
    provider_name: str,
    kwargs: Dict[str, Any],
    mode: Literal["chat", "text"] = "chat",
) -> LLMModel:
    """Initialize an LLM model with proper error handling.

    Args:
        model_name: Name of the model to initialize
        provider_name: Name of the provider to use
        kwargs: Additional arguments to pass to the model initialization
        mode: Literal taking either "chat" or "text" values

    Returns:
        An initialized LLM model

    Raises:
        ModelInitializationError: If model initialization fails
    """
    model_kwargs = dict(kwargs) if kwargs else {}
    model_kwargs["mode"] = mode

    framework = get_framework(get_default_framework())
    return framework.create_model(
        model_name=model_name,
        provider_name=provider_name,
        model_kwargs=model_kwargs,
    )


__all__ = ["init_llm_model", "ModelInitializationError"]

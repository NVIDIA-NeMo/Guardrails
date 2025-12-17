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

"""OpenAI API schema definitions for the NeMo Guardrails server."""

import os
from typing import List, Optional

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.model import Model
from pydantic import BaseModel, Field


class GuardrailsChatCompletion(ChatCompletion):
    """OpenAI API response body with NeMo-Guardrails extensions."""

    config_id: Optional[str] = Field(
        default=None,
        description="The guardrails configuration ID associated with this response.",
    )
    state: Optional[dict] = Field(default=None, description="State object for continuing the conversation.")
    llm_output: Optional[dict] = Field(default=None, description="Additional LLM output data.")
    output_data: Optional[dict] = Field(default=None, description="Additional output data.")
    log: Optional[dict] = Field(default=None, description="Generation log data.")


class GuardrailsModel(Model):
    """OpenAI API model with NeMo-Guardrails extensions."""

    config_id: Optional[str] = Field(
        default=None,
        description="[NeMo Guardrails extension] The guardrails configuration ID associated with this model.",
    )
    engine: Optional[str] = Field(
        default_factory=lambda: os.getenv("MAIN_MODEL_ENGINE", "nim"),
        description="[NeMo Guardrails extension] The engine associated with this model.",
    )
    base_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("MAIN_MODEL_BASE_URL", "https://localhost:8000/v1"),
        description="[NeMo Guardrails extension] The base URL this model serves on.",
    )
    api_key_env_var: Optional[str] = Field(
        default_factory=lambda: os.getenv("MAIN_MODEL_API_KEY", None),
        description="[NeMo Guardrails extension] This model's API key.",
    )


class GuardrailsModelsResponse(BaseModel):
    """OpenAI API models list response with NeMo-Guardrails extensions."""

    object: str = Field(default="list", description="The object type, which is always 'list'.")
    data: List[GuardrailsModel] = Field(description="The list of models.")

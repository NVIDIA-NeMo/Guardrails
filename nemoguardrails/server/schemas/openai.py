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

from typing import List, Optional, Union

from pydantic import BaseModel, Field


class OpenAIRequestFields(BaseModel):
    """OpenAI API request fields that can be mixed into other request schemas."""

    # Standard OpenAI completion parameters
    model: Optional[str] = Field(
        default=None,
        description="The model to use for chat completion. Maps to config_id for backward compatibility.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to generate.",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature to use.",
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Top-p sampling parameter.",
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Stop sequences.",
    )
    presence_penalty: Optional[float] = Field(
        default=None,
        description="Presence penalty parameter.",
    )
    frequency_penalty: Optional[float] = Field(
        default=None,
        description="Frequency penalty parameter.",
    )
    function_call: Optional[dict] = Field(
        default=None,
        description="Function call parameter.",
    )
    logit_bias: Optional[dict] = Field(
        default=None,
        description="Logit bias parameter.",
    )
    log_probs: Optional[bool] = Field(
        default=None,
        description="Log probabilities parameter.",
    )


class Choice(BaseModel):
    """OpenAI API choice structure in chat completion responses."""

    index: Optional[int] = Field(
        default=None, description="The index of the choice in the list of choices."
    )
    message: Optional[dict] = Field(
        default=None, description="The message of the choice"
    )
    logprobs: Optional[dict] = Field(
        default=None, description="The log probabilities of the choice"
    )
    finish_reason: Optional[str] = Field(
        default=None, description="The reason the model stopped generating tokens."
    )


class ResponseBody(BaseModel):
    """OpenAI API response body with NeMo-Guardrails extensions."""

    # OpenAI API fields
    id: Optional[str] = Field(
        default=None, description="A unique identifier for the chat completion."
    )
    object: str = Field(
        default="chat.completion",
        description="The object type, which is always chat.completion",
    )
    created: Optional[int] = Field(
        default=None,
        description="The Unix timestamp (in seconds) of when the chat completion was created.",
    )
    model: Optional[str] = Field(
        default=None, description="The model used for the chat completion."
    )
    choices: Optional[List[Choice]] = Field(
        default=None, description="A list of chat completion choices."
    )
    # NeMo-Guardrails specific fields for backward compatibility
    state: Optional[dict] = Field(
        default=None, description="State object for continuing the conversation."
    )
    llm_output: Optional[dict] = Field(
        default=None, description="Additional LLM output data."
    )
    output_data: Optional[dict] = Field(
        default=None, description="Additional output data."
    )
    log: Optional[dict] = Field(default=None, description="Generation log data.")


class Model(BaseModel):
    """OpenAI API model representation."""

    id: str = Field(
        description="The model identifier, which can be referenced in the API endpoints."
    )
    object: str = Field(
        default="model", description="The object type, which is always 'model'."
    )
    created: int = Field(
        description="The Unix timestamp (in seconds) of when the model was created."
    )
    owned_by: str = Field(
        default="nemo-guardrails", description="The organization that owns the model."
    )


class ModelsResponse(BaseModel):
    """OpenAI API models list response."""

    object: str = Field(
        default="list", description="The object type, which is always 'list'."
    )
    data: List[Model] = Field(description="The list of models.")

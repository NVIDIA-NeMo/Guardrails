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

"""OpenAI API schema definitions for the NeMo Guardrails server."""

import os
from typing import Any, List, Optional, Union

from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from nemoguardrails.rails.llm.options import GenerationOptions


class GuardrailsDataOutput(BaseModel):
    """Guardrails-specific output data."""

    config_id: Optional[str] = Field(
        default=None,
        description="The guardrails configuration ID associated with this response.",
    )
    state: Optional[dict] = Field(default=None, description="State object for continuing the conversation.")
    llm_output: Optional[dict] = Field(default=None, description="Additional LLM output data.")
    output_data: Optional[dict] = Field(default=None, description="Additional output data.")
    log: Optional[dict] = Field(default=None, description="Generation log data.")


class GuardrailsChatCompletion(ChatCompletion):
    """OpenAI API response body with NeMo-Guardrails extensions."""

    guardrails: Optional[GuardrailsDataOutput] = Field(default=None, description="Guardrails specific output data.")


class OpenAIChatCompletionRequest(BaseModel):
    """Standard OpenAI chat completion request parameters."""

    messages: Optional[List[dict]] = Field(
        default=None,
        description="The list of messages in the current conversation.",
    )
    model: str = Field(
        ...,
        description="The LLM model to use for chat completion (e.g., 'gpt-4o', 'llama-3.1-8b').",
    )
    stream: Optional[bool] = Field(
        default=False,
        description="If set, partial message deltas will be sent as server-sent events.",
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
    logprobs: Optional[bool] = Field(
        default=None,
        description="Log probabilities parameter.",
    )


class GuardrailsDataInput(BaseModel):
    """Guardrails-specific options for the request."""

    config_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("DEFAULT_CONFIG_ID", None),
        description="The guardrails configuration ID to use.",
    )
    config_ids: Optional[List[str]] = Field(
        default=None,
        description="List of configuration IDs to combine.",
        validate_default=True,
    )
    thread_id: Optional[str] = Field(
        default=None,
        min_length=16,
        max_length=255,
        description="The ID of an existing thread to continue.",
    )
    context: Optional[dict] = Field(
        default=None,
        description="Additional context data for the conversation.",
    )
    options: GenerationOptions = Field(
        default_factory=GenerationOptions,
        description="Additional generation options.",
    )
    state: Optional[dict] = Field(
        default=None,
        description="State object to continue the interaction.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_config_ids(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("config_id") is not None and data.get("config_ids") is not None:
                raise ValueError("Only one of config_id or config_ids should be specified")
        return data

    @field_validator("config_ids", mode="before")
    @classmethod
    def ensure_config_ids(cls, v: Any, info: ValidationInfo) -> Any:
        if v is None and info.data.get("config_id"):
            return [info.data["config_id"]]
        return v


class GuardrailsChatCompletionRequest(OpenAIChatCompletionRequest):
    """OpenAI chat completion request with NeMo Guardrails extensions."""

    guardrails: GuardrailsDataInput = Field(
        default_factory=GuardrailsDataInput,
        description="Guardrails specific options for the request.",
    )

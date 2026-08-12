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

"""Anthropic Messages API schema definitions for the NeMo Guardrails server."""

from typing import Any, Dict, List, Optional, Union

from anthropic.types import Message
from pydantic import BaseModel, Field

from nemoguardrails.server.schemas.openai import (
    GuardrailsDataInput,
    GuardrailsDataOutput,
)


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request parameters."""

    model: str = Field(
        ...,
        description="The model to use (e.g. 'claude-opus-4-8').",
    )
    max_tokens: int = Field(
        ...,
        description="The maximum number of tokens to generate.",
    )
    messages: List[Dict[str, Any]] = Field(
        ...,
        description="Input messages. Each message has a role and content.",
    )
    system: Optional[Union[str, List[Dict[str, Any]]]] = Field(
        default=None,
        description="System prompt (string or array of content blocks).",
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Whether to stream the response using SSE.",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature (0.0 to 1.0).",
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Nucleus sampling parameter.",
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Top-k sampling parameter.",
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None,
        description="Custom text sequences that cause the model to stop generating.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Request metadata (e.g. user_id for abuse detection).",
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Tool definitions available to the model.",
    )
    tool_choice: Optional[Dict[str, Any]] = Field(
        default=None,
        description="How the model should use the provided tools.",
    )
    thinking: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extended thinking configuration.",
    )
    cache_control: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Cache control for prompt caching.",
    )
    container: Optional[str] = Field(
        default=None,
        description="Container ID for sandboxed execution.",
    )
    inference_geo: Optional[str] = Field(
        default=None,
        description="Geographic constraint for inference.",
    )
    output_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output format configuration (e.g. structured output).",
    )
    service_tier: Optional[str] = Field(
        default=None,
        description="Service tier selection ('auto' or 'standard_only').",
    )
    user_profile_id: Optional[str] = Field(
        default=None,
        description="User profile ID for personalization.",
    )


class GuardrailsMessagesRequest(AnthropicMessagesRequest):
    """Anthropic Messages request with NeMo Guardrails extensions."""

    guardrails: GuardrailsDataInput = Field(
        default_factory=GuardrailsDataInput,
        description="Guardrails specific options for the request.",
    )


class GuardrailsMessagesCheckRequest(AnthropicMessagesRequest):
    """Anthropic Messages request for the checks endpoint."""

    guardrails: GuardrailsDataInput = Field(
        default_factory=GuardrailsDataInput,
        description="Guardrails specific options for the check request.",
    )


class GuardrailsMessagesResponse(Message):
    """Anthropic Messages API response with NeMo-Guardrails extensions."""

    guardrails: Optional[GuardrailsDataOutput] = Field(
        default=None,
        description="Guardrails specific output data.",
    )

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

from typing import Any, Optional, Union

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message model."""

    role: str = Field(..., description="The role of the message author")
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""

    model: str = Field(..., description="ID of the model to use")
    messages: list[Message] = Field(..., description="List of messages comprising the conversation")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate", ge=1)
    temperature: Optional[float] = Field(1.0, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    n: Optional[int] = Field(1, description="Number of completions to generate", ge=1, le=128)
    stream: Optional[bool] = Field(False, description="Whether to stream back partial progress")
    stop: Optional[Union[str, list[str]]] = Field(None, description="Sequences where the API will stop generating")
    presence_penalty: Optional[float] = Field(0.0, description="Presence penalty", ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, description="Frequency penalty", ge=-2.0, le=2.0)
    logit_bias: Optional[dict[str, float]] = Field(None, description="Modify likelihood of specified tokens")
    user: Optional[str] = Field(None, description="Unique identifier representing your end-user")


class CompletionRequest(BaseModel):
    """Text completion request model."""

    model: str = Field(..., description="ID of the model to use")
    prompt: Union[str, list[str]] = Field(..., description="The prompt(s) to generate completions for")
    max_tokens: Optional[int] = Field(16, description="Maximum number of tokens to generate", ge=1)
    temperature: Optional[float] = Field(1.0, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    n: Optional[int] = Field(1, description="Number of completions to generate", ge=1, le=128)
    stream: Optional[bool] = Field(False, description="Whether to stream back partial progress")
    logprobs: Optional[int] = Field(None, description="Include log probabilities", ge=0, le=5)
    echo: Optional[bool] = Field(False, description="Echo back the prompt in addition to completion")
    stop: Optional[Union[str, list[str]]] = Field(None, description="Sequences where the API will stop generating")
    presence_penalty: Optional[float] = Field(0.0, description="Presence penalty", ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, description="Frequency penalty", ge=-2.0, le=2.0)
    best_of: Optional[int] = Field(1, description="Number of completions to generate server-side", ge=1)
    logit_bias: Optional[dict[str, float]] = Field(None, description="Modify likelihood of specified tokens")
    user: Optional[str] = Field(None, description="Unique identifier representing your end-user")


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens used")


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""

    index: int = Field(..., description="The index of this choice")
    message: Message = Field(..., description="The generated message")
    finish_reason: str = Field(..., description="The reason the model stopped generating")


class CompletionChoice(BaseModel):
    """Text completion choice."""

    text: str = Field(..., description="The generated text")
    index: int = Field(..., description="The index of this choice")
    logprobs: Optional[dict[str, Any]] = Field(None, description="Log probability information")
    finish_reason: str = Field(..., description="The reason the model stopped generating")


class ChatCompletionResponse(BaseModel):
    """Chat completion response - https://platform.openai.com/docs/api-reference/chat/object"""

    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp when the completion was created")
    model: str = Field(..., description="The model used for completion")
    choices: list[ChatCompletionChoice] = Field(..., description="List of completion choices")
    usage: Usage = Field(..., description="Token usage information")


class CompletionResponse(BaseModel):
    """Text completion response. https://platform.openai.com/docs/api-reference/completions/object"""

    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("text_completion", description="Object type")
    created: int = Field(..., description="Unix timestamp when the completion was created")
    model: str = Field(..., description="The model used for completion")
    choices: list[CompletionChoice] = Field(..., description="List of completion choices")
    usage: Usage = Field(..., description="Token usage information")


class Model(BaseModel):
    """Model information."""

    id: str = Field(..., description="Model identifier")
    object: str = Field("model", description="Object type")
    created: int = Field(..., description="Unix timestamp when the model was created")
    owned_by: str = Field(..., description="Organization that owns the model")


class ModelsResponse(BaseModel):
    """Models list response."""

    object: str = Field("list", description="Object type")
    data: list[Model] = Field(..., description="List of available models")

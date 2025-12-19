# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Pydantic models for GLiNER server API."""

from pydantic import BaseModel

# =============================================================================
# Core Entity Models
# =============================================================================


class EntitySpan(BaseModel):
    """Represents a detected entity with its position and metadata."""

    value: str
    suggested_label: str
    start_position: int  # inclusive - character index where entity starts
    end_position: int  # exclusive - character index where entity ends (Python slicing style)
    score: float


class GLiNERRequest(BaseModel):
    """Request model for GLiNER entity extraction."""

    text: str
    labels: list[str] | None = None
    threshold: float | None = 0.5
    chunk_length: int | None = 384
    overlap: int | None = 128
    flat_ner: bool | None = False


class GLiNERResponse(BaseModel):
    """Response model for GLiNER entity extraction."""

    entities: list[EntitySpan]  # List of entity spans with positions
    total_entities: int  # Total count of entities found
    tagged_text: str  # Tagged text with [entity](label) format


# =============================================================================
# OpenAI-Compatible Chat Completion Models
# =============================================================================


class ChatMessage(BaseModel):
    """Chat message model for OpenAI-compatible API."""

    role: str
    content: str


class GLiNERChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request with GLiNER parameters."""

    model: str = "gliner-ner"
    messages: list[ChatMessage]
    temperature: float | None = 0.1
    max_tokens: int | None = 1000
    stream: bool | None = False
    # GLiNER-specific parameters
    entity_labels: list[str] | None = None
    threshold: float | None = 0.5
    chunk_length: int | None = 384
    overlap: int | None = 128
    flat_ner: bool | None = False


class GLiNERChatCompletionChoice(BaseModel):
    """Chat completion choice model."""

    index: int
    message: ChatMessage
    finish_reason: str


class GLiNERChatCompletionUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GLiNERChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[GLiNERChatCompletionChoice]
    usage: GLiNERChatCompletionUsage


# =============================================================================
# Models Endpoint Models
# =============================================================================


class ModelInfo(BaseModel):
    """Model information for OpenAI-compatible models endpoint."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "gliner"


class ModelsResponse(BaseModel):
    """Response for models listing endpoint."""

    object: str = "list"
    data: list[ModelInfo]

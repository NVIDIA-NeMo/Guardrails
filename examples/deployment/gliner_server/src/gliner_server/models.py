# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pydantic import BaseModel, Field

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
    labels: list[str] | None = None  # None means use server defaults
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    chunk_length: int = 384
    overlap: int = 128
    flat_ner: bool = False


class GLiNERResponse(BaseModel):
    """Response model for GLiNER entity extraction."""

    entities: list[EntitySpan]  # List of entity spans with positions
    total_entities: int  # Total count of entities found
    tagged_text: str  # Tagged text with [entity](label) format


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

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

"""Pydantic models for GLiNER requests and responses."""

from typing import List, Optional

from pydantic import BaseModel, Field


class GLiNERRequest(BaseModel):
    """Request model for GLiNER entity extraction.

    This model defines the parameters for making requests to a GLiNER server.
    Default values are centralized here to avoid duplication.
    """

    text: str = Field(description="The text to analyze for entities.")
    labels: Optional[List[str]] = Field(
        default=None,
        description="List of entity labels to detect. If None, uses server defaults.",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for entity detection (0.0 to 1.0).",
    )
    chunk_length: int = Field(
        default=384,
        description="Length of text chunks for processing.",
    )
    overlap: int = Field(
        default=128,
        description="Overlap between chunks.",
    )
    flat_ner: bool = Field(
        default=False,
        description="Whether to use flat NER mode. Setting to False allows for nested entities.",
    )


class EntitySpan(BaseModel):
    """Represents a detected entity with its position and metadata."""

    value: str = Field(description="The detected entity text.")
    suggested_label: str = Field(description="The entity label/type.")
    start_position: int = Field(description="Start character index (inclusive).")
    end_position: int = Field(description="End character index (exclusive).")
    score: float = Field(description="Confidence score.")


class GLiNERResponse(BaseModel):
    """Response model for GLiNER entity extraction."""

    entities: List[EntitySpan] = Field(description="List of detected entities.")
    total_entities: int = Field(description="Total count of entities found.")
    tagged_text: str = Field(description="Text with entities tagged as [entity](label).")

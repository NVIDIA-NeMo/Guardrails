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

from typing import List, Optional

from nemoguardrails.manifests.config_schema import (
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    model_validator,
    rail_field,
)


class GLiNERDetectionOptions(RailConfigBaseModel):
    """Configuration options for GLiNER."""

    model_config = {"extra": "forbid"}

    entities: List[str] = Field(
        default_factory=list,
        description="The list of entity labels to detect (e.g., 'email', 'phone_number', 'ssn').",
    )


class GLiNERDetection(RailConfigBaseModel):
    """Configuration for GLiNER PII detection."""

    model_config = {"extra": "forbid"}

    server_endpoint: str = Field(
        default="http://localhost:8000/v1/chat/completions",
        description=(
            "The endpoint for the GLiNER detection server. "
            "By default, this is for a locally hosted NIM instance running the GLiNER model. "
            "Changed from http://localhost:1235/v1/extract (custom server) to "
            "http://localhost:8000/v1/chat/completions (NIM) in this release. "
            "If you use the custom gliner_server, set this explicitly to http://localhost:1235/v1/extract."
        ),
    )
    model: str = Field(
        default="nvidia/gliner-pii",
        description="Model identifier sent in NIM API requests (only used when server_endpoint ends with /v1/chat/completions).",
    )
    api_key_env_var: Optional[str] = Field(
        default=None,
        description="Name of the environment variable containing the API key for authenticated endpoints (e.g., NVIDIA_API_KEY).",
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
    input: GLiNERDetectionOptions = Field(
        default_factory=GLiNERDetectionOptions,
        description="Configuration of the entities to be detected on the user input.",
    )
    output: GLiNERDetectionOptions = Field(
        default_factory=GLiNERDetectionOptions,
        description="Configuration of the entities to be detected on the bot output.",
    )
    retrieval: GLiNERDetectionOptions = Field(
        default_factory=GLiNERDetectionOptions,
        description="Configuration of the entities to be detected on retrieved relevant chunks.",
    )

    @model_validator(mode="after")
    def _validate_chunking(self) -> "GLiNERDetection":
        if self.chunk_length <= 0:
            raise ValueError("chunk_length must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be nonnegative")
        if self.overlap >= self.chunk_length:
            raise ValueError("overlap must be less than chunk_length")
        return self


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[GLiNERDetection],
        field_info=rail_field(
            default_factory=GLiNERDetection,
            description="Configuration for GLiNER PII detection.",
        ),
        exports={
            "GLiNERDetection": GLiNERDetection,
            "GLiNERDetectionOptions": GLiNERDetectionOptions,
        },
    )

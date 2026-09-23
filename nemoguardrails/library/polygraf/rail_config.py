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

from pydantic import ConfigDict

from nemoguardrails.manifests.config_schema import Field, RailConfigBaseModel, RailConfigSpec, rail_field


class PolygrafDetectionOptions(RailConfigBaseModel):
    """Configuration options for Polygraf."""

    model_config = ConfigDict(extra="forbid")

    entities: List[str] = Field(
        default_factory=list,
        description="The list of entities that should be detected.",
    )


class PolygrafDetection(RailConfigBaseModel):
    """Configuration for Polygraf PII detection."""

    model_config = ConfigDict(extra="forbid")

    server_endpoint: str = Field(
        default="http://localhost:8000/v1/pii/text-detect",
        description="The endpoint for the Polygraf detection server.",
    )
    input: PolygrafDetectionOptions = Field(
        default_factory=PolygrafDetectionOptions,
        description="Configuration of the entities to be detected on the user input.",
    )
    output: PolygrafDetectionOptions = Field(
        default_factory=PolygrafDetectionOptions,
        description="Configuration of the entities to be detected on the bot output.",
    )
    retrieval: PolygrafDetectionOptions = Field(
        default_factory=PolygrafDetectionOptions,
        description="Configuration of the entities to be detected on retrieved relevant chunks.",
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[PolygrafDetection],
        field_info=rail_field(
            default_factory=PolygrafDetection,
            description="Configuration for Polygraf PII detection.",
        ),
        exports={
            "PolygrafDetection": PolygrafDetection,
            "PolygrafDetectionOptions": PolygrafDetectionOptions,
        },
    )

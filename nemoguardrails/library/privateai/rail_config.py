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
    rail_field,
)


class PrivateAIDetectionOptions(RailConfigBaseModel):
    """Configuration options for Private AI."""

    entities: List[str] = Field(
        default_factory=list,
        description="The list of entities that should be detected.",
    )


class PrivateAIDetection(RailConfigBaseModel):
    """Configuration for Private AI."""

    server_endpoint: str = Field(
        default="http://localhost:8080/process/text",
        description="The endpoint for the private AI detection server.",
    )
    input: PrivateAIDetectionOptions = Field(
        default_factory=PrivateAIDetectionOptions,
        description="Configuration of the entities to be detected on the user input.",
    )
    output: PrivateAIDetectionOptions = Field(
        default_factory=PrivateAIDetectionOptions,
        description="Configuration of the entities to be detected on the bot output.",
    )
    retrieval: PrivateAIDetectionOptions = Field(
        default_factory=PrivateAIDetectionOptions,
        description="Configuration of the entities to be detected on retrieved relevant chunks.",
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[PrivateAIDetection],
        field_info=rail_field(
            default_factory=PrivateAIDetection,
            description="Configuration for Private AI.",
        ),
        exports={
            "PrivateAIDetection": PrivateAIDetection,
            "PrivateAIDetectionOptions": PrivateAIDetectionOptions,
        },
    )

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


class SensitiveDataDetectionOptions(RailConfigBaseModel):
    entities: List[str] = Field(
        default_factory=list,
        description="The list of entities that should be detected. "
        "Check out https://microsoft.github.io/presidio/supported_entities/ for"
        "the list of supported entities.",
    )
    # TODO: this is not currently in use.
    mask_token: str = Field(
        default="*",
        description="The token that should be used to mask the sensitive data.",
    )

    score_threshold: float = Field(
        default=0.2,
        description="The score threshold that should be used to detect the sensitive data.",
    )


class SensitiveDataDetection(RailConfigBaseModel):
    """Configuration of what sensitive data should be detected."""

    recognizers: List[dict] = Field(
        default_factory=list,
        description="Additional custom recognizers. "
        "Check out https://microsoft.github.io/presidio/tutorial/08_no_code/ for more details.",
    )
    input: SensitiveDataDetectionOptions = Field(
        default_factory=SensitiveDataDetectionOptions,
        description="Configuration of the entities to be detected on the user input.",
    )
    output: SensitiveDataDetectionOptions = Field(
        default_factory=SensitiveDataDetectionOptions,
        description="Configuration of the entities to be detected on the bot output.",
    )
    retrieval: SensitiveDataDetectionOptions = Field(
        default_factory=SensitiveDataDetectionOptions,
        description="Configuration of the entities to be detected on retrieved relevant chunks.",
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[SensitiveDataDetection],
        field_info=rail_field(
            default_factory=SensitiveDataDetection,
            description="Configuration for detecting sensitive data.",
        ),
        exports={
            "SensitiveDataDetection": SensitiveDataDetection,
            "SensitiveDataDetectionOptions": SensitiveDataDetectionOptions,
        },
    )

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

from typing import Dict, List, Literal, Optional

from nemoguardrails.manifests.config_schema import (
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    rail_field,
)


class ClavataRailOptions(RailConfigBaseModel):
    """Configuration data for the Clavata API"""

    policy: str = Field(
        description="The policy alias to use when evaluating inputs or outputs.",
    )

    labels: List[str] = Field(
        default_factory=list,
        description="""A list of labels to match against the policy.
        If no labels are provided, the overall policy result will be returned.
        If labels are provided, only hits on the provided labels will be considered a hit.""",
    )


class ClavataRailConfig(RailConfigBaseModel):
    """Configuration data for the Clavata API"""

    server_endpoint: str = Field(
        default="https://gateway.app.clavata.ai:8443",
        description="The endpoint for the Clavata API",
    )

    policies: Dict[str, str] = Field(
        default_factory=dict,
        description="A dictionary of policy aliases and their corresponding IDs.",
    )

    label_match_logic: Literal["ANY", "ALL"] = Field(
        default="ANY",
        description="""The logic to use when deciding whether the evaluation matched.
        If ANY, only one of the configured labels needs to be found in the input or output.
        If ALL, all configured labels must be found in the input or output.""",
    )

    input: Optional[ClavataRailOptions] = Field(
        default=None,
        description="Clavata configuration for an Input Guardrail",
    )
    output: Optional[ClavataRailOptions] = Field(
        default=None,
        description="Clavata configuration for an Output Guardrail",
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[ClavataRailConfig],
        field_info=rail_field(
            default_factory=ClavataRailConfig,
            description="Configuration for Clavata.",
        ),
        exports={
            "ClavataRailConfig": ClavataRailConfig,
            "ClavataRailOptions": ClavataRailOptions,
        },
    )

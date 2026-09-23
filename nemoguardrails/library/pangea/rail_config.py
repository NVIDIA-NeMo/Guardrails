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

from typing import Optional

from nemoguardrails.manifests.config_schema import (
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    rail_field,
)


class PangeaRailOptions(RailConfigBaseModel):
    """Configuration data for the Pangea AI Guard API"""

    recipe: str = Field(
        description="""Recipe key of a configuration of data types and settings defined in the Pangea User Console. It
        specifies the rules that are to be applied to the text, such as defang malicious URLs."""
    )


class PangeaRailConfig(RailConfigBaseModel):
    """Configuration data for the Pangea AI Guard API"""

    input: Optional[PangeaRailOptions] = Field(
        default=None,
        description="Pangea configuration for an Input Guardrail",
    )
    output: Optional[PangeaRailOptions] = Field(
        default=None,
        description="Pangea configuration for an Output Guardrail",
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[PangeaRailConfig],
        field_info=rail_field(
            default_factory=PangeaRailConfig,
            description="Configuration for Pangea.",
        ),
        exports={
            "PangeaRailConfig": PangeaRailConfig,
            "PangeaRailOptions": PangeaRailOptions,
        },
    )

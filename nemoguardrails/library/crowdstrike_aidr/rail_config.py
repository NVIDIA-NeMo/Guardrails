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


class CrowdStrikeAIDRRailConfig(RailConfigBaseModel):
    """Configuration data for the CrowdStrike AIDR API"""

    timeout: float = Field(
        default=30.0,
        description="Timeout in seconds for API requests to CrowdStrike AIDR",
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[CrowdStrikeAIDRRailConfig],
        field_info=rail_field(
            default_factory=CrowdStrikeAIDRRailConfig,
            description="Configuration for CrowdStrike AIDR.",
        ),
        exports={"CrowdStrikeAIDRRailConfig": CrowdStrikeAIDRRailConfig},
    )

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

from typing import Any, Dict

from nemoguardrails.manifests.config_schema import (
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    rail_field,
)


class FactCheckingRailConfig(RailConfigBaseModel):
    """Configuration data for the fact-checking rail."""

    parameters: Dict[str, Any] = Field(default_factory=dict)
    fallback_to_self_check: bool = Field(
        default=False,
        description="Whether to fall back to self-check if another method fail.",
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=FactCheckingRailConfig,
        field_info=rail_field(
            default_factory=FactCheckingRailConfig,
            description="Configuration data for the fact-checking rail.",
        ),
        exports={"FactCheckingRailConfig": FactCheckingRailConfig},
    )

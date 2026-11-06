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

DEFAULT_BLOCK_SEVERITIES = ["critical", "high"]


class ATRDetection(RailConfigBaseModel):
    block_severities: List[str] = Field(
        default_factory=lambda: list(DEFAULT_BLOCK_SEVERITIES),
        description="ATR match severities that flag the input. Options are 'critical', "
        "'high', 'medium', and 'low'. Matches below these severities are still "
        "returned but do not flag, which keeps false positives low. An explicit "
        "empty list flags nothing, giving a monitor-only rail that records "
        "matches without blocking.",
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[ATRDetection],
        field_info=rail_field(
            default_factory=ATRDetection,
            description="Configuration for Agent Threat Rules detection.",
        ),
        exports={"ATRDetection": ATRDetection},
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from nemoguardrails.manifests.config_schema import (
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    rail_field,
)


class ATRDetectionConfig(RailConfigBaseModel):
    severities: List[str] = Field(
        default_factory=lambda: ["critical", "high"],
        description="ATR severity levels that should block the request.",
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[ATRDetectionConfig],
        field_info=rail_field(
            default_factory=ATRDetectionConfig,
            description="Configuration for Agent Threat Rules detection.",
        ),
        exports={"ATRDetectionConfig": ATRDetectionConfig},
    )

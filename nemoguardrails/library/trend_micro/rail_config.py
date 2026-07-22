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

import logging
import os
from typing import Optional

from nemoguardrails.manifests.config_schema import (
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    rail_field,
)

log = logging.getLogger(__name__)


class TrendMicroRailConfig(RailConfigBaseModel):
    """Configuration data for the Trend Micro AI Guard API"""

    v1_url: str = Field(
        default="https://api.xdr.trendmicro.com/v3.0/aiSecurity/applyGuardrails",
        description="The endpoint for the Trend Micro AI Guard API. For other regions, use: https://api.{region}.xdr.trendmicro.com/v3.0/aiSecurity/applyGuardrails where region is eu, jp, au, in, sg, or mea.",
    )

    api_key_env_var: Optional[str] = Field(
        default=None,
        description="Environment variable containing API key for Trend Micro AI Guard",
    )

    application_name: str = Field(
        default="nemo-guardrails",
        description="Application name for TMV1-Application-Name header (REQUIRED). Must contain only letters, numbers, hyphens, and underscores, with a maximum length of 64 characters.",
        pattern=r"^[a-zA-Z0-9_-]+$",
        max_length=64,
    )

    detailed_response: bool = Field(
        default=False,
        description="If True, returns detailed AI Guard results with confidence scores (Prefer: return=representation). If False, returns minimal response with only action and reasons (Prefer: return=minimal).",
    )

    def get_api_key(self) -> Optional[str]:
        """Helper to return an API key (if it exists) from a Trend Micro configuration.
        The `api_key_env_var` field, a string stored in this environment variable.

        If the environment variable is not found None is returned.
        """

        if self.api_key_env_var:
            v1_api_key = os.getenv(self.api_key_env_var)
            if v1_api_key:
                return v1_api_key

            log.warning(
                "Specified a value for Trend Micro config api_key_env_var, but the referenced environment variable was not set."
            )

        return None


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[TrendMicroRailConfig],
        field_info=rail_field(
            default_factory=TrendMicroRailConfig,
            description="Configuration for Trend Micro.",
        ),
        exports={"TrendMicroRailConfig": TrendMicroRailConfig},
    )

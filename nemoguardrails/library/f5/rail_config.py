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


class F5GuardrailsRailConfig(RailConfigBaseModel):
    """Configuration for the F5 Guardrails integration.

    The API key must be provided through the F5_GUARDRAILS_API_KEY environment variable.
    """

    api_url: str = Field(
        default="https://us1.calypsoai.app",
        description="Base URL for the F5 Guardrails API.",
    )
    fail_open: bool = Field(
        default=False,
        description=(
            "If True, allow content through when the F5 Guardrails API is "
            "unreachable or returns an error. This changes the security posture "
            "of the rail and should be reviewed as part of the guardrails config."
        ),
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description=(
            "Number of additional attempts after receiving HTTP 429 from the F5 "
            "Guardrails API. Total attempts equal max_retries + 1. Set to 0 to "
            "disable rate-limit retries."
        ),
    )
    max_retry_after_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description=(
            "Upper bound (in seconds) on how long to honor a Retry-After header "
            "returned by the F5 Guardrails API. Larger values are clamped to "
            "this cap to prevent unbounded waits."
        ),
    )
    retry_backoff_seconds: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Base delay (in seconds) used with exponential backoff when a 429 "
            "response has no usable Retry-After header. The delay for attempt N "
            "is retry_backoff_seconds * 2**N, clamped to max_retry_after_seconds."
        ),
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[F5GuardrailsRailConfig],
        field_info=rail_field(
            default_factory=F5GuardrailsRailConfig,
            description="Configuration for F5 Guardrails (CalypsoAI).",
        ),
        exports={"F5GuardrailsRailConfig": F5GuardrailsRailConfig},
    )

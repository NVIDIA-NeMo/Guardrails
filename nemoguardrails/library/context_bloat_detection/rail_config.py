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

from typing import Literal, Optional

from nemoguardrails.manifests.config_schema import Field, RailConfigBaseModel, RailConfigSpec, rail_field


class ContextBloatDetectionConfig(RailConfigBaseModel):
    """Configuration for context bloat / context manipulation detection."""

    max_chars: int = Field(
        default=5000,
        gt=0,
        description="Size cap in characters. Inputs exceeding this are flagged.",
    )
    min_chars: int = Field(
        default=50,
        ge=0,
        description="Minimum characters before entropy/run/repetition checks apply. Shorter texts are only checked against size cap.",
    )
    min_entropy: float = Field(
        default=3.5,
        ge=0.0,
        le=8.0,
        description="Shannon entropy floor (bits/char). English prose is ~4.0-4.5.",
    )
    max_repetition_ratio: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Max fraction of repeated n-grams (0.0-1.0).",
    )
    ngram_size: int = Field(
        default=3,
        ge=1,
        description="Size of n-grams used for repetition detection.",
    )
    max_run_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Max fraction of text that is the longest single-char run.",
    )
    action: Literal["reject", "truncate", "warn"] = Field(
        default="reject",
        description="Action on detection: 'reject', 'truncate', or 'warn'.",
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[ContextBloatDetectionConfig],
        field_info=rail_field(
            default_factory=ContextBloatDetectionConfig,
            description="Configuration for context bloat / context manipulation detection.",
        ),
        exports={"ContextBloatDetectionConfig": ContextBloatDetectionConfig},
    )

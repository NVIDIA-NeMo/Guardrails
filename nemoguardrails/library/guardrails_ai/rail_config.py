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

from typing import Any, Dict, List, Optional

from nemoguardrails.manifests.config_schema import (
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    rail_field,
)


class GuardrailsAIValidatorConfig(RailConfigBaseModel):
    """Configuration for a single Guardrails AI validator."""

    name: str = Field(
        description="Unique identifier or import path for the Guardrails AI validator (e.g., 'toxic_language', 'pii', 'regex_match', or 'guardrails/competitor_check')."
    )

    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to pass to the validator during initialization (e.g., threshold, regex pattern).",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata to pass to the validator during validation (e.g., valid_topics, context).",
    )


class GuardrailsAIRailConfig(RailConfigBaseModel):
    """Configuration data for Guardrails AI integration."""

    validators: List[GuardrailsAIValidatorConfig] = Field(
        default_factory=list,
        description="List of Guardrails AI validators to apply. Each validator can have its own parameters and metadata.",
    )

    def get_validator_config(self, name: str) -> Optional[GuardrailsAIValidatorConfig]:
        """Get a specific validator configuration by name."""
        for _validator in self.validators:
            if _validator.name == name:
                return _validator
        return None


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[GuardrailsAIRailConfig],
        field_info=rail_field(
            default_factory=GuardrailsAIRailConfig,
            description="Configuration for Guardrails AI validators.",
        ),
        exports={
            "GuardrailsAIRailConfig": GuardrailsAIRailConfig,
            "GuardrailsAIValidatorConfig": GuardrailsAIValidatorConfig,
        },
    )

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

from enum import Enum
from typing import Any, Dict, Optional

from nemoguardrails.manifests.config_schema import (
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    rail_field,
)


class PatronusEvaluationSuccessStrategy(str, Enum):
    """
    Strategy for determining whether a Patronus Evaluation API
    request should pass, especially when multiple evaluators
    are called in a single request.
    ALL_PASS requires all evaluators to pass for success.
    ANY_PASS requires only one evaluator to pass for success.
    """

    ALL_PASS = "all_pass"
    ANY_PASS = "any_pass"


class PatronusEvaluateApiParams(RailConfigBaseModel):
    """Config to parameterize the Patronus Evaluate API call"""

    success_strategy: Optional[PatronusEvaluationSuccessStrategy] = Field(
        default=PatronusEvaluationSuccessStrategy.ALL_PASS,
        description="Strategy to determine whether the Patronus Evaluate API Guardrail passes or not.",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to the Patronus Evaluate API",
    )


class PatronusEvaluateConfig(RailConfigBaseModel):
    """Config for the Patronus Evaluate API call"""

    evaluate_config: PatronusEvaluateApiParams = Field(
        default_factory=PatronusEvaluateApiParams,
        description="Configuration passed to the Patronus Evaluate API",
    )


class PatronusRailConfig(RailConfigBaseModel):
    """Configuration data for the Patronus Evaluate API"""

    input: Optional[PatronusEvaluateConfig] = Field(
        default_factory=PatronusEvaluateConfig,
        description="Patronus Evaluate API configuration for an Input Guardrail",
    )
    output: Optional[PatronusEvaluateConfig] = Field(
        default_factory=PatronusEvaluateConfig,
        description="Patronus Evaluate API configuration for an Output Guardrail",
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[PatronusRailConfig],
        field_info=rail_field(
            default_factory=PatronusRailConfig,
            description="Configuration data for the Patronus Evaluate API.",
        ),
        exports={
            "PatronusEvaluationSuccessStrategy": PatronusEvaluationSuccessStrategy,
            "PatronusEvaluateApiParams": PatronusEvaluateApiParams,
            "PatronusEvaluateConfig": PatronusEvaluateConfig,
            "PatronusRailConfig": PatronusRailConfig,
        },
    )

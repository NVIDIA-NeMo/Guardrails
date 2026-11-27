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

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StringConstraints, model_validator

from nemoguardrails.manifests.config_schema import RailConfigBaseModel, RailConfigSpec, rail_field

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class NoulQuestionConfig(RailConfigBaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["noul"]
    instructions: NonBlank


class ChoiceQuestionConfig(RailConfigBaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice"]
    instructions: NonBlank
    choices: Annotated[dict[NonBlank, NonBlank], Field(min_length=2)]


class ProbabilityRule(RailConfigBaseModel):
    model_config = ConfigDict(extra="forbid")
    probability_at_least: float = Field(strict=True, ge=0, le=1, allow_inf_nan=False)


class ChoiceRule(RailConfigBaseModel):
    model_config = ConfigDict(extra="forbid")
    choice_in: list[NonBlank] = Field(min_length=1)


class DecisionRule(RailConfigBaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Annotated[NoulQuestionConfig | ChoiceQuestionConfig, Field(discriminator="type")]
    block_if: ProbabilityRule | ChoiceRule

    @model_validator(mode="after")
    def validate_block_condition(self):
        if isinstance(self.question, NoulQuestionConfig) and isinstance(self.block_if, ProbabilityRule):
            return self
        if isinstance(self.question, ChoiceQuestionConfig) and isinstance(self.block_if, ChoiceRule):
            if set(self.block_if.choice_in) <= self.question.choices.keys():
                return self
            raise ValueError("choice_in must reference declared choices")
        raise ValueError("block_if must match the decision question type")


class DecisionModelConfig(RailConfigBaseModel):
    model_config = ConfigDict(extra="forbid")
    rules: dict[NonBlank, DecisionRule] = Field(
        default_factory=dict, description="Named rules selected by $rule in decision model flows."
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(annotation=DecisionModelConfig, field_info=rail_field(default_factory=DecisionModelConfig))

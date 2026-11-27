# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Named Noul rules for input and output decision rails."""

from typing import Literal, Optional

from pydantic import ConfigDict, field_validator

from nemoguardrails.manifests.config_schema import Field, RailConfigBaseModel, RailConfigSpec, rail_field


class NoulQuestion(RailConfigBaseModel):
    """A yes/no question whose answer is a probability."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["noul"] = "noul"
    instructions: str = Field(min_length=1, description="A yes/no question; yes means the text should be blocked.")

    @field_validator("instructions")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instructions must not be blank")
        return value


class DecisionModelBlockCondition(RailConfigBaseModel):
    model_config = ConfigDict(extra="forbid")
    probability_at_least: float = Field(ge=0.0, le=1.0, description="Block at or above this probability.")


class DecisionModelRule(RailConfigBaseModel):
    model_config = ConfigDict(extra="forbid")
    question: NoulQuestion
    block_if: DecisionModelBlockCondition


class DecisionModelConfig(RailConfigBaseModel):
    """Rules selected by each flow's $rule parameter."""

    model_config = ConfigDict(extra="forbid")
    rules: dict[str, DecisionModelRule] = Field(min_length=1, description="Named decision rules.")
    fail_open: bool = Field(default=False, description="Allow text on provider failure instead of blocking.")

    @field_validator("rules")
    @classmethod
    def _nonblank_names(cls, value: dict[str, DecisionModelRule]) -> dict[str, DecisionModelRule]:
        if any(not name.strip() for name in value):
            raise ValueError("rule names must not be blank")
        return value


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[DecisionModelConfig],
        field_info=rail_field(default=None, description="Configuration for the decision model rail."),
        exports={
            "DecisionModelConfig": DecisionModelConfig,
            "DecisionModelRule": DecisionModelRule,
            "DecisionModelBlockCondition": DecisionModelBlockCondition,
            "NoulQuestion": NoulQuestion,
        },
    )

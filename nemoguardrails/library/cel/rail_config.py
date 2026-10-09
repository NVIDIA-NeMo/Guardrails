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

from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import ConfigDict

from nemoguardrails.manifests.config_schema import (
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    model_validator,
    rail_field,
)

if TYPE_CHECKING:
    from cel import Program


def compile_expression(expression: str) -> "Program":
    """Compile *expression*, importing the optional CEL library only when a config uses it."""
    try:
        import cel
    except ImportError as e:
        raise ImportError(
            "The CEL rail requires the 'common-expression-language' package (Python 3.11+). "
            "Install it with `pip install nemoguardrails[cel]`."
        ) from e
    return cel.compile(expression)


class CelOptions(RailConfigBaseModel):
    """CEL expressions to check for one tool."""

    model_config = ConfigDict(extra="forbid")

    expressions: List[str] = Field(
        ...,
        min_length=1,
        description="List of CEL expressions to evaluate, where any that evaluates to true is a match.",
    )

    @model_validator(mode="after")
    def compile_expressions(self) -> "CelOptions":
        """Compile the expressions at config load time, so a syntax error fails the load.

        The compiled programs are not kept: a cel.Program cannot be deep-copied or pickled, which the
        synchronous generate() and check() do to the config.
        """
        for i, expression in enumerate(self.expressions):
            try:
                compile_expression(expression)
            except ValueError as e:
                raise ValueError(f"Invalid CEL expression at index {i} ({expression!r}): {e}") from e
        return self


class CelConfig(RailConfigBaseModel):
    """Configuration for CEL expression checks."""

    model_config = ConfigDict(extra="forbid")

    tool_output: Dict[str, CelOptions] = Field(
        default_factory=dict,
        description="Per-tool CEL expressions to check against a tool call, keyed by tool name.",
    )
    tool_input: Dict[str, CelOptions] = Field(
        default_factory=dict,
        description="Per-tool CEL expressions to check against a tool result, keyed by tool name.",
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[CelConfig],
        field_info=rail_field(
            default_factory=CelConfig,
            description="Configuration for CEL expression checks.",
        ),
        exports={
            "CelConfig": CelConfig,
            "CelOptions": CelOptions,
        },
    )

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

from nemoguardrails.manifests.config_schema import (
    Field,
    PrivateAttr,
    RailConfigBaseModel,
    RailConfigSpec,
    model_validator,
    rail_field,
)

if TYPE_CHECKING:
    from cel import Program


def _compile(expression: str) -> "Program":
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

    expressions: List[str] = Field(
        default_factory=list,
        description="List of CEL expressions to evaluate, where any that evaluates to true is a match.",
    )

    _compiled_expressions: List["Program"] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def compile_expressions(self) -> "CelOptions":
        """Compile the expressions at config load time, so a syntax error fails the load."""
        compiled = []
        for i, expression in enumerate(self.expressions):
            try:
                compiled.append(_compile(expression))
            except ValueError as e:
                raise ValueError(f"Invalid CEL expression at index {i} ({expression!r}): {e}") from e
        object.__setattr__(self, "_compiled_expressions", compiled)
        return self

    @property
    def compiled_expressions(self) -> List["Program"]:
        """Return the compiled CEL programs."""
        return self._compiled_expressions


class CelConfig(RailConfigBaseModel):
    """Configuration for CEL expression checks."""

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

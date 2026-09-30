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

"""Tool-call safety rail for IORails.

Validates the tool calls a model emitted against the request's declared
``Toolset``: every call must name an allowed tool, and its arguments must
satisfy that tool's JSON Schema. The rail is local and model-free -- it runs
through :meth:`ToolRailAction._guarded`, so a malformed call or an unexpected
error fails closed (blocks) rather than propagating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.guardrails.tool_rail_action import ToolRailAction, violations_outcome
from nemoguardrails.guardrails.tool_schema import validate_arguments
from nemoguardrails.rails.llm.options import ToolViolation, ToolViolationType

if TYPE_CHECKING:
    from nemoguardrails.guardrails.tool_schema import Toolset
    from nemoguardrails.types import ToolCall


class ToolCallRailAction(ToolRailAction):
    """Check the model's tool calls against the declared toolset (allowlist + schema)."""

    action_name = "tool call validation"

    async def run(self, toolset: "Toolset", tool_calls: List["ToolCall"]) -> RailOutcome:
        """Block unless every tool call names an allowed tool with schema-valid arguments."""
        return self._guarded(lambda: self._validate(toolset, tool_calls))

    def _validate(self, toolset: "Toolset", tool_calls: List["ToolCall"]) -> RailOutcome:
        """Allowlist each call by name, then validate its arguments, reporting every call that fails."""
        violations: list[ToolViolation] = []
        for index, call in enumerate(tool_calls):
            violation = self._call_violation(toolset, index, call)
            if violation is not None:
                violations.append(violation)
        return violations_outcome(violations)

    def _call_violation(self, toolset: "Toolset", index: int, call: "ToolCall") -> "ToolViolation | None":
        """The first check *call* fails, as a violation at its *index*, or None when it passes."""
        # Hosted/server tools (e.g. web_search) have no function name; fall back to
        # call.type, mirroring Tool.key = name or type used when indexing the toolset.
        name = call.function.name or call.type
        tool = toolset.get(name)
        if tool is None:
            return ToolViolation(
                kind="tool_call",
                violation_type=ToolViolationType.TOOL_NOT_ALLOWED,
                reason=f"tool call '{name}' is not an allowed tool",
                tool_call_id=call.id or None,
                tool_name=name,
                index=index,
            )
        arguments = validate_arguments(tool, call.function.arguments)
        if arguments is None:
            return None
        return ToolViolation(
            kind="tool_call",
            violation_type=arguments.violation_type,
            reason=arguments.reason,
            tool_call_id=call.id or None,
            tool_name=name,
            index=index,
            argument_path=arguments.argument_path,
            schema_keyword=arguments.schema_keyword,
        )

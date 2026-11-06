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

"""Tool-result validation rail for IORails.

Structurally validates the tool results carried on an incoming request against
the tool calls the model previously made: every result must link to a prior
call by ``call_id``, must not name a different tool than that call, and must carry
well-formed content. This PR validates structure only -- there are no declared
response schemas yet. The rail is local and model-free; it runs through
:meth:`ToolRailAction._guarded`, so a malformed result or an unexpected error
fails closed (blocks) rather than propagating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.guardrails.guardrails_types import quoted_identity
from nemoguardrails.guardrails.tool_rail_action import ToolRailAction, violations_outcome
from nemoguardrails.rails.llm.options import ToolViolation, ToolViolationType

if TYPE_CHECKING:
    from nemoguardrails.guardrails.tool_schema import ToolResult
    from nemoguardrails.types import ToolCall


def _is_well_formed_content(content: object) -> bool:
    """Tool-result content is a string, or a list of content-block dicts.

    Matches the declared ``ToolResult.content`` type (``str | list[dict] | None``);
    a list of non-dict values (e.g. ``[1, 2, 3]``) is not well-formed.
    """
    if isinstance(content, str):
        return True
    return isinstance(content, list) and all(isinstance(block, dict) for block in content)


def _names_a_different_tool(result: "ToolResult", prior: "ToolCall") -> bool:
    """Whether the result's own name contradicts the tool its linked call named."""
    if not prior.function.name:
        return False
    if not result.name:
        # An OpenAI tool message carries no name; its call_id already binds it to exactly one call.
        return False
    return result.name != prior.function.name


def _result_violation(
    result: "ToolResult",
    violation_type: ToolViolationType,
    reason: str,
    prior: "ToolCall | None" = None,
) -> ToolViolation:
    """A violation for *result*, naming the tool from its linked call, never from the result itself."""
    return ToolViolation(
        kind="tool_result",
        violation_type=violation_type,
        reason=reason,
        tool_call_id=result.call_id or None,
        tool_name=(prior.function.name or prior.type) if prior is not None else None,
        index=result.message_index,
    )


class ToolResultRailAction(ToolRailAction):
    """Check incoming tool results link to a prior call and are structurally well-formed."""

    action_name = "tool result validation"

    async def run(self, tool_results: List["ToolResult"], prior_calls: List["ToolCall"]) -> RailOutcome:
        """Block unless every tool result links to a prior call with a consistent name and valid content."""
        return self._guarded(lambda: self._validate(tool_results, prior_calls))

    def _validate(self, tool_results: List["ToolResult"], prior_calls: List["ToolCall"]) -> RailOutcome:
        """Check call_id linkage, name consistency and content shape, reporting every result that fails."""
        prior_violations = self._duplicate_prior_call_violations(prior_calls)
        if prior_violations:
            # No result can be linked to an ambiguous call, so checking them would only add noise.
            return violations_outcome(prior_violations)
        calls_by_id = {call.id: call for call in prior_calls if call.id}
        return violations_outcome(self._result_violations(tool_results, calls_by_id))

    def _duplicate_prior_call_violations(self, prior_calls: List["ToolCall"]) -> list[ToolViolation]:
        """One violation per call id that several prior calls share, in first-seen order."""
        seen: set[str] = set()
        duplicated: list[str] = []
        for call in prior_calls:
            if not call.id:
                continue
            if call.id in seen and call.id not in duplicated:
                duplicated.append(call.id)
            seen.add(call.id)
        return [
            ToolViolation(
                kind="tool_result",
                violation_type=ToolViolationType.DUPLICATE_PRIOR_CALL_ID,
                reason=f"duplicate prior tool call id '{quoted_identity(call_id)}' makes tool-result linkage ambiguous",
                tool_call_id=call_id,
            )
            for call_id in duplicated
        ]

    def _result_violations(
        self, tool_results: List["ToolResult"], calls_by_id: "dict[str, ToolCall]"
    ) -> list[ToolViolation]:
        """The first check each result fails, in result order."""
        violations: list[ToolViolation] = []
        seen_ids: set[str] = set()
        for result in tool_results:
            violation = self._first_failed_check(result, calls_by_id, seen_ids)
            if violation is not None:
                violations.append(violation)
            if result.call_id:
                seen_ids.add(result.call_id)
        return violations

    def _first_failed_check(
        self, result: "ToolResult", calls_by_id: "dict[str, ToolCall]", seen_ids: set[str]
    ) -> "ToolViolation | None":
        """The first check *result* fails: its call_id, a duplicate, linkage, name, then content."""
        call_id = result.call_id
        if not call_id:
            return _result_violation(result, ToolViolationType.MISSING_CALL_ID, "tool result is missing a call_id")
        prior = calls_by_id.get(call_id)
        quoted_call_id = quoted_identity(call_id)
        if call_id in seen_ids:
            return _result_violation(
                result,
                ToolViolationType.DUPLICATE_RESULT,
                f"duplicate tool result for call_id '{quoted_call_id}': each tool call must have exactly one result",
                prior,
            )
        if prior is None:
            return _result_violation(
                result,
                ToolViolationType.UNKNOWN_CALL_ID,
                f"tool result for call_id '{quoted_call_id}' does not correspond to a prior tool call",
            )
        if _names_a_different_tool(result, prior):
            return _result_violation(
                result,
                ToolViolationType.NAME_MISMATCH,
                f"tool result name '{quoted_identity(result.name)}' does not match the called tool "
                f"'{quoted_identity(prior.function.name)}' for call_id '{quoted_call_id}'",
                prior,
            )
        if result.content is not None and not _is_well_formed_content(result.content):
            return _result_violation(
                result,
                ToolViolationType.MALFORMED_CONTENT,
                f"tool result for call_id '{quoted_call_id}' has malformed content",
                prior,
            )
        return None

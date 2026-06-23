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
call by ``call_id``, name a tool consistent with that call, and carry
well-formed content. This PR validates structure only -- there are no declared
response schemas yet. The rail is local and model-free; it runs through
:meth:`ToolRailAction._guarded`, so a malformed result or an unexpected error
fails closed (blocks) rather than propagating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.tool_rail_action import ToolRailAction

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


class ToolResultRailAction(ToolRailAction):
    """Check incoming tool results link to a prior call and are structurally well-formed."""

    action_name = "tool result validation"

    async def run(self, tool_results: List["ToolResult"], prior_calls: List["ToolCall"]) -> RailResult:
        """Block unless every tool result links to a prior call with a consistent name and valid content."""
        return self._guarded(lambda: self._validate(tool_results, prior_calls))

    def _validate(self, tool_results: List["ToolResult"], prior_calls: List["ToolCall"]) -> RailResult:
        """Check call_id linkage, name consistency, and content shape for each result."""
        calls_by_id = self._validate_prior_calls(prior_calls)
        if isinstance(calls_by_id, RailResult):
            return calls_by_id
        return self._validate_results(tool_results, calls_by_id)

    def _validate_prior_calls(self, prior_calls: List["ToolCall"]) -> "RailResult | dict[str, ToolCall]":
        """Build a call_id index from prior_calls; return a blocking RailResult on duplicate IDs."""
        calls_by_id: dict[str, "ToolCall"] = {}
        for call in prior_calls:
            if not call.id:
                continue
            if call.id in calls_by_id:
                return RailResult(
                    is_safe=False,
                    reason=f"duplicate prior tool call id '{call.id}' makes tool-result linkage ambiguous",
                )
            calls_by_id[call.id] = call
        return calls_by_id

    def _validate_results(self, tool_results: List["ToolResult"], calls_by_id: "dict[str, ToolCall]") -> RailResult:
        """Check each result links to a prior call with a consistent name and well-formed content."""
        rail_result = self._validate_tool_result_ids(tool_results)
        if rail_result:
            return rail_result

        for result in tool_results:
            rail_result = self._validate_result_call_id(result, calls_by_id)
            if rail_result:
                return rail_result

            prior = calls_by_id[result.call_id]  # type: ignore[index]
            rail_result = self._validate_result_name(result, prior)
            if rail_result:
                return rail_result

            rail_result = self._validate_result_content(result)
            if rail_result:
                return rail_result

        return RailResult(is_safe=True)

    def _validate_tool_result_ids(self, tool_results: List["ToolResult"]) -> "RailResult | None":
        """Return a blocking RailResult if any call_id appears more than once in the result list."""
        seen: set[str] = set()
        for result in tool_results:
            if not result.call_id:
                continue
            if result.call_id in seen:
                return RailResult(
                    is_safe=False,
                    reason=f"duplicate tool result for call_id '{result.call_id}': each tool call must have exactly one result",
                )
            seen.add(result.call_id)
        return None

    def _validate_result_call_id(self, result: "ToolResult", calls_by_id: "dict[str, ToolCall]") -> "RailResult | None":
        """Return a blocking RailResult if the result is missing a call_id or it has no prior call."""
        call_id = result.call_id
        if not call_id:
            return RailResult(is_safe=False, reason="tool result is missing a call_id")
        if calls_by_id.get(call_id) is None:
            return RailResult(
                is_safe=False,
                reason=f"tool result for call_id '{call_id}' does not correspond to a prior tool call",
            )
        return None

    def _validate_result_name(self, result: "ToolResult", prior: "ToolCall") -> "RailResult | None":
        """Return a blocking RailResult if the result name conflicts with the prior call's function name."""
        if result.name and prior.function.name and result.name != prior.function.name:
            return RailResult(
                is_safe=False,
                reason=(
                    f"tool result name '{result.name}' does not match the called tool "
                    f"'{prior.function.name}' for call_id '{result.call_id}'"
                ),
            )
        return None

    def _validate_result_content(self, result: "ToolResult") -> "RailResult | None":
        """Return a blocking RailResult if the result content is not a string or list of dicts."""
        if result.content is not None and not _is_well_formed_content(result.content):
            return RailResult(
                is_safe=False,
                reason=f"tool result for call_id '{result.call_id}' has malformed content",
            )
        return None

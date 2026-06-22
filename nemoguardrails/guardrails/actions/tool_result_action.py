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
        calls_by_id = {call.id: call for call in prior_calls if call.id}
        for result in tool_results:
            call_id = result.call_id
            if not call_id:
                return RailResult(is_safe=False, reason="tool result is missing a call_id")
            prior = calls_by_id.get(call_id)
            if prior is None:
                return RailResult(
                    is_safe=False,
                    reason=f"tool result for call_id '{call_id}' does not correspond to a prior tool call",
                )
            if result.name and prior.function.name and result.name != prior.function.name:
                return RailResult(
                    is_safe=False,
                    reason=(
                        f"tool result name '{result.name}' does not match the called tool "
                        f"'{prior.function.name}' for call_id '{call_id}'"
                    ),
                )
            if result.content is not None and not _is_well_formed_content(result.content):
                return RailResult(
                    is_safe=False,
                    reason=f"tool result for call_id '{call_id}' has malformed content",
                )
        return RailResult(is_safe=True)

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

"""Unit tests for ToolResultRailAction (call_id linkage + structural validation)."""

from typing import Any

import pytest

from nemoguardrails.guardrails.actions.tool_result_action import ToolResultRailAction
from nemoguardrails.guardrails.tool_schema import ToolResult
from nemoguardrails.rails.llm.options import ToolViolation, ToolViolationType
from nemoguardrails.types import ToolCall, ToolCallFunction
from tests.guardrails.tool_helpers import assert_outcome_blocked, violations_in

# Content of a type ToolResult does not declare, as a malformed client payload carries.
_MALFORMED_CONTENT: Any = {"unexpected": "shape"}


def _prior_calls() -> list:
    return [
        ToolCall(id="c1", function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"})),
        ToolCall(id="c2", function=ToolCallFunction(name="search", arguments={"q": "x"})),
    ]


def _prior_call(call_id: str, name: str) -> ToolCall:
    return ToolCall(id=call_id, function=ToolCallFunction(name=name, arguments={}))


def _result(
    call_id, name=None, content: "str | list[dict] | None" = "18C", message_index: "int | None" = None
) -> ToolResult:
    return ToolResult(call_id=call_id, name=name, content=content, message_index=message_index)


def _result_violation(
    violation_type: ToolViolationType,
    reason: str,
    *,
    tool_call_id: "str | None",
    index: "int | None" = 3,
    tool_name: "str | None" = None,
) -> ToolViolation:
    """A tool-result violation; results in these tests sit at message index 3 unless stated."""
    return ToolViolation(
        kind="tool_result",
        violation_type=violation_type,
        reason=reason,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        index=index,
    )


class TestToolResultRailAction:
    @pytest.mark.asyncio
    async def test_linked_result_with_matching_name_is_safe(self):
        result = await ToolResultRailAction().run([_result("c1", name="get_weather")], _prior_calls())
        assert result.is_blocked is False

    @pytest.mark.asyncio
    async def test_result_without_name_is_identified_by_its_call_id(self):
        """A result without a name is safe when its call_id links it to a prior call."""
        result = await ToolResultRailAction().run([_result("c2")], _prior_calls())
        assert result.is_blocked is False

    @pytest.mark.asyncio
    async def test_prior_call_without_a_name_accepts_a_named_result(self):
        """A prior call with no function name leaves nothing for the result's name to contradict."""
        result = await ToolResultRailAction().run([_result("c1", name="get_weather")], [_prior_call("c1", "")])
        assert result.is_blocked is False

    @pytest.mark.asyncio
    async def test_list_content_is_well_formed(self):
        result = await ToolResultRailAction().run(
            [_result("c1", name="get_weather", content=[{"type": "text", "text": "18C"}])], _prior_calls()
        )
        assert result.is_blocked is False

    @pytest.mark.asyncio
    async def test_none_content_is_well_formed(self):
        """A result with no content is not malformed."""
        result = await ToolResultRailAction().run([_result("c1", name="get_weather", content=None)], _prior_calls())
        assert result.is_blocked is False

    @pytest.mark.asyncio
    async def test_empty_results_is_safe(self):
        result = await ToolResultRailAction().run([], _prior_calls())
        assert result.is_blocked is False

    @pytest.mark.asyncio
    async def test_missing_call_id_is_blocked(self):
        """A result with no call_id blocks with a ``missing_call_id`` violation at its message index."""
        result = await ToolResultRailAction().run([_result("", message_index=3)], _prior_calls())
        assert_outcome_blocked(result, "missing a call_id")
        assert violations_in(result) == [
            _result_violation(ToolViolationType.MISSING_CALL_ID, "tool result is missing a call_id", tool_call_id=None)
        ]

    @pytest.mark.asyncio
    async def test_unlinked_call_id_is_blocked(self):
        """A result whose call_id matches no prior call blocks with an ``unknown_call_id`` violation."""
        result = await ToolResultRailAction().run([_result("c9", message_index=3)], _prior_calls())
        assert_outcome_blocked(result, "c9", "does not correspond to a prior tool call")
        assert violations_in(result) == [
            _result_violation(
                ToolViolationType.UNKNOWN_CALL_ID,
                "tool result for call_id 'c9' does not correspond to a prior tool call",
                tool_call_id="c9",
            )
        ]

    @pytest.mark.asyncio
    async def test_name_mismatch_is_blocked(self):
        """A result naming a different tool than its linked call blocks, and the violation names the called tool."""
        result = await ToolResultRailAction().run([_result("c1", name="search", message_index=3)], _prior_calls())
        assert_outcome_blocked(result, "does not match the called tool", "get_weather")
        assert violations_in(result) == [
            _result_violation(
                ToolViolationType.NAME_MISMATCH,
                "tool result name 'search' does not match the called tool 'get_weather' for call_id 'c1'",
                tool_call_id="c1",
                tool_name="get_weather",
            )
        ]

    @pytest.mark.asyncio
    async def test_malformed_content_is_blocked(self):
        """Content that is neither a string nor a list of blocks gives a ``malformed_content`` violation."""
        malformed = _result("c1", name="get_weather", content=_MALFORMED_CONTENT, message_index=3)
        result = await ToolResultRailAction().run([malformed], _prior_calls())
        assert_outcome_blocked(result, "malformed content")
        assert violations_in(result) == [
            _result_violation(
                ToolViolationType.MALFORMED_CONTENT,
                "tool result for call_id 'c1' has malformed content",
                tool_call_id="c1",
                tool_name="get_weather",
            )
        ]

    @pytest.mark.asyncio
    async def test_list_of_non_dicts_is_blocked(self):
        result = await ToolResultRailAction().run(
            [_result("c1", name="get_weather", content=[1, 2, 3])],  # type: ignore[arg-type]
            _prior_calls(),
        )
        assert_outcome_blocked(result, "malformed content")

    @pytest.mark.asyncio
    async def test_empty_content_list_is_well_formed(self):
        result = await ToolResultRailAction().run([_result("c1", name="get_weather", content=[])], _prior_calls())
        assert result.is_blocked is False

    @pytest.mark.asyncio
    async def test_every_bad_result_is_reported_in_order(self):
        """Each bad result gets its own violation, in list order, and the reason is the first one's."""
        results = [
            _result("c9", message_index=2),
            _result("c1", name="get_weather", message_index=3),
            _result("", message_index=4),
        ]
        result = await ToolResultRailAction().run(results, _prior_calls())
        assert [(v.violation_type, v.index) for v in violations_in(result)] == [
            (ToolViolationType.UNKNOWN_CALL_ID, 2),
            (ToolViolationType.MISSING_CALL_ID, 4),
        ]
        assert result.reason == "tool result for call_id 'c9' does not correspond to a prior tool call"

    @pytest.mark.asyncio
    async def test_a_result_reports_only_its_first_failing_check(self):
        """A result that fails both the name and the content check reports only ``name_mismatch``."""
        bad = _result("c1", name="search", content=_MALFORMED_CONTENT, message_index=3)
        result = await ToolResultRailAction().run([bad], _prior_calls())
        assert [v.violation_type for v in violations_in(result)] == [ToolViolationType.NAME_MISMATCH]

    @pytest.mark.asyncio
    async def test_results_without_call_ids_are_each_missing_not_duplicates(self):
        """Two results with no call_id each report ``missing_call_id``; neither counts as a duplicate."""
        results = [_result(None, message_index=2), _result(None, message_index=3)]
        result = await ToolResultRailAction().run(results, _prior_calls())
        assert [(v.violation_type, v.index) for v in violations_in(result)] == [
            (ToolViolationType.MISSING_CALL_ID, 2),
            (ToolViolationType.MISSING_CALL_ID, 3),
        ]

    @pytest.mark.asyncio
    async def test_duplicate_prior_call_id_is_blocked(self):
        """Two prior calls sharing an id block with a ``duplicate_prior_call_id`` violation for that id."""
        prior = [_prior_call("c1", "get_weather"), _prior_call("c1", "search")]
        result = await ToolResultRailAction().run([_result("c1")], prior)
        assert_outcome_blocked(result, "duplicate prior tool call id", "c1")
        assert violations_in(result) == [
            _result_violation(
                ToolViolationType.DUPLICATE_PRIOR_CALL_ID,
                "duplicate prior tool call id 'c1' makes tool-result linkage ambiguous",
                tool_call_id="c1",
                index=None,
            )
        ]

    @pytest.mark.asyncio
    async def test_duplicate_prior_call_ids_are_each_reported_and_skip_result_checks(self):
        """Every duplicated prior id is reported once, and no result is checked against the ambiguous calls."""
        prior = [
            _prior_call("c1", "get_weather"),
            _prior_call("c1", "search"),
            _prior_call("c2", "get_weather"),
            _prior_call("c2", "search"),
            _prior_call("c2", "list_files"),
        ]
        result = await ToolResultRailAction().run([_result("c9", message_index=6)], prior)
        assert [(v.violation_type, v.tool_call_id) for v in violations_in(result)] == [
            (ToolViolationType.DUPLICATE_PRIOR_CALL_ID, "c1"),
            (ToolViolationType.DUPLICATE_PRIOR_CALL_ID, "c2"),
        ]

    @pytest.mark.asyncio
    async def test_prior_call_with_empty_id_is_skipped(self):
        prior = [ToolCall(id="", function=ToolCallFunction(name="get_weather", arguments={}))]
        result = await ToolResultRailAction().run([], prior)
        assert result.is_blocked is False

    @pytest.mark.asyncio
    async def test_duplicate_result_ids_are_blocked(self):
        """The second result for a call id blocks with a ``duplicate_result`` violation at its own message index."""
        results = [
            _result("c1", name="get_weather", message_index=2),
            _result("c1", name="get_weather", message_index=3),
        ]
        result = await ToolResultRailAction().run(results, _prior_calls())
        assert_outcome_blocked(result, "duplicate tool result", "c1")
        assert violations_in(result) == [
            _result_violation(
                ToolViolationType.DUPLICATE_RESULT,
                "duplicate tool result for call_id 'c1': each tool call must have exactly one result",
                tool_call_id="c1",
                tool_name="get_weather",
            )
        ]

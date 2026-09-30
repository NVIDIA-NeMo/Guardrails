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
from nemoguardrails.rails.llm.options import ToolViolationType
from nemoguardrails.types import ToolCall, ToolCallFunction
from tests.guardrails.tool_helpers import result_violation, violations_in

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


# Content types ToolResult does not declare, as a malformed client payload carries them.
_NON_DICT_BLOCKS: Any = [1, 2, 3]

# Case id -> (results, prior calls) the validator accepts.
_ACCEPTED = {
    "linked_result_with_matching_name": ([_result("c1", name="get_weather")], _prior_calls()),
    "result_without_name_linked_by_call_id": ([_result("c2")], _prior_calls()),
    "prior_call_without_a_name": ([_result("c1", name="get_weather")], [_prior_call("c1", "")]),
    "list_of_content_blocks": (
        [_result("c1", name="get_weather", content=[{"type": "text", "text": "18C"}])],
        _prior_calls(),
    ),
    "no_content": ([_result("c1", name="get_weather", content=None)], _prior_calls()),
    "empty_content_list": ([_result("c1", name="get_weather", content=[])], _prior_calls()),
    "no_results": ([], _prior_calls()),
    "prior_call_with_empty_id": ([], [_prior_call("", "get_weather")]),
}

# Case id -> (results, prior calls, the one violation the validator reports).
_REJECTED = {
    "missing_call_id": (
        [_result("", message_index=3)],
        _prior_calls(),
        result_violation("missing_call_id", "tool result is missing a call_id", index=3),
    ),
    "unknown_call_id": (
        [_result("c9", message_index=3)],
        _prior_calls(),
        result_violation(
            "unknown_call_id",
            "tool result for call_id 'c9' does not correspond to a prior tool call",
            tool_call_id="c9",
            index=3,
        ),
    ),
    "name_mismatch": (
        [_result("c1", name="search", message_index=3)],
        _prior_calls(),
        result_violation(
            "name_mismatch",
            "tool result name 'search' does not match the called tool 'get_weather' for call_id 'c1'",
            tool_call_id="c1",
            tool_name="get_weather",
            index=3,
        ),
    ),
    "malformed_content": (
        [_result("c1", name="get_weather", content=_MALFORMED_CONTENT, message_index=3)],
        _prior_calls(),
        result_violation(
            "malformed_content",
            "tool result for call_id 'c1' has malformed content",
            tool_call_id="c1",
            tool_name="get_weather",
            index=3,
        ),
    ),
    "list_of_non_dicts": (
        [_result("c1", name="get_weather", content=_NON_DICT_BLOCKS)],
        _prior_calls(),
        result_violation(
            "malformed_content",
            "tool result for call_id 'c1' has malformed content",
            tool_call_id="c1",
            tool_name="get_weather",
        ),
    ),
    "duplicate_prior_call_id": (
        [_result("c1")],
        [_prior_call("c1", "get_weather"), _prior_call("c1", "search")],
        result_violation(
            "duplicate_prior_call_id",
            "duplicate prior tool call id 'c1' makes tool-result linkage ambiguous",
            tool_call_id="c1",
        ),
    ),
    "duplicate_result": (
        [_result("c1", name="get_weather", message_index=2), _result("c1", name="get_weather", message_index=3)],
        _prior_calls(),
        result_violation(
            "duplicate_result",
            "duplicate tool result for call_id 'c1': each tool call must have exactly one result",
            tool_call_id="c1",
            tool_name="get_weather",
            index=3,
        ),
    ),
}


class TestToolResultRailAction:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("results", "prior"), list(_ACCEPTED.values()), ids=list(_ACCEPTED))
    async def test_well_formed_linked_results_are_safe(self, results, prior):
        """Results that link to one prior call, don't contradict its name and carry well-formed content pass."""
        outcome = await ToolResultRailAction().run(results, prior)
        assert outcome.is_blocked is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("results", "prior", "violation"), list(_REJECTED.values()), ids=list(_REJECTED))
    async def test_a_bad_result_is_blocked_with_its_violation(self, results, prior, violation):
        """A result that breaks one rule blocks, reporting that rule's violation at the result's message index."""
        outcome = await ToolResultRailAction().run(results, prior)
        assert outcome.is_blocked
        assert outcome.reason == violation.reason
        assert violations_in(outcome) == [violation]

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

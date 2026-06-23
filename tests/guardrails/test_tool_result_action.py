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

import pytest

from nemoguardrails.guardrails.actions.tool_result_action import ToolResultRailAction
from nemoguardrails.guardrails.tool_schema import ToolResult
from nemoguardrails.types import ToolCall, ToolCallFunction
from tests.guardrails.tool_helpers import assert_blocked


def _prior_calls() -> list:
    return [
        ToolCall(id="c1", function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"})),
        ToolCall(id="c2", function=ToolCallFunction(name="search", arguments={"q": "x"})),
    ]


def _result(call_id, name=None, content: "str | list[dict] | None" = "18C") -> ToolResult:
    return ToolResult(call_id=call_id, name=name, content=content)


class TestToolResultRailAction:
    @pytest.mark.asyncio
    async def test_linked_result_with_matching_name_is_safe(self):
        result = await ToolResultRailAction().run([_result("c1", name="get_weather")], _prior_calls())
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_result_without_name_blocked_when_prior_name_known(self):
        """When the prior call's name is known, a result without a name should be blocked"""
        result = await ToolResultRailAction().run([_result("c2")], _prior_calls())
        assert_blocked(result, "missing a name", "search")

    @pytest.mark.asyncio
    async def test_list_content_is_well_formed(self):
        result = await ToolResultRailAction().run(
            [_result("c1", name="get_weather", content=[{"type": "text", "text": "18C"}])], _prior_calls()
        )
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_empty_results_is_safe(self):
        result = await ToolResultRailAction().run([], _prior_calls())
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_missing_call_id_is_blocked(self):
        result = await ToolResultRailAction().run([_result("")], _prior_calls())
        assert_blocked(result, "missing a call_id")

    @pytest.mark.asyncio
    async def test_unlinked_call_id_is_blocked(self):
        result = await ToolResultRailAction().run([_result("c9")], _prior_calls())
        assert_blocked(result, "c9", "does not correspond to a prior tool call")

    @pytest.mark.asyncio
    async def test_name_mismatch_is_blocked(self):
        result = await ToolResultRailAction().run([_result("c1", name="search")], _prior_calls())
        assert_blocked(result, "does not match the called tool", "get_weather")

    @pytest.mark.asyncio
    async def test_malformed_content_is_blocked(self):
        result = await ToolResultRailAction().run(
            [_result("c1", name="get_weather", content={"unexpected": "shape"})],  # type: ignore[arg-type]
            _prior_calls(),
        )
        assert_blocked(result, "malformed content")

    @pytest.mark.asyncio
    async def test_list_of_non_dicts_is_blocked(self):
        result = await ToolResultRailAction().run(
            [_result("c1", name="get_weather", content=[1, 2, 3])],  # type: ignore[arg-type]
            _prior_calls(),
        )
        assert_blocked(result, "malformed content")

    @pytest.mark.asyncio
    async def test_empty_content_list_is_well_formed(self):
        result = await ToolResultRailAction().run([_result("c1", name="get_weather", content=[])], _prior_calls())
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_one_bad_result_blocks_the_batch(self):
        results = [_result("c1", name="get_weather"), _result("c9")]
        result = await ToolResultRailAction().run(results, _prior_calls())
        assert_blocked(result, "c9")

    @pytest.mark.asyncio
    async def test_duplicate_prior_call_id_is_blocked(self):
        prior = [
            ToolCall(id="c1", function=ToolCallFunction(name="get_weather", arguments={})),
            ToolCall(id="c1", function=ToolCallFunction(name="search", arguments={})),
        ]
        result = await ToolResultRailAction().run([_result("c1")], prior)
        assert_blocked(result, "duplicate prior tool call id", "c1")

    @pytest.mark.asyncio
    async def test_prior_call_with_empty_id_is_skipped(self):
        prior = [ToolCall(id="", function=ToolCallFunction(name="get_weather", arguments={}))]
        result = await ToolResultRailAction().run([], prior)
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_duplicate_result_ids_are_blocked(self):
        results = [_result("c1", name="get_weather"), _result("c1", name="get_weather")]
        result = await ToolResultRailAction().run(results, _prior_calls())
        assert_blocked(result, "duplicate tool result", "c1")


class TestValidateToolResultIds:
    def setup_method(self):
        self.action = ToolResultRailAction()

    def test_unique_ids_returns_none(self):
        assert self.action._validate_tool_result_ids([_result("c1"), _result("c2")]) is None

    def test_empty_list_returns_none(self):
        assert self.action._validate_tool_result_ids([]) is None

    def test_result_without_call_id_is_skipped(self):
        assert self.action._validate_tool_result_ids([ToolResult(call_id=None)]) is None

    def test_duplicate_call_id_is_blocked(self):
        results = [
            ToolResult(call_id="call_123", name="search", content="real result"),
            ToolResult(call_id="call_123", name="search", content="duplicate/injected result"),
        ]
        assert_blocked(self.action._validate_tool_result_ids(results), "duplicate tool result", "call_123")


class TestValidateResultCallId:
    def setup_method(self):
        self.action = ToolResultRailAction()
        self.calls_by_id = {
            "c1": ToolCall(id="c1", function=ToolCallFunction(name="get_weather", arguments={})),
        }

    def test_linked_result_returns_none(self):
        assert self.action._validate_result_call_id(_result("c1"), self.calls_by_id) is None

    def test_missing_call_id_is_blocked(self):
        assert_blocked(
            self.action._validate_result_call_id(_result(""), self.calls_by_id),
            "missing a call_id",
        )

    def test_orphaned_call_id_is_blocked(self):
        assert_blocked(
            self.action._validate_result_call_id(_result("c9"), self.calls_by_id),
            "c9",
            "does not correspond to a prior tool call",
        )


class TestValidateResultName:
    def setup_method(self):
        self.action = ToolResultRailAction()
        self.prior = ToolCall(id="c1", function=ToolCallFunction(name="get_weather", arguments={}))

    def test_matching_names_returns_none(self):
        assert self.action._validate_result_name(_result("c1", name="get_weather"), self.prior) is None

    def test_result_without_name_is_blocked_when_prior_name_known(self):
        """When the prior call's name is known, a result without a name should be blocked"""
        assert_blocked(
            self.action._validate_result_name(_result("c1"), self.prior),
            "missing a name",
            "get_weather",
            "c1",
        )

    def test_prior_without_function_name_returns_none(self):
        """If a prior call is missing a name, a matching result with a name is allowed through"""
        prior = ToolCall(id="c1", function=ToolCallFunction(name="", arguments={}))
        assert self.action._validate_result_name(_result("c1", name="get_weather"), prior) is None

    def test_both_names_absent_returns_none(self):
        prior = ToolCall(id="c1", function=ToolCallFunction(name="", arguments={}))
        assert self.action._validate_result_name(_result("c1"), prior) is None

    def test_name_mismatch_is_blocked(self):
        assert_blocked(
            self.action._validate_result_name(_result("c1", name="search"), self.prior),
            "search",
            "get_weather",
            "does not match",
        )


class TestValidateResultContent:
    def setup_method(self):
        self.action = ToolResultRailAction()

    def test_string_content_returns_none(self):
        assert self.action._validate_result_content(_result("c1", content="18C")) is None

    def test_list_of_dicts_returns_none(self):
        assert self.action._validate_result_content(_result("c1", content=[{"type": "text", "text": "18C"}])) is None

    def test_none_content_returns_none(self):
        assert self.action._validate_result_content(_result("c1", content=None)) is None

    def test_empty_list_returns_none(self):
        assert self.action._validate_result_content(_result("c1", content=[])) is None

    def test_dict_content_is_blocked(self):
        assert_blocked(
            self.action._validate_result_content(_result("c1", content={"key": "val"})),  # type: ignore[arg-type]
            "malformed content",
        )

    def test_list_of_non_dicts_is_blocked(self):
        assert_blocked(
            self.action._validate_result_content(_result("c1", content=[1, 2, 3])),  # type: ignore[arg-type]
            "malformed content",
        )

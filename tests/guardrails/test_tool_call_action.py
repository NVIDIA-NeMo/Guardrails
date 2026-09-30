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

"""Unit tests for ToolCallRailAction (allowlist + argument-schema validation)."""

import pytest

from nemoguardrails.guardrails.actions.tool_call_action import ToolCallRailAction
from nemoguardrails.guardrails.tool_schema import Tool, Toolset
from nemoguardrails.rails.llm.options import ToolViolation, ToolViolationType
from nemoguardrails.types import ToolCall, ToolCallFunction
from tests.guardrails.tool_helpers import WEATHER_SCHEMA, assert_outcome_blocked, violations_in


def _toolset() -> Toolset:
    return Toolset(tools=[Tool(name="get_weather", arguments_schema=WEATHER_SCHEMA), Tool(name="ping")])


def _call(name: str, arguments: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, function=ToolCallFunction(name=name, arguments=arguments))


def _call_violation(
    violation_type: ToolViolationType,
    reason: str,
    *,
    tool_name: str,
    argument_path: str | None = None,
    schema_keyword: str | None = None,
) -> ToolViolation:
    """A violation for the only call in the list, ``c1`` at index 0."""
    return ToolViolation(
        kind="tool_call",
        violation_type=violation_type,
        reason=reason,
        tool_call_id="c1",
        tool_name=tool_name,
        index=0,
        argument_path=argument_path,
        schema_keyword=schema_keyword,
    )


class TestToolCallRailAction:
    @pytest.mark.asyncio
    async def test_allowed_call_with_valid_arguments_is_safe(self):
        result = await ToolCallRailAction().run(_toolset(), [_call("get_weather", {"city": "Paris"})])
        assert result.is_blocked is False

    @pytest.mark.asyncio
    async def test_no_parameter_function_tool_with_arguments_is_blocked(self):
        """A function tool that declares no parameters accepts no arguments, so a call supplying any is blocked."""
        result = await ToolCallRailAction().run(_toolset(), [_call("ping", {"anything": 1})])
        assert_outcome_blocked(result, "ping", "no arguments")
        assert violations_in(result) == [
            _call_violation(
                ToolViolationType.UNEXPECTED_ARGUMENTS,
                "tool 'ping' accepts no arguments but the call supplied: ['anything']",
                tool_name="ping",
            )
        ]

    @pytest.mark.asyncio
    async def test_no_parameter_function_tool_without_arguments_is_safe(self):
        """A no-parameter function tool called with no arguments passes."""
        result = await ToolCallRailAction().run(_toolset(), [_call("ping", {})])
        assert result.is_blocked is False

    @pytest.mark.asyncio
    async def test_undeclared_tool_is_blocked(self):
        """A call to an undeclared tool blocks with a ``tool_not_allowed`` violation naming the call."""
        result = await ToolCallRailAction().run(_toolset(), [_call("rm_rf", {})])
        assert_outcome_blocked(result, "rm_rf", "not an allowed tool")
        assert violations_in(result) == [
            _call_violation(
                ToolViolationType.TOOL_NOT_ALLOWED, "tool call 'rm_rf' is not an allowed tool", tool_name="rm_rf"
            )
        ]

    @pytest.mark.asyncio
    async def test_invalid_arguments_are_blocked(self):
        """Schema-invalid arguments block with an ``arguments_invalid`` violation pointing at the failing argument."""
        result = await ToolCallRailAction().run(_toolset(), [_call("get_weather", {})])
        assert_outcome_blocked(result, "get_weather")
        assert violations_in(result) == [
            _call_violation(
                ToolViolationType.ARGUMENTS_INVALID,
                "arguments for tool 'get_weather' do not match its schema: 'required' failed at '/city'",
                tool_name="get_weather",
                argument_path="/city",
                schema_keyword="required",
            )
        ]

    @pytest.mark.asyncio
    async def test_empty_tool_calls_is_safe(self):
        result = await ToolCallRailAction().run(_toolset(), [])
        assert result.is_blocked is False

    @pytest.mark.asyncio
    async def test_every_bad_call_is_reported_in_order(self):
        """Each bad call in the list gets its own violation, in list order, and the reason is the first one's."""
        calls = [
            _call("rm_rf", {}, call_id="c1"),
            _call("get_weather", {"city": "Paris"}, call_id="c2"),
            _call("ping", {"anything": 1}, call_id="c3"),
        ]
        result = await ToolCallRailAction().run(_toolset(), calls)
        assert [(v.violation_type, v.index, v.tool_call_id) for v in violations_in(result)] == [
            (ToolViolationType.TOOL_NOT_ALLOWED, 0, "c1"),
            (ToolViolationType.UNEXPECTED_ARGUMENTS, 2, "c3"),
        ]
        assert result.reason == "tool call 'rm_rf' is not an allowed tool"

    @pytest.mark.asyncio
    async def test_hosted_tool_matched_by_type_when_name_is_empty(self):
        toolset = Toolset(tools=[Tool(name=None, type="web_search")])
        call = ToolCall(id="c1", type="web_search", function=ToolCallFunction(name="", arguments={}))
        result = await ToolCallRailAction().run(toolset, [call])
        assert result.is_blocked is False

    @pytest.mark.asyncio
    async def test_undeclared_hosted_tool_type_is_blocked(self):
        toolset = Toolset(tools=[Tool(name=None, type="web_search")])
        call = ToolCall(id="c1", type="code_interpreter", function=ToolCallFunction(name="", arguments={}))
        result = await ToolCallRailAction().run(toolset, [call])
        assert_outcome_blocked(result, "code_interpreter", "not an allowed tool")

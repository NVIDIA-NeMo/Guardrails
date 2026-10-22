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

from typing import Any

import pytest

from nemoguardrails.guardrails.actions.tool_call_action import ToolCallRailAction
from nemoguardrails.guardrails.tool_policy import ToolPolicy
from nemoguardrails.guardrails.tool_schema import Tool, Toolset
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import ToolViolationType
from nemoguardrails.types import ToolCall, ToolCallFunction
from tests.guardrails.tool_helpers import WEATHER_SCHEMA, assert_outcome_blocked, call_violation, violations_in


def _toolset() -> Toolset:
    return Toolset(tools=[Tool(name="get_weather", arguments_schema=WEATHER_SCHEMA), Tool(name="ping")])


def _call(name: str, arguments: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, function=ToolCallFunction(name=name, arguments=arguments))


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
            call_violation(
                "unexpected_arguments",
                "tool 'ping' accepts no arguments but the call supplied 1 argument",
                tool_call_id="c1",
                tool_name="ping",
                index=0,
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
            call_violation(
                "tool_not_allowed",
                "tool call 'rm_rf' is not an allowed tool",
                tool_call_id="c1",
                tool_name="rm_rf",
                index=0,
            )
        ]

    @pytest.mark.asyncio
    async def test_undeclared_tool_name_is_capped(self):
        """An undeclared name, which the model made up, is cut to 64 characters in the reason and the violation."""
        capped = "x" * 64 + "..."

        result = await ToolCallRailAction().run(_toolset(), [_call("x" * 200, {})])

        assert result.reason == f"tool call '{capped}' is not an allowed tool"
        assert [violation.tool_name for violation in violations_in(result)] == [capped]

    @pytest.mark.asyncio
    async def test_invalid_arguments_are_blocked(self):
        """Schema-invalid arguments block with an ``arguments_invalid`` violation pointing at the failing argument."""
        result = await ToolCallRailAction().run(_toolset(), [_call("get_weather", {})])
        assert_outcome_blocked(result, "get_weather")
        assert violations_in(result) == [
            call_violation(
                "arguments_invalid",
                "arguments for tool 'get_weather' do not match its schema: 'required' failed at '/city'",
                tool_call_id="c1",
                tool_name="get_weather",
                index=0,
                argument_path="/city",
                schema_keyword="required",
            )
        ]

    @pytest.mark.asyncio
    async def test_undeclared_call_with_a_non_string_id_is_still_reported(self):
        """A call whose id is not a string blocks as ``tool_not_allowed`` with no id, rather than breaking the rail."""
        non_string_id: Any = 5
        call = ToolCall(id=non_string_id, function=ToolCallFunction(name="rm_rf", arguments={}))
        result = await ToolCallRailAction().run(_toolset(), [call])
        assert violations_in(result) == [
            call_violation("tool_not_allowed", "tool call 'rm_rf' is not an allowed tool", tool_name="rm_rf", index=0)
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


def _policy(**tool_lists: Any) -> ToolPolicy:
    """The policy a config with *tool_lists* in rails.tool_call sets."""
    config = RailsConfig.from_content(
        config={"models": [], "rails": {"tool_call": {"flows": ["tool call validation"], **tool_lists}}}
    )
    return ToolPolicy.from_config(config.rails.tool_call)


async def _violations(policy: ToolPolicy, toolset: Toolset, call: ToolCall) -> list:
    """The violations `tool call validation` with *policy* reports for *call*, with *toolset* offered."""
    return violations_in(await ToolCallRailAction(policy=policy).run(toolset, [call]))


def _offering(name: str) -> Toolset:
    """A toolset offering *name*, taking any arguments, so only the lists can deny a call to it."""
    return Toolset(tools=[Tool(name=name, arguments_schema={"type": "object", "additionalProperties": True})])


class TestToolLists:
    @pytest.mark.asyncio
    async def test_tool_blocked_by_name_is_tool_blocked(self):
        """A call to a tool blocked_tools names bare is a `tool_blocked` violation."""
        violations = await _violations(_policy(blocked_tools=["Bash"]), Toolset(), _call("Bash", {"command": "ls"}))

        assert violations == [
            call_violation(
                "tool_blocked", "tool call 'Bash' is a blocked tool", tool_call_id="c1", tool_name="Bash", index=0
            )
        ]

    @pytest.mark.asyncio
    async def test_call_passing_a_blocked_argument_is_argument_blocked(self):
        """A call that passes an argument a blocked_tools entry lists is `argument_blocked`, pointing at it."""
        policy = _policy(blocked_tools=[{"name": "Bash", "arguments": ["sandbox", "timeout"]}])

        violations = await _violations(policy, Toolset(), _call("Bash", {"command": "ls", "timeout": 5}))

        assert violations == [
            call_violation(
                "argument_blocked",
                "tool call 'Bash' passes argument 'timeout', which is blocked",
                tool_call_id="c1",
                tool_name="Bash",
                index=0,
                argument_path="/timeout",
            )
        ]

    @pytest.mark.asyncio
    async def test_call_without_a_blocked_argument_is_allowed(self):
        """A call to a tool with a blocked_tools entry passes when it passes none of the entry's arguments."""
        policy = _policy(blocked_tools=[{"name": "Bash", "arguments": ["sandbox"]}])

        assert await _violations(policy, _offering("Bash"), _call("Bash", {"command": "ls"})) == []

    @pytest.mark.asyncio
    async def test_argument_outside_an_allowed_entry_is_argument_not_allowed(self):
        """Arguments an allowed_tools entry doesn't list are `argument_not_allowed`, counted rather than named."""
        policy = _policy(allowed_tools=[{"name": "search", "arguments": ["query"]}])

        violations = await _violations(
            policy, Toolset(), _call("search", {"query": "q", "token": "secret", "limit": 3})
        )

        assert violations == [
            call_violation(
                "argument_not_allowed",
                "tool call 'search' passes 2 arguments its allowed_tools entry doesn't list",
                tool_call_id="c1",
                tool_name="search",
                index=0,
            )
        ]

    @pytest.mark.asyncio
    async def test_unlisted_tool_is_allowed_by_default(self):
        """A call to a tool allowed_tools doesn't name passes while allow_unlisted_tools keeps its default."""
        policy = _policy(allowed_tools=["Read"])

        assert await _violations(policy, _offering("Bash"), _call("Bash", {"command": "ls"})) == []

    @pytest.mark.asyncio
    async def test_unlisted_tool_is_tool_not_allowed_when_unlisted_tools_are_denied(self):
        """With allow_unlisted_tools false, a call to a tool allowed_tools doesn't name is `tool_not_allowed`."""
        policy = _policy(allowed_tools=["Read"], allow_unlisted_tools=False)

        violations = await _violations(policy, Toolset(), _call("WebFetch", {"url": "https://example.com"}))

        assert violations == [
            call_violation(
                "tool_not_allowed",
                "tool call 'WebFetch' is not an allowed tool",
                tool_call_id="c1",
                tool_name="WebFetch",
                index=0,
            )
        ]

    @pytest.mark.asyncio
    async def test_blocked_argument_entry_leaves_the_tool_unlisted(self):
        """A tool only blocked_tools names, by an argument entry, is denied while unlisted tools are denied."""
        policy = _policy(
            allowed_tools=["Read"],
            blocked_tools=[{"name": "Bash", "arguments": ["sandbox"]}],
            allow_unlisted_tools=False,
        )

        violations = await _violations(policy, Toolset(), _call("Bash", {"command": "ls"}))

        assert [violation.violation_type for violation in violations] == [ToolViolationType.TOOL_NOT_ALLOWED]

    @pytest.mark.asyncio
    async def test_lists_decide_before_the_offered_tools(self):
        """A call to an offered tool that blocked_tools blocks is `tool_blocked`."""
        violations = await _violations(_policy(blocked_tools=["get_weather"]), _toolset(), _call("get_weather", {}))

        assert [violation.violation_type for violation in violations] == [ToolViolationType.TOOL_BLOCKED]

    @pytest.mark.asyncio
    async def test_allowed_tool_that_was_not_offered_is_tool_not_allowed(self):
        """When tools were offered, a call to a tool the lists allow but the request didn't offer is denied."""
        violations = await _violations(_policy(allowed_tools=["Bash"]), _toolset(), _call("Bash", {"command": "ls"}))

        assert violations == [
            call_violation(
                "tool_not_allowed",
                "tool call 'Bash' is not an allowed tool",
                tool_call_id="c1",
                tool_name="Bash",
                index=0,
            )
        ]

    @pytest.mark.asyncio
    async def test_offered_schema_still_checks_what_the_lists_allow(self):
        """When tools were offered, the offered schema checks the arguments of a call the lists allow."""
        policy = _policy(allowed_tools=["get_weather"])

        violations = await _violations(policy, _toolset(), _call("get_weather", {"city": 5}))

        assert [violation.violation_type for violation in violations] == [ToolViolationType.ARGUMENTS_INVALID]

    @pytest.mark.asyncio
    async def test_with_nothing_offered_a_call_the_lists_allow_is_tool_not_allowed(self):
        """With no tools offered, a call the lists allow is still denied: the lists narrow the offered tools."""
        policy = _policy(allowed_tools=["Bash"], allow_unlisted_tools=False)

        violations = await _violations(policy, Toolset(), _call("Bash", {"command": "ls"}))

        assert [violation.violation_type for violation in violations] == [ToolViolationType.TOOL_NOT_ALLOWED]

    @pytest.mark.asyncio
    async def test_with_no_lists_and_nothing_offered_every_call_is_denied(self):
        """A config without the lists, given no offered tools, denies every call as before the lists existed."""
        violations = await _violations(_policy(), Toolset(), _call("Bash", {"command": "ls"}))

        assert [violation.violation_type for violation in violations] == [ToolViolationType.TOOL_NOT_ALLOWED]

    @pytest.mark.asyncio
    async def test_blocked_argument_is_reported_before_one_outside_the_allowed_entry(self):
        """An argument both blocked and outside the tool's allowed_tools entry is reported as `argument_blocked`."""
        policy = _policy(
            allowed_tools=[{"name": "search", "arguments": ["query"]}],
            blocked_tools=[{"name": "search", "arguments": ["limit"]}],
        )

        violations = await _violations(policy, Toolset(), _call("search", {"query": "q", "limit": 3}))

        assert [violation.violation_type for violation in violations] == [ToolViolationType.ARGUMENT_BLOCKED]

    @pytest.mark.asyncio
    async def test_every_call_is_reported_in_order_across_the_lists_and_the_offered_tools(self):
        """Calls that fail the lists and calls that fail the offered tools each get a violation, in call order."""
        calls = [
            _call("get_weather", {"city": "Paris"}, call_id="c1"),
            _call("rm_rf", {}, call_id="c2"),
            _call("get_weather", {"city": 5}, call_id="c3"),
            _call("delete_files", {}, call_id="c4"),
        ]

        result = await ToolCallRailAction(policy=_policy(blocked_tools=["rm_rf"])).run(_toolset(), calls)

        assert [(violation.violation_type, violation.index) for violation in violations_in(result)] == [
            (ToolViolationType.TOOL_BLOCKED, 1),
            (ToolViolationType.ARGUMENTS_INVALID, 2),
            (ToolViolationType.TOOL_NOT_ALLOWED, 3),
        ]
        assert result.reason == "tool call 'rm_rf' is a blocked tool"

    @pytest.mark.asyncio
    async def test_action_without_a_policy_checks_only_the_offered_tools(self):
        """`tool call validation` built without a policy behaves as before the lists existed."""
        result = await ToolCallRailAction().run(_toolset(), [_call("get_weather", {"city": "Paris"})])

        assert result.is_blocked is False

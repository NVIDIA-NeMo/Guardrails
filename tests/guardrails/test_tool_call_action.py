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
from nemoguardrails.types import ToolCall, ToolCallFunction
from tests.guardrails.tool_helpers import WEATHER_SCHEMA, assert_blocked


def _toolset() -> Toolset:
    return Toolset(tools=[Tool(name="get_weather", arguments_schema=WEATHER_SCHEMA), Tool(name="ping")])


def _call(name: str, arguments: dict) -> ToolCall:
    return ToolCall(id="c1", function=ToolCallFunction(name=name, arguments=arguments))


class TestToolCallRailAction:
    @pytest.mark.asyncio
    async def test_allowed_call_with_valid_arguments_is_safe(self):
        result = await ToolCallRailAction().run(_toolset(), [_call("get_weather", {"city": "Paris"})])
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_no_parameter_function_tool_with_arguments_is_blocked(self):
        """A function tool that declares no parameters accepts no arguments, so a call supplying any is blocked."""
        result = await ToolCallRailAction().run(_toolset(), [_call("ping", {"anything": 1})])
        assert_blocked(result, "ping", "no arguments")

    @pytest.mark.asyncio
    async def test_no_parameter_function_tool_without_arguments_is_safe(self):
        """A no-parameter function tool called with no arguments passes."""
        result = await ToolCallRailAction().run(_toolset(), [_call("ping", {})])
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_undeclared_tool_is_blocked(self):
        result = await ToolCallRailAction().run(_toolset(), [_call("rm_rf", {})])
        assert_blocked(result, "rm_rf", "not an allowed tool")

    @pytest.mark.asyncio
    async def test_invalid_arguments_are_blocked(self):
        result = await ToolCallRailAction().run(_toolset(), [_call("get_weather", {})])
        assert_blocked(result, "get_weather")

    @pytest.mark.asyncio
    async def test_empty_tool_calls_is_safe(self):
        result = await ToolCallRailAction().run(_toolset(), [])
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_one_bad_call_blocks_the_batch(self):
        calls = [_call("get_weather", {"city": "Paris"}), _call("rm_rf", {})]
        result = await ToolCallRailAction().run(_toolset(), calls)
        assert_blocked(result, "rm_rf")

    @pytest.mark.asyncio
    async def test_hosted_tool_matched_by_type_when_name_is_empty(self):
        toolset = Toolset(tools=[Tool(name=None, type="web_search")])
        call = ToolCall(id="c1", type="web_search", function=ToolCallFunction(name="", arguments={}))
        result = await ToolCallRailAction().run(toolset, [call])
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_undeclared_hosted_tool_type_is_blocked(self):
        toolset = Toolset(tools=[Tool(name=None, type="web_search")])
        call = ToolCall(id="c1", type="code_interpreter", function=ToolCallFunction(name="", arguments={}))
        result = await ToolCallRailAction().run(toolset, [call])
        assert_blocked(result, "code_interpreter", "not an allowed tool")

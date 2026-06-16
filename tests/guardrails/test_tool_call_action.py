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


def _toolset() -> Toolset:
    return Toolset(
        tools=[
            Tool(
                name="get_weather",
                arguments_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
            Tool(name="ping"),
        ]
    )


def _call(name: str, arguments: dict) -> ToolCall:
    return ToolCall(id="c1", function=ToolCallFunction(name=name, arguments=arguments))


class TestToolCallRailAction:
    @pytest.mark.asyncio
    async def test_allowed_call_with_valid_arguments_is_safe(self):
        result = await ToolCallRailAction().run(_toolset(), [_call("get_weather", {"city": "Paris"})])
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_tool_without_schema_passes_allowlist_only(self):
        result = await ToolCallRailAction().run(_toolset(), [_call("ping", {"anything": 1})])
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_undeclared_tool_is_blocked(self):
        result = await ToolCallRailAction().run(_toolset(), [_call("rm_rf", {})])
        assert result.is_safe is False
        assert result.reason is not None
        assert "rm_rf" in result.reason
        assert "not an allowed tool" in result.reason

    @pytest.mark.asyncio
    async def test_invalid_arguments_are_blocked(self):
        result = await ToolCallRailAction().run(_toolset(), [_call("get_weather", {})])
        assert result.is_safe is False
        assert result.reason is not None
        assert "get_weather" in result.reason

    @pytest.mark.asyncio
    async def test_empty_tool_calls_is_safe(self):
        result = await ToolCallRailAction().run(_toolset(), [])
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_one_bad_call_blocks_the_batch(self):
        calls = [_call("get_weather", {"city": "Paris"}), _call("rm_rf", {})]
        result = await ToolCallRailAction().run(_toolset(), calls)
        assert result.is_safe is False
        assert result.reason is not None
        assert "rm_rf" in result.reason

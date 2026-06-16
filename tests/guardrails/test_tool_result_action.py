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
    async def test_result_without_name_links_on_call_id_only(self):
        result = await ToolResultRailAction().run([_result("c2")], _prior_calls())
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_list_content_is_well_formed(self):
        result = await ToolResultRailAction().run(
            [_result("c1", content=[{"type": "text", "text": "18C"}])], _prior_calls()
        )
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_empty_results_is_safe(self):
        result = await ToolResultRailAction().run([], _prior_calls())
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_missing_call_id_is_blocked(self):
        result = await ToolResultRailAction().run([_result("")], _prior_calls())
        assert result.is_safe is False
        assert result.reason is not None
        assert "missing a call_id" in result.reason

    @pytest.mark.asyncio
    async def test_unlinked_call_id_is_blocked(self):
        result = await ToolResultRailAction().run([_result("c9")], _prior_calls())
        assert result.is_safe is False
        assert result.reason is not None
        assert "c9" in result.reason
        assert "does not correspond to a prior tool call" in result.reason

    @pytest.mark.asyncio
    async def test_name_mismatch_is_blocked(self):
        result = await ToolResultRailAction().run([_result("c1", name="search")], _prior_calls())
        assert result.is_safe is False
        assert result.reason is not None
        assert "does not match the called tool" in result.reason
        assert "get_weather" in result.reason

    @pytest.mark.asyncio
    async def test_malformed_content_is_blocked(self):
        result = await ToolResultRailAction().run(
            [_result("c1", content={"unexpected": "shape"})],  # type: ignore[arg-type]
            _prior_calls(),
        )
        assert result.is_safe is False
        assert result.reason is not None
        assert "malformed content" in result.reason

    @pytest.mark.asyncio
    async def test_one_bad_result_blocks_the_batch(self):
        results = [_result("c1", name="get_weather"), _result("c9")]
        result = await ToolResultRailAction().run(results, _prior_calls())
        assert result.is_safe is False
        assert result.reason is not None
        assert "c9" in result.reason

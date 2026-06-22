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

"""Unit tests for the tool_schema module (canonical tool types + arg validation)."""

from dataclasses import FrozenInstanceError

import pytest

from nemoguardrails.guardrails.tool_schema import (
    Tool,
    ToolResult,
    Toolset,
    validate_arguments,
)

_WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer"},
    },
    "required": ["city"],
}


def _weather_tool() -> Tool:
    return Tool(
        name="get_weather",
        description="Get the weather for a city.",
        arguments_schema=_WEATHER_SCHEMA,
    )


class TestTool:
    def test_defaults(self):
        tool = Tool()
        assert tool.name is None
        assert tool.type == "function"
        assert tool.description is None
        assert tool.arguments_schema is None
        assert tool.strict is None

    def test_key_uses_name_when_present(self):
        assert Tool(name="get_weather").key == "get_weather"

    def test_key_falls_back_to_type_for_hosted_tool(self):
        # Hosted/server tools carry no name; the key is the type.
        assert Tool(name=None, type="web_search").key == "web_search"

    def test_is_frozen(self):
        tool = Tool(name="get_weather")
        with pytest.raises(FrozenInstanceError):
            setattr(tool, "name", "other")


class TestToolset:
    def test_empty(self):
        ts = Toolset()
        assert ts.tools == ()
        assert ts.get("anything") is None

    def test_get_returns_function_and_hosted_tools_by_key(self):
        fn = Tool(name="get_weather", arguments_schema=_WEATHER_SCHEMA)
        hosted = Tool(name=None, type="web_search")
        ts = Toolset(tools=[fn, hosted])

        assert ts.get("get_weather") is fn
        assert ts.get("web_search") is hosted
        assert ts.get("missing") is None

    def test_duplicate_function_name_raises(self):
        with pytest.raises(ValueError, match="duplicate tool 'dup'"):
            Toolset(tools=[Tool(name="dup", description="first"), Tool(name="dup", description="second")])

    def test_duplicate_hosted_tool_type_raises(self):
        with pytest.raises(ValueError, match="duplicate tool 'web_search'"):
            Toolset(tools=[Tool(name=None, type="web_search"), Tool(name=None, type="web_search")])

    def test_tool_with_empty_key_is_skipped(self):
        # name=None and an empty type => empty key => not indexed.
        ts = Toolset(tools=[Tool(name=None, type="")])
        assert ts.get("") is None

    def test_multiple_empty_key_tools_do_not_collide(self):
        # Empty-key tools are skipped, so duplicates among them never raise.
        ts = Toolset(tools=[Tool(name=None, type=""), Tool(name=None, type="")])
        assert ts.get("") is None


class TestToolResult:
    def test_defaults(self):
        result = ToolResult()
        assert result.call_id is None
        assert result.name is None
        assert result.content is None
        assert result.is_error is False

    def test_string_content(self):
        result = ToolResult(call_id="call_1", name="get_weather", content="18C")
        assert result.call_id == "call_1"
        assert result.name == "get_weather"
        assert result.content == "18C"
        assert result.is_error is False

    def test_block_content_and_error(self):
        blocks = [{"type": "text", "text": "boom"}]
        result = ToolResult(call_id="call_2", content=blocks, is_error=True)
        assert result.content == blocks
        assert result.is_error is True

    def test_is_frozen(self):
        result = ToolResult(call_id="call_1")
        with pytest.raises(FrozenInstanceError):
            setattr(result, "call_id", "other")


class TestValidateArguments:
    def test_valid_arguments_pass(self):
        assert validate_arguments(_weather_tool(), {"city": "Paris", "days": 3}) is None

    def test_valid_with_only_required(self):
        assert validate_arguments(_weather_tool(), {"city": "Paris"}) is None

    def test_type_mismatch_is_rejected(self):
        reason = validate_arguments(_weather_tool(), {"city": 123})
        assert reason is not None
        assert "get_weather" in reason

    def test_missing_required_is_rejected(self):
        reason = validate_arguments(_weather_tool(), {})
        assert reason is not None
        assert "get_weather" in reason

    def test_no_schema_skips_validation(self):
        # A tool with no declared schema (hosted tool, or a function with no
        # parameters) is allowlist-only: any arguments are accepted here.
        hosted = Tool(name=None, type="web_search")
        assert validate_arguments(hosted, {"anything": [1, 2, 3]}) is None

    def test_malformed_schema_is_reported_not_raised(self):
        # A declared schema that is not valid JSON Schema degrades to a reason
        # rather than raising (e.g. a non-JSON-Schema dialect reaching the
        # validator before its engine adapter normalizes it).
        bad = Tool(name="bad", arguments_schema={"type": "not-a-real-type"})
        reason = validate_arguments(bad, {})
        assert reason is not None
        assert "bad" in reason

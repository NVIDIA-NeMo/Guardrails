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

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.guardrails.tool_schema import (
    Tool,
    ToolResult,
    Toolset,
    scope_arguments,
    tool_output_validation,
    validate_arguments,
)
from nemoguardrails.rails.llm.options import ToolViolationType
from nemoguardrails.types import ToolCall, ToolCallFunction

_SECRET_VALUE = "SECRET-VALUE"

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
        """A type mismatch names the failing argument and keyword, never the value that failed."""
        violation = validate_arguments(_weather_tool(), {"city": "Paris", "days": _SECRET_VALUE})
        assert (violation.violation_type, violation.argument_path, violation.schema_keyword) == (
            ToolViolationType.ARGUMENTS_INVALID,
            "/days",
            "type",
        )
        assert violation.reason == "arguments for tool 'get_weather' do not match its schema: 'type' failed at '/days'"

    def test_missing_required_is_rejected(self):
        """A missing required argument is named in the path, although jsonschema reports it at the parent object."""
        violation = validate_arguments(_weather_tool(), {})
        assert (violation.violation_type, violation.argument_path, violation.schema_keyword) == (
            ToolViolationType.ARGUMENTS_INVALID,
            "/city",
            "required",
        )
        assert violation.reason == (
            "arguments for tool 'get_weather' do not match its schema: 'required' failed at '/city'"
        )

    def test_hosted_tool_without_schema_skips_validation(self):
        """A hosted/server tool (name is None) declares no schema and the provider owns the call shape, so it stays allowlist-only: any arguments are accepted here."""
        hosted = Tool(name=None, type="web_search")
        assert validate_arguments(hosted, {"anything": [1, 2, 3]}) is None

    def test_function_tool_without_parameters_allows_empty_arguments(self):
        """A no-parameter function tool legitimately called with no arguments stays safe."""
        tool = Tool(name="get_time", arguments_schema=None)
        assert validate_arguments(tool, {}) is None

    def test_function_tool_without_parameters_rejects_arguments(self):
        """A function tool that declares no parameters accepts no arguments, so supplying any blocks instead of skipping."""
        tool = Tool(name="get_time", arguments_schema=None)
        violation = validate_arguments(tool, {"path": "/etc/shadow"})
        assert (violation.violation_type, violation.argument_path, violation.schema_keyword) == (
            ToolViolationType.UNEXPECTED_ARGUMENTS,
            None,
            None,
        )
        assert violation.reason == "tool 'get_time' accepts no arguments but the call supplied: ['path']"

    def test_function_tool_empty_dict_schema_rejects_arguments(self):
        """An explicit empty schema ({}) declares no properties; jsonschema would accept anything, so it is treated as no-args."""
        tool = Tool(name="ping", arguments_schema={})
        violation = validate_arguments(tool, {"x": 1})
        assert violation.violation_type is ToolViolationType.UNEXPECTED_ARGUMENTS

    def test_function_tool_empty_properties_rejects_arguments(self):
        """An object schema with empty properties and no additionalProperties:false defaults to permissive, so it is treated as no-args."""
        tool = Tool(name="ping", arguments_schema={"type": "object", "properties": {}})
        violation = validate_arguments(tool, {"x": 1})
        assert violation.violation_type is ToolViolationType.UNEXPECTED_ARGUMENTS

    def test_function_tool_object_schema_without_properties_rejects_arguments(self):
        """An object schema with no properties key at all declares no inputs, so it is treated as no-args."""
        tool = Tool(name="ping", arguments_schema={"type": "object"})
        violation = validate_arguments(tool, {"x": 1})
        assert violation.violation_type is ToolViolationType.UNEXPECTED_ARGUMENTS

    def test_function_tool_additional_properties_schema_accepts_arguments(self):
        """A schema that opens additionalProperties declares an input channel, so it is not treated as no-args and free-form arguments pass."""
        tool = Tool(name="kv", arguments_schema={"type": "object", "additionalProperties": {"type": "string"}})
        assert validate_arguments(tool, {"foo": "bar"}) is None

    def test_function_tool_pattern_properties_schema_accepts_arguments(self):
        """A schema that opens patternProperties declares an input channel, so it is not treated as no-args and matching arguments pass."""
        tool = Tool(name="kv", arguments_schema={"type": "object", "patternProperties": {"^x": {"type": "integer"}}})
        assert validate_arguments(tool, {"x1": 1}) is None

    def test_function_tool_composition_schema_accepts_arguments(self):
        """A schema using a composition keyword declares an input channel, so it is not treated as no-args and conforming arguments pass."""
        tool = Tool(name="kv", arguments_schema={"anyOf": [{"type": "object"}]})
        assert validate_arguments(tool, {"x": 1}) is None

    def test_malformed_schema_is_reported_not_raised(self):
        """A declared schema that is not valid JSON Schema gives ``invalid_tool_schema`` rather than raising."""
        bad = Tool(name="bad", arguments_schema={"type": "not-a-real-type"})
        violation = validate_arguments(bad, {})
        assert violation.violation_type is ToolViolationType.INVALID_TOOL_SCHEMA
        assert violation.reason.startswith("declared schema for tool 'bad' is not valid JSON Schema")


_CLOSED_WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "additionalProperties": False,
}


class TestArgumentPath:
    """Where ``validate_arguments`` points ``argument_path`` for each kind of schema failure."""

    def test_unexpected_key_is_named(self):
        """``additionalProperties: false`` points at the first unexpected key, and the reason omits its value."""
        tool = Tool(name="get_weather", arguments_schema=_CLOSED_WEATHER_SCHEMA)
        violation = validate_arguments(tool, {"city": "Paris", "debug": _SECRET_VALUE, "trace": 1})
        assert (violation.argument_path, violation.schema_keyword) == ("/debug", "additionalProperties")
        assert violation.reason == (
            "arguments for tool 'get_weather' do not match its schema: 'additionalProperties' failed at '/debug'"
        )

    def test_long_unexpected_key_is_truncated(self):
        """A path segment longer than 64 characters is cut to its first 64."""
        tool = Tool(name="get_weather", arguments_schema=_CLOSED_WEATHER_SCHEMA)
        violation = validate_arguments(tool, {"city": "Paris", "k" * 100: 1})
        assert violation.argument_path == "/" + "k" * 64

    def test_nested_failure_gives_the_full_pointer(self):
        """A failure inside a nested object is pointed at through its parent."""
        schema = {
            "type": "object",
            "properties": {
                "opts": {"type": "object", "properties": {"units": {"type": "string"}}, "additionalProperties": False}
            },
        }
        tool = Tool(name="configure", arguments_schema=schema)
        violation = validate_arguments(tool, {"opts": {"units": "C", "rm_rf": _SECRET_VALUE}})
        assert violation.argument_path == "/opts/rm_rf"

    def test_missing_dependency_is_named(self):
        """``dependentRequired`` points at the missing dependency, whose name comes from the schema."""
        schema = {"type": "object", "properties": {"a": {}, "b": {}}, "dependentRequired": {"a": ["b"]}}
        violation = validate_arguments(Tool(name="pair", arguments_schema=schema), {"a": 1})
        assert (violation.argument_path, violation.schema_keyword) == ("/b", "dependentRequired")

    def test_pointer_segments_are_escaped(self):
        """``~`` and ``/`` in a property name are escaped as JSON Pointer requires."""
        schema = {"type": "object", "properties": {"a/b~c": {"type": "string"}}}
        violation = validate_arguments(Tool(name="odd", arguments_schema=schema), {"a/b~c": 1})
        assert violation.argument_path == "/a~1b~0c"

    def test_property_name_failure_keeps_the_parent_path(self):
        """A ``propertyNames`` failure is reported at the object, so neither the path nor the reason names the key."""
        schema = {"type": "object", "properties": {"a": {}}, "propertyNames": {"maxLength": 3}}
        violation = validate_arguments(Tool(name="short_keys", arguments_schema=schema), {_SECRET_VALUE: 1})
        assert violation.argument_path == ""
        assert _SECRET_VALUE not in violation.reason

    def test_unevaluated_property_failure_keeps_the_parent_path(self):
        """An ``unevaluatedProperties`` failure is reported at the object; naming the key needs jsonschema internals."""
        schema = {"type": "object", "properties": {"city": {"type": "string"}}, "unevaluatedProperties": False}
        violation = validate_arguments(Tool(name="get_weather", arguments_schema=schema), {"city": "Paris", "debug": 1})
        assert (violation.argument_path, violation.schema_keyword) == ("", "unevaluatedProperties")

    def test_top_level_failure_reads_as_the_top_level(self):
        """A failure at the root keeps an empty ``argument_path``, and its reason says "the top level"."""
        schema = {"type": "object", "properties": {"a": {}, "b": {}}, "minProperties": 2}
        violation = validate_arguments(Tool(name="pair", arguments_schema=schema), {"a": 1})
        assert violation.argument_path == ""
        assert violation.reason == (
            "arguments for tool 'pair' do not match its schema: 'minProperties' failed at the top level"
        )


def _weather_call(arguments: dict) -> ToolCall:
    return ToolCall(id="call_1", type="function", function=ToolCallFunction(name="get_weather", arguments=arguments))


class TestScopeArguments:
    def test_no_argument_name_returns_full_arguments(self):
        arguments = {"city": "Paris", "units": "metric"}
        assert scope_arguments(arguments, None) == arguments

    def test_present_argument_name_narrows_to_that_field(self):
        arguments = {"city": "Paris", "units": "metric"}
        assert scope_arguments(arguments, "city") == {"city": "Paris"}

    def test_absent_argument_name_falls_back_to_full_arguments(self):
        """A call that omits the scoped field (e.g. an optional one) leans toward more
        scrutiny rather than scoping to a misleading {name: None}."""
        arguments = {"city": "Paris"}
        assert scope_arguments(arguments, "units") == arguments


class TestToolOutputValidation:
    """The @tool_output_validation decorator, focused on its $argument= check."""

    @staticmethod
    @tool_output_validation
    async def _action(tool_call, tool_definition, argument_name=None, **kwargs):
        return RailOutcome.allow()

    @pytest.mark.asyncio
    async def test_no_argument_name_skips_the_check(self):
        outcome = await self._action(tool_call=_weather_call({"city": "Paris"}), tool_definition=_weather_tool())
        assert outcome == RailOutcome.allow()

    @pytest.mark.asyncio
    async def test_declared_argument_name_passes(self):
        outcome = await self._action(
            tool_call=_weather_call({"city": "Paris"}), tool_definition=_weather_tool(), argument_name="city"
        )
        assert outcome == RailOutcome.allow()

    @pytest.mark.asyncio
    async def test_undeclared_name_on_closed_schema_raises(self):
        closed = Tool(
            name="get_weather",
            arguments_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "additionalProperties": False,
            },
        )
        with pytest.raises(ValueError, match="not declared"):
            await self._action(
                tool_call=_weather_call({"city": "Paris"}), tool_definition=closed, argument_name="units"
            )

    @pytest.mark.asyncio
    async def test_undeclared_name_matching_pattern_properties_passes(self):
        """A name matching a patternProperties regex is accepted, same as validate_arguments
        already accepts a call whose fields match that pattern."""
        pattern_schema = Tool(
            name="kv",
            arguments_schema={
                "type": "object",
                "patternProperties": {"^x": {"type": "integer"}},
                "additionalProperties": False,
            },
        )
        outcome = await self._action(
            tool_call=ToolCall(id="call_1", type="function", function=ToolCallFunction(name="kv", arguments={"x1": 1})),
            tool_definition=pattern_schema,
            argument_name="x1",
        )
        assert outcome == RailOutcome.allow()

    @pytest.mark.asyncio
    async def test_name_not_matching_any_pattern_property_raises(self):
        """A closed schema with patternProperties still rejects a name none of its
        patterns match, so it doesn't silently scope to a field the schema never covers."""
        pattern_schema = Tool(
            name="kv",
            arguments_schema={
                "type": "object",
                "patternProperties": {"^x": {"type": "integer"}},
                "additionalProperties": False,
            },
        )
        with pytest.raises(ValueError, match="not declared"):
            await self._action(
                tool_call=ToolCall(id="call_1", type="function", function=ToolCallFunction(name="kv", arguments={})),
                tool_definition=pattern_schema,
                argument_name="y1",
            )

    @pytest.mark.asyncio
    async def test_undeclared_name_on_permissive_schema_passes(self):
        """additionalProperties left permissive (or true) may accept a field it doesn't list, so it's not evidence of a typo."""
        permissive = Tool(name="get_weather", arguments_schema={"type": "object", "additionalProperties": True})
        outcome = await self._action(
            tool_call=_weather_call({"units": "metric"}), tool_definition=permissive, argument_name="units"
        )
        assert outcome == RailOutcome.allow()

    @pytest.mark.asyncio
    async def test_argument_name_with_no_schema_raises(self):
        hosted = Tool(name=None, type="web_search")
        with pytest.raises(ValueError, match="declares no schema"):
            await self._action(
                tool_call=ToolCall(id="call_1", type="web_search", function=ToolCallFunction(name="", arguments={})),
                tool_definition=hosted,
                argument_name="query",
            )

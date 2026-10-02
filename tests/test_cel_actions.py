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

"""Tests for the CEL rail's config and its tool output / tool input actions.

The action-level companion to tests/guardrails/test_per_tool_cel_rails_iorails.py, which drives
the same rail through IORails.
"""

import sys

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.rail_outcome import RailDecision
from nemoguardrails.guardrails.tool_schema import Tool, ToolResult
from nemoguardrails.imports import check_optional_dependency
from nemoguardrails.library.cel.actions import evaluate_tool_input_cel, evaluate_tool_output_cel
from nemoguardrails.library.cel.rail_config import CelOptions
from nemoguardrails.types import ToolCall, ToolCallFunction

pytestmark = pytest.mark.skipif(not check_optional_dependency("cel"), reason="requires the 'cel' extra (Python 3.11+)")

RUN_SHELL_TOOL = Tool(
    name="run_shell",
    arguments_schema={
        "type": "object",
        "properties": {"command": {"type": "string"}, "flags": {"type": "array", "items": {"type": "string"}}},
        "required": ["command"],
    },
)


def _config(direction: str, *expressions: str, tool: str = "run_shell") -> RailsConfig:
    return RailsConfig.from_content(
        config={"rails": {"config": {"cel": {direction: {tool: {"expressions": list(expressions)}}}}}}
    )


def _shell_call(**arguments) -> ToolCall:
    return ToolCall(id="call_1", function=ToolCallFunction(name="run_shell", arguments=arguments))


async def _check_call(config: RailsConfig, call: ToolCall, tool_definition=RUN_SHELL_TOOL):
    return await evaluate_tool_output_cel(
        source="tool_output", tool_call=call, tool_definition=tool_definition, config=config
    )


async def _check_result(config: RailsConfig, content, call: ToolCall = None):
    return await evaluate_tool_input_cel(
        source="tool_input",
        tool_call=call or _shell_call(command="cat key.pem"),
        tool_result=ToolResult(call_id="call_1", content=content),
        config=config,
    )


# Expressions that compile but fail on a `run_shell` call with only `command` set, keyed by case,
# with the exception type the evaluation error names.
_EVALUATION_ERRORS = {
    "argument_typo": ('args.comand == "rm"', "KeyError"),
    "variable_typo": ('arg.command == "rm"', "RuntimeError"),
    "function_typo": ('args.command.lowerAsci() == "rm"', "RuntimeError"),
    "type_mismatch": ("args.command > 5", "TypeError"),
}


class TestConfig:
    def test_expressions_compile_at_load(self):
        config = _config("tool_output", 'args.command == "ls"', "true")
        assert len(config.rails.config.cel.tool_output["run_shell"].compiled_expressions) == 2

    def test_syntax_error_fails_the_load(self):
        with pytest.raises(ValueError, match=r"Invalid CEL expression at index 1 \('args.command.matches\('\)"):
            _config("tool_output", "true", "args.command.matches(")

    @pytest.mark.parametrize(
        "expression",
        ['args.command == "rm', "args.command ==", "args.command && && true"],
        ids=["unterminated_string", "dangling_operator", "repeated_operator"],
    )
    def test_other_syntax_errors_fail_the_load(self, expression):
        with pytest.raises(ValueError, match="Invalid CEL expression at index 0"):
            _config("tool_output", expression)

    @pytest.mark.parametrize(
        "expression", [expression for expression, _ in _EVALUATION_ERRORS.values()], ids=_EVALUATION_ERRORS.keys()
    )
    def test_only_syntax_is_checked_at_load(self, expression):
        """Typos and type mismatches compile, and are caught only when evaluated (see TestFailures)."""
        _config("tool_output", expression)

    def test_missing_extra_names_it(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cel", None)
        with pytest.raises(ImportError, match=r"nemoguardrails\[cel\]"):
            CelOptions(expressions=["true"])

    def test_no_expressions_needs_no_cel(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cel", None)
        assert CelOptions().compiled_expressions == []


class TestToolOutput:
    @pytest.mark.asyncio
    async def test_match_blocks_with_every_matched_expression(self):
        config = _config(
            "tool_output",
            'args.command.startsWith("rm")',
            'args.command == "ls"',
            'args.command.contains("-rf")',
        )
        outcome = await _check_call(config, _shell_call(command="rm -rf /"))
        assert outcome.decision is RailDecision.BLOCK
        assert outcome.metadata == {
            "source": "tool_output",
            "matched_expressions": ['args.command.startsWith("rm")', 'args.command.contains("-rf")'],
        }

    @pytest.mark.asyncio
    async def test_no_match_allows(self):
        outcome = await _check_call(_config("tool_output", 'args.command == "rm"'), _shell_call(command="ls"))
        assert outcome.decision is RailDecision.ALLOW
        assert outcome.metadata == {"source": "tool_output"}

    @pytest.mark.asyncio
    async def test_tool_without_expressions_allows(self):
        outcome = await _check_call(_config("tool_output", "true", tool="other_tool"), _shell_call(command="rm"))
        assert outcome.decision is RailDecision.ALLOW

    @pytest.mark.asyncio
    async def test_null_cel_section_allows(self):
        config = RailsConfig.from_content(config={"rails": {"config": {"cel": None}}})
        outcome = await _check_call(config, _shell_call(command="rm -rf /"))
        assert outcome.decision is RailDecision.ALLOW

    @pytest.mark.asyncio
    async def test_tool_name_is_available(self):
        outcome = await _check_call(_config("tool_output", 'tool == "run_shell"'), _shell_call(command="ls"))
        assert outcome.decision is RailDecision.BLOCK

    @pytest.mark.asyncio
    async def test_stdlib_extensions_are_available(self):
        config = _config("tool_output", 'args.command.lowerAscii().startsWith("rm")')
        outcome = await _check_call(config, _shell_call(command="RM -rf /"))
        assert outcome.decision is RailDecision.BLOCK

    @pytest.mark.asyncio
    async def test_undeclared_tool_blocks_before_evaluation(self):
        # The expression raises if evaluated, so a block (not a raise) shows it never ran.
        config = _config("tool_output", "args.missing == 1")
        outcome = await _check_call(config, _shell_call(command="ls"), tool_definition=None)
        assert outcome.decision is RailDecision.BLOCK
        assert "not an allowed tool" in outcome.reason

    @pytest.mark.asyncio
    async def test_schema_violation_blocks_before_evaluation(self):
        config = _config("tool_output", "args.missing == 1")
        outcome = await _check_call(config, _shell_call(command=123))
        assert outcome.decision is RailDecision.BLOCK
        assert "matched_expressions" not in outcome.metadata

    @pytest.mark.asyncio
    async def test_wrong_source_raises(self):
        with pytest.raises(ValueError, match="source must be 'tool_output'"):
            await evaluate_tool_output_cel(
                source="tool_input",
                tool_call=_shell_call(command="ls"),
                tool_definition=RUN_SHELL_TOOL,
                config=_config("tool_output", "true"),
            )


class TestFailures:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("expression, error_type", _EVALUATION_ERRORS.values(), ids=_EVALUATION_ERRORS.keys())
    async def test_evaluation_error_fails_closed(self, expression, error_type):
        with pytest.raises(RuntimeError) as excinfo:
            await _check_call(_config("tool_output", expression), _shell_call(command="rm"))
        assert f"{expression!r}: {error_type}: " in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_failing_expression_raises_despite_a_match(self):
        config = _config("tool_output", 'args.flags.exists(f, f == "--force")', 'args.command == "rm"')
        with pytest.raises(RuntimeError, match="KeyError: 'flags'"):
            await _check_call(config, _shell_call(command="rm"))

    @pytest.mark.asyncio
    async def test_failures_without_a_match_raise_listing_each(self):
        config = _config("tool_output", 'args.flags.exists(f, f == "--force")', "args.command", 'args.command == "rm"')
        with pytest.raises(RuntimeError) as excinfo:
            await _check_call(config, _shell_call(command="ls"))
        message = str(excinfo.value)
        assert message.startswith("CEL expressions for tool 'run_shell' failed: ")
        assert "'args.flags.exists(f, f == \"--force\")': KeyError: 'flags'" in message
        assert "'args.command': TypeError: returned str, not bool" in message

    @pytest.mark.asyncio
    async def test_failure_message_carries_no_argument_values(self):
        config = _config("tool_output", "args.command > 5")
        with pytest.raises(RuntimeError) as excinfo:
            await _check_call(config, _shell_call(command="SECRET_TOKEN"))
        assert "SECRET_TOKEN" not in str(excinfo.value)


class TestToolInput:
    @pytest.mark.asyncio
    async def test_string_result_match_blocks(self):
        config = _config("tool_input", 'result.contains("BEGIN RSA PRIVATE KEY")')
        outcome = await _check_result(config, "-----BEGIN RSA PRIVATE KEY-----")
        assert outcome.decision is RailDecision.BLOCK
        assert outcome.metadata == {
            "source": "tool_input",
            "matched_expressions": ['result.contains("BEGIN RSA PRIVATE KEY")'],
        }

    @pytest.mark.asyncio
    async def test_string_result_no_match_allows(self):
        config = _config("tool_input", 'result.contains("BEGIN RSA PRIVATE KEY")')
        outcome = await _check_result(config, "no keys here")
        assert outcome.decision is RailDecision.ALLOW
        assert outcome.metadata == {"source": "tool_input"}

    @pytest.mark.asyncio
    async def test_content_blocks_are_queryable(self):
        config = _config("tool_input", 'result.exists(b, b.type == "text" && b.text.contains("password"))')
        outcome = await _check_result(config, [{"type": "text", "text": "password=hunter2"}])
        assert outcome.decision is RailDecision.BLOCK

    @pytest.mark.asyncio
    async def test_type_guard_skips_other_shapes(self):
        config = _config("tool_input", 'type(result) == string && result.contains("password")')
        for content in ([{"type": "text", "text": "password"}], None):
            outcome = await _check_result(config, content)
            assert outcome.decision is RailDecision.ALLOW

    @pytest.mark.asyncio
    async def test_null_cel_section_allows(self):
        config = RailsConfig.from_content(config={"rails": {"config": {"cel": None}}})
        outcome = await _check_result(config, "-----BEGIN RSA PRIVATE KEY-----")
        assert outcome.decision is RailDecision.ALLOW

    @pytest.mark.asyncio
    async def test_args_and_tool_come_from_the_answered_call(self):
        config = _config("tool_input", 'tool == "run_shell" && args.command.endsWith(".pem")')
        outcome = await _check_result(config, "ok", call=_shell_call(command="cat key.pem"))
        assert outcome.decision is RailDecision.BLOCK

    @pytest.mark.asyncio
    async def test_wrong_source_raises(self):
        with pytest.raises(ValueError, match="source must be 'tool_input'"):
            await evaluate_tool_input_cel(
                source="tool_output",
                tool_call=_shell_call(command="ls"),
                tool_result=ToolResult(call_id="call_1", content="ok"),
                config=_config("tool_input", "true"),
            )

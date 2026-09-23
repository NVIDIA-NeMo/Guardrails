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

import pytest
from pydantic import ValidationError

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.guardrails.tool_schema import Tool, ToolResult
from nemoguardrails.library.regex.actions import (
    detect_regex_pattern,
    detect_tool_input_regex_pattern,
    detect_tool_output_regex_pattern,
)
from nemoguardrails.types import ToolCall, ToolCallFunction
from tests.utils import TestChat


def _tool_call(name: str, arguments: dict, call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, type="function", function=ToolCallFunction(name=name, arguments=arguments))


def _permissive_tool(name: str) -> Tool:
    """A declared tool whose schema accepts any arguments, for tests that aren't
    exercising @tool_output_validation's own schema check."""
    return Tool(name=name, arguments_schema={"type": "object", "additionalProperties": True})


@pytest.mark.unit
def test_regex_detection_input_blocks_matching_pattern():
    """Test that input containing a matching regex pattern is blocked."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "\\\\d{3}-\\\\d{2}-\\\\d{4}"
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hello there!"'],
    )

    # This should trigger the input rail (contains SSN pattern)
    chat >> "My SSN is 123-45-6789"
    chat << "I'm sorry, I can't respond to that."


@pytest.mark.unit
def test_regex_detection_input_allows_non_matching():
    """Test that input not matching any regex pattern is allowed."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "\\\\d{3}-\\\\d{2}-\\\\d{4}"
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hello there!"'],
    )

    # This should NOT trigger the input rail (no SSN pattern)
    chat >> "Hi there!"
    chat << "Hello there!"


@pytest.mark.unit
def test_regex_detection_output_blocks_matching_pattern():
    """Test that output containing a matching regex pattern is blocked."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  output:
                    patterns:
                      - "\\\\bconfidential\\\\b"
              output:
                flows:
                  - regex check output
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "This is confidential information."'],
    )

    # The LLM output contains "confidential" which should be blocked
    chat >> "Hi!"
    chat << "I'm sorry, I can't respond to that."


@pytest.mark.unit
def test_regex_detection_case_insensitive():
    """Test that case insensitive matching works correctly."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "\\\\bpassword\\\\b"
                    case_insensitive: true
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hello!"'],
    )

    # Should match regardless of case
    chat >> "My PASSWORD is secret"
    chat << "I'm sorry, I can't respond to that."


@pytest.mark.unit
def test_regex_detection_case_sensitive():
    """Test that case sensitive matching works correctly (default behavior)."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "\\\\bpassword\\\\b"
                    case_insensitive: false
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hello!"'],
    )

    # Should NOT match because case is different
    chat >> "My PASSWORD is secret"
    chat << "Hello!"


@pytest.mark.unit
def test_regex_detection_multiple_patterns():
    """Test that multiple regex patterns work correctly."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "\\\\bsecret\\\\b"
                      - "\\\\bpassword\\\\b"
                      - "\\\\bapi[_-]?key\\\\b"
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello!"',
            "  express greeting",
            '  "Hello!"',
            "  express greeting",
            '  "Hello!"',
        ],
    )

    # First pattern should match
    chat >> "This is a secret message"
    chat << "I'm sorry, I can't respond to that."

    # Second pattern should match
    chat >> "My password is 12345"
    chat << "I'm sorry, I can't respond to that."

    # Third pattern should match (with hyphen variant)
    chat >> "The api-key is abc123"
    chat << "I'm sorry, I can't respond to that."


@pytest.mark.unit
def test_regex_detection_empty_patterns_allows_all():
    """Test that empty patterns list allows all input."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns: []
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hello!"'],
    )

    # Should pass because no patterns are defined
    chat >> "My password is secret and my SSN is 123-45-6789"
    chat << "Hello!"


@pytest.mark.unit
def test_regex_detection_input_and_output():
    """Test regex detection on both input and output."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "\\\\bpassword\\\\b"
                  output:
                    patterns:
                      - "\\\\bsecret\\\\b"
              input:
                flows:
                  - regex check input
              output:
                flows:
                  - regex check output
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Here is a secret!"',
            "  express greeting",
            '  "Hello!"',
        ],
    )

    # Input should pass, but output should be blocked
    chat >> "Hi there!"
    chat << "I'm sorry, I can't respond to that."

    # Input should be blocked
    chat >> "My password is 12345"
    chat << "I'm sorry, I can't respond to that."


@pytest.mark.unit
def test_regex_detection_complex_patterns():
    """Test regex detection with complex patterns like email and phone."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}"
                      - "\\\\(?\\\\d{3}\\\\)?[-.\\\\s]?\\\\d{3}[-.\\\\s]?\\\\d{4}"
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello!"',
            "  express greeting",
            '  "Hello!"',
        ],
    )

    # Email pattern should match
    chat >> "Contact me at john.doe@example.com"
    chat << "I'm sorry, I can't respond to that."

    # Phone pattern should match
    chat >> "Call me at (555) 123-4567"
    chat << "I'm sorry, I can't respond to that."


@pytest.mark.unit
def test_regex_detection_invalid_pattern_raises_at_config_load():
    """Invalid regex patterns are caught at config load time via model_validator, not at runtime."""
    with pytest.raises(ValidationError) as excinfo:
        RailsConfig.from_content(
            yaml_content="""
                models: []
                rails:
                  config:
                    regex_detection:
                      input:
                        patterns:
                          - "[unclosed"
                          - "\\\\bvalid\\\\b"
                  input:
                    flows:
                      - regex check input
            """,
            colang_content="""
                define user express greeting
                  "hi"

                define flow
                  user express greeting
                  bot express greeting

                define bot inform answer unknown
                  "I can't answer that."
            """,
        )
    assert "Invalid regex pattern" in str(excinfo.value)
    assert "[unclosed" in str(excinfo.value)


@pytest.mark.unit
def test_regex_detection_word_boundary():
    """Test that word boundaries work correctly."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "\\\\bpass\\\\b"
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hello!"'],
    )

    # "password" should NOT match because we're looking for "pass" as a whole word
    chat >> "My password is secret"
    chat << "Hello!"


@pytest.mark.unit
def test_regex_detection_word_boundary_matches():
    """Test that word boundaries match correctly when the word is standalone."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns:
                      - "\\\\bpass\\\\b"
              input:
                flows:
                  - regex check input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot refuse to respond
              "I'm sorry, I can't respond to that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Hello!"'],
    )

    # "pass" as standalone word should match
    chat >> "Please pass the test"
    chat << "I'm sorry, I can't respond to that."


@pytest.mark.unit
def test_regex_detection_retrieval_clears_matching_chunks():
    """Test that retrieved chunks matching a regex pattern are cleared.

    Retrieval rails run inside bot message generation, so they cannot emit a new
    bot message.  Instead the matched chunks are cleared (set to empty string) so
    the LLM never sees the flagged content.  A second retrieval flow verifies the
    chunks were actually cleared.
    """
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  retrieval:
                    patterns:
                      - "\\\\bclassified\\\\b"
                    case_insensitive: true
              retrieval:
                flows:
                  - regex check retrieval
                  - check relevant chunks
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define flow check relevant chunks
              execute check_relevant_chunks(relevant_chunks=$relevant_chunks)
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Here is what I found."'],
    )

    @action()
    def retrieve_relevant_chunks():
        context_updates = {"relevant_chunks": "This document is classified material."}
        return ActionResult(
            return_value=context_updates["relevant_chunks"],
            context_updates=context_updates,
        )

    @action()
    def check_relevant_chunks(relevant_chunks: str):
        # After the regex retrieval rail, matched chunks should be cleared.
        assert relevant_chunks == ""

    chat.app.register_action(retrieve_relevant_chunks)
    chat.app.register_action(check_relevant_chunks)

    # The retrieved chunk contains "classified" — the rail clears it, but the
    # bot still responds (without the dangerous KB context).
    chat >> "Hi!"
    chat << "Here is what I found."


@pytest.mark.unit
def test_regex_detection_retrieval_allows_non_matching():
    """Test that retrieved chunks not matching any pattern are left untouched."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  retrieval:
                    patterns:
                      - "\\\\bclassified\\\\b"
              retrieval:
                flows:
                  - regex check retrieval
                  - check relevant chunks
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define flow check relevant chunks
              execute check_relevant_chunks(relevant_chunks=$relevant_chunks)
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", '  "Here is what I found."'],
    )

    @action()
    def retrieve_relevant_chunks():
        context_updates = {"relevant_chunks": "This document is public information."}
        return ActionResult(
            return_value=context_updates["relevant_chunks"],
            context_updates=context_updates,
        )

    @action()
    def check_relevant_chunks(relevant_chunks: str):
        # No match, so chunks should be passed through unchanged.
        assert relevant_chunks == "This document is public information."

    chat.app.register_action(retrieve_relevant_chunks)
    chat.app.register_action(check_relevant_chunks)

    # The retrieved chunk does NOT contain "classified" — passes through unchanged.
    chat >> "Hi!"
    chat << "Here is what I found."


@pytest.mark.unit
@pytest.mark.asyncio
async def test_regex_action_accepts_extra_kwargs():
    """Regression: detect_regex_pattern must accept the extra kwargs that
    the action dispatcher injects during output streaming."""
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  output:
                    patterns:
                      - "\\\\bconfidential\\\\b"
        """,
        colang_content="",
    )

    result = await detect_regex_pattern(
        source="output",
        text="This is confidential information.",
        config=config,
        context={"user_message": "hi"},
        llm_task_manager=object(),
        model_name="test-model",
        llms={"main": object()},
        llm=object(),
    )

    assert result.is_blocked is True
    assert result.metadata["is_match"] is True
    assert "\\bconfidential\\b" in result.metadata["detections"]


@pytest.mark.unit
def test_regex_output_verdict_blocks_on_match():
    from nemoguardrails.actions.rail_outcome import RailOutcome

    matched = RailOutcome.block(
        metadata={"is_match": True, "text": "fight club", "detections": ["\\bfight\\s+club\\b"]}
    )
    no_match = RailOutcome.allow(metadata={"is_match": False, "text": "hello", "detections": []})

    assert matched.is_blocked is True
    assert no_match.is_blocked is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_tool_output_regex_pattern_rejects_invalid_source():
    config = RailsConfig.from_content(yaml_content="models: []", colang_content="")

    with pytest.raises(ValueError, match="source must be 'tool_output'"):
        await detect_tool_output_regex_pattern(
            source="bogus",
            tool_call=_tool_call("run_sql", {"query": "hi"}),
            tool_definition=_permissive_tool("run_sql"),
            config=config,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_tool_output_regex_pattern_blocks_undeclared_tool():
    """@tool_output_validation blocks before the regex check itself runs."""
    config = RailsConfig.from_content(yaml_content="models: []", colang_content="")

    result = await detect_tool_output_regex_pattern(
        source="tool_output",
        tool_call=_tool_call("run_sql", {"query": "hi"}),
        tool_definition=None,
        config=config,
    )

    assert result.is_blocked is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_tool_output_regex_pattern_blocks_schema_violation():
    """@tool_output_validation blocks arguments that don't match the declared schema."""
    config = RailsConfig.from_content(yaml_content="models: []", colang_content="")
    tool = Tool(name="run_sql", arguments_schema={"type": "object", "properties": {"query": {"type": "string"}}})

    result = await detect_tool_output_regex_pattern(
        source="tool_output",
        tool_call=_tool_call("run_sql", {"query": 123}),
        tool_definition=tool,
        config=config,
    )

    assert result.is_blocked is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_tool_input_regex_pattern_rejects_invalid_source():
    config = RailsConfig.from_content(yaml_content="models: []", colang_content="")

    with pytest.raises(ValueError, match="source must be 'tool_input'"):
        await detect_tool_input_regex_pattern(
            source="bogus",
            tool_call=_tool_call("run_sql", {}),
            tool_result=ToolResult(call_id="call_1", name="run_sql", content="hi"),
            config=config,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_tool_output_regex_pattern_allows_when_regex_detection_section_absent():
    """No `regex_detection` section at all: fails open.

    `RailsConfig` always populates `regex_detection` via a Pydantic default_factory, so
    this hits the same "no pattern group for this tool" branch as an unconfigured tool
    name. There is no separate `regex_config is None` branch, since it would be
    unreachable through any real config.
    """
    config = RailsConfig.from_content(yaml_content="models: []", colang_content="")

    result = await detect_tool_output_regex_pattern(
        source="tool_output",
        tool_call=_tool_call("run_sql", {"query": "DROP TABLE users"}),
        tool_definition=_permissive_tool("run_sql"),
        config=config,
    )

    assert result.is_blocked is False
    assert result.metadata["is_match"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_tool_output_regex_pattern_allows_no_regex_configured_tool():
    """A tool name with no configured pattern group fails open, rather than refusing.

    This is the known gap load-time validation would close: a typo'd tool name in
    `per_tool` silently never blocks anything, instead of failing config load.
    """
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  tool_output:
                    run_sql:
                      patterns:
                        - "DROP\\\\s+TABLE"
        """,
        colang_content="",
    )

    result = await detect_tool_output_regex_pattern(
        source="tool_output",
        tool_call=_tool_call("other_tool", {"query": "DROP TABLE users"}),
        tool_definition=_permissive_tool("other_tool"),
        config=config,
    )

    assert result.is_blocked is False
    assert result.metadata["is_match"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_tool_output_regex_pattern_allows_when_no_patterns_configured_for_tool():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  tool_output:
                    run_sql: {}
        """,
        colang_content="",
    )

    result = await detect_tool_output_regex_pattern(
        source="tool_output",
        tool_call=_tool_call("run_sql", {"query": "DROP TABLE users"}),
        tool_definition=_permissive_tool("run_sql"),
        config=config,
    )

    assert result.is_blocked is False
    assert result.metadata["is_match"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_tool_input_regex_pattern_allows_no_regex_configured_tool():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  tool_input:
                    run_sql:
                      patterns:
                        - "ssn:\\\\s*\\\\d{3}-\\\\d{2}-\\\\d{4}"
        """,
        colang_content="",
    )

    result = await detect_tool_input_regex_pattern(
        source="tool_input",
        tool_call=_tool_call("other_tool", {}),
        tool_result=ToolResult(call_id="call_1", name="other_tool", content="ssn: 123-45-6789"),
        config=config,
    )

    assert result.is_blocked is False
    assert result.metadata["is_match"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detect_tool_input_regex_pattern_allows_empty_content():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                regex_detection:
                  tool_input:
                    run_sql:
                      patterns:
                        - "DROP\\\\s+TABLE"
        """,
        colang_content="",
    )

    result = await detect_tool_input_regex_pattern(
        source="tool_input",
        tool_call=_tool_call("run_sql", {}),
        tool_result=ToolResult(call_id="call_1", name="run_sql", content=""),
        config=config,
    )

    assert result.is_blocked is False
    assert result.metadata["is_match"] is False

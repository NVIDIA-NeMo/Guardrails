# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for event-based tool_calls extraction."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from tests.utils import TestChat


@action(is_system_action=True)
async def validate_tool_parameters(tool_calls, context=None, **kwargs):
    """Test implementation of tool parameter validation."""
    tool_calls = tool_calls or (context.get("tool_calls", []) if context else [])

    dangerous_patterns = ["eval", "exec", "system", "../", "rm -", "DROP", "DELETE"]

    for tool_call in tool_calls:
        args = tool_call.get("args", {})
        for param_value in args.values():
            if isinstance(param_value, str):
                if any(
                    pattern.lower() in param_value.lower()
                    for pattern in dangerous_patterns
                ):
                    return False
    return True


@action(is_system_action=True)
async def self_check_tool_calls(tool_calls, context=None, **kwargs):
    """Test implementation of tool call validation."""
    tool_calls = tool_calls or (context.get("tool_calls", []) if context else [])

    return all(
        isinstance(call, dict) and "name" in call and "id" in call
        for call in tool_calls
    )


@pytest.mark.asyncio
async def test_tool_calls_preserved_when_rails_block():
    test_tool_calls = [
        {
            "name": "dangerous_tool",
            "args": {"param": "eval('malicious code')"},
            "id": "call_dangerous",
            "type": "tool_call",
        }
    ]

    config = RailsConfig.from_content(
        """
        define subflow validate tool parameters
          $valid = execute validate_tool_parameters(tool_calls=$tool_calls)

          if not $valid
            bot refuse dangerous tool parameters
            abort

        define bot refuse dangerous tool parameters
          "I cannot execute this tool request because the parameters may be unsafe."
        """,
        """
        models: []
        passthrough: true
        rails:
          tool_output:
            flows:
              - validate tool parameters
        """,
    )

    class MockLLMWithDangerousTools:
        def invoke(self, messages, **kwargs):
            return AIMessage(content="", tool_calls=test_tool_calls)

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    rails = RunnableRails(config, llm=MockLLMWithDangerousTools())

    rails.rails.runtime.register_action(
        validate_tool_parameters, name="validate_tool_parameters"
    )
    rails.rails.runtime.register_action(
        self_check_tool_calls, name="self_check_tool_calls"
    )
    result = await rails.ainvoke(HumanMessage(content="Execute dangerous tool"))

    assert (
        result.tool_calls is not None
    ), "tool_calls should be preserved in final response"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "dangerous_tool"
    assert "cannot execute this tool request" in result.content


@pytest.mark.asyncio
async def test_generation_action_pops_tool_calls_once():
    from unittest.mock import patch

    test_tool_calls = [
        {
            "name": "test_tool",
            "args": {"param": "value"},
            "id": "call_test",
            "type": "tool_call",
        }
    ]

    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    call_count = 0

    def mock_get_and_clear():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return test_tool_calls
        return None

    with patch(
        "nemoguardrails.actions.llm.utils.get_and_clear_tool_calls_contextvar",
        side_effect=mock_get_and_clear,
    ):
        chat = TestChat(config, llm_completions=[""])

        result = await chat.app.generate_async(
            messages=[{"role": "user", "content": "Test"}]
        )

        assert call_count >= 1, "get_and_clear_tool_calls_contextvar should be called"
        assert result["tool_calls"] is not None
        assert result["tool_calls"][0]["name"] == "test_tool"


@pytest.mark.asyncio
async def test_llmrails_extracts_tool_calls_from_events():
    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    test_tool_calls = [
        {
            "name": "extract_test",
            "args": {"data": "test"},
            "id": "call_extract",
            "type": "tool_call",
        }
    ]

    mock_events = [
        {"type": "BotToolCalls", "tool_calls": test_tool_calls, "uid": "test_uid"}
    ]

    from nemoguardrails.actions.llm.utils import extract_tool_calls_from_events

    extracted_tool_calls = extract_tool_calls_from_events(mock_events)

    assert extracted_tool_calls is not None
    assert len(extracted_tool_calls) == 1
    assert extracted_tool_calls[0]["name"] == "extract_test"


@pytest.mark.asyncio
async def test_tool_rails_cannot_clear_context_variable():
    from nemoguardrails.context import tool_calls_var

    test_tool_calls = [
        {
            "name": "blocked_tool",
            "args": {"param": "rm -rf /"},
            "id": "call_blocked",
            "type": "tool_call",
        }
    ]

    tool_calls_var.set(test_tool_calls)

    context = {"tool_calls": test_tool_calls}
    result = await validate_tool_parameters(test_tool_calls, context=context)

    assert result is False
    assert (
        tool_calls_var.get() is not None
    ), "Context variable should not be cleared by tool rails"
    assert tool_calls_var.get()[0]["name"] == "blocked_tool"


@pytest.mark.asyncio
async def test_complete_fix_integration():
    """Integration test demonstrating the complete fix for tool_calls preservation."""

    dangerous_tool_calls = [
        {
            "name": "dangerous_function",
            "args": {"code": "eval('malicious')"},
            "id": "call_dangerous_123",
            "type": "tool_call",
        }
    ]

    config = RailsConfig.from_content(
        """
        define subflow validate tool parameters
          $valid = execute validate_tool_parameters(tool_calls=$tool_calls)

          if not $valid
            bot refuse dangerous tool parameters
            abort

        define bot refuse dangerous tool parameters
          "I cannot execute this request due to security concerns."
        """,
        """
        models: []
        passthrough: true
        rails:
          tool_output:
            flows:
              - validate tool parameters
        """,
    )

    class MockLLMReturningDangerousTools:
        def invoke(self, messages, **kwargs):
            return AIMessage(content="", tool_calls=dangerous_tool_calls)

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    rails = RunnableRails(config, llm=MockLLMReturningDangerousTools())

    rails.rails.runtime.register_action(
        validate_tool_parameters, name="validate_tool_parameters"
    )
    rails.rails.runtime.register_action(
        self_check_tool_calls, name="self_check_tool_calls"
    )
    result = await rails.ainvoke(HumanMessage(content="Run dangerous code"))

    assert "security concerns" in result.content

    assert result.tool_calls is not None, "tool_calls preserved despite being blocked"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "dangerous_function"


@pytest.mark.asyncio
async def test_passthrough_mode_with_multiple_tool_calls():
    test_tool_calls = [
        {
            "name": "get_weather",
            "args": {"location": "NYC"},
            "id": "call_123",
            "type": "tool_call",
        },
        {
            "name": "calculate",
            "args": {"a": 2, "b": 2},
            "id": "call_456",
            "type": "tool_call",
        },
    ]

    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    class MockLLMWithMultipleTools:
        def invoke(self, messages, **kwargs):
            return AIMessage(
                content="I'll help you with the weather and calculation.",
                tool_calls=test_tool_calls,
            )

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    rails = RunnableRails(config, llm=MockLLMWithMultipleTools())
    result = await rails.ainvoke(
        HumanMessage(content="What's the weather in NYC and what's 2+2?")
    )

    assert result.tool_calls is not None
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0]["name"] == "get_weather"
    assert result.tool_calls[1]["name"] == "calculate"
    assert result.content == ""


@pytest.mark.asyncio
async def test_passthrough_mode_no_tool_calls():
    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    class MockLLMNoTools:
        def invoke(self, messages, **kwargs):
            return AIMessage(content="I can help with general questions.")

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    rails = RunnableRails(config, llm=MockLLMNoTools())
    result = await rails.ainvoke(HumanMessage(content="Hello"))

    assert result.tool_calls is None or result.tool_calls == []
    assert result.content == "I can help with general questions."


@pytest.mark.asyncio
async def test_passthrough_mode_empty_tool_calls():
    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    class MockLLMEmptyTools:
        def invoke(self, messages, **kwargs):
            return AIMessage(content="No tools needed.", tool_calls=[])

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    rails = RunnableRails(config, llm=MockLLMEmptyTools())
    result = await rails.ainvoke(HumanMessage(content="Simple question"))

    assert result.tool_calls == []
    assert result.content == "No tools needed."


@pytest.mark.asyncio
async def test_tool_calls_with_prompt_response():
    test_tool_calls = [
        {
            "name": "search",
            "args": {"query": "latest news"},
            "id": "call_prompt",
            "type": "tool_call",
        }
    ]

    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    class MockLLMPromptMode:
        def invoke(self, messages, **kwargs):
            return AIMessage(content="", tool_calls=test_tool_calls)

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    rails = RunnableRails(config, llm=MockLLMPromptMode())
    result = await rails.ainvoke(HumanMessage(content="Get me the latest news"))

    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "search"
    assert result.tool_calls[0]["args"]["query"] == "latest news"


@pytest.mark.asyncio
async def test_tool_calls_preserve_metadata():
    test_tool_calls = [
        {
            "name": "preserve_test",
            "args": {"data": "preserved"},
            "id": "call_preserve",
            "type": "tool_call",
        }
    ]

    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    class MockLLMWithMetadata:
        def invoke(self, messages, **kwargs):
            msg = AIMessage(
                content="Processing with metadata.", tool_calls=test_tool_calls
            )
            msg.response_metadata = {"model": "test-model", "usage": {"tokens": 50}}
            return msg

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    rails = RunnableRails(config, llm=MockLLMWithMetadata())
    result = await rails.ainvoke(HumanMessage(content="Process with metadata"))

    assert result.tool_calls is not None
    assert result.tool_calls[0]["name"] == "preserve_test"
    assert result.content == ""
    assert hasattr(result, "response_metadata")


@pytest.mark.asyncio
async def test_tool_output_rails_blocking_behavior():
    dangerous_tool_calls = [
        {
            "name": "dangerous_exec",
            "args": {"command": "rm -rf /"},
            "id": "call_dangerous_exec",
            "type": "tool_call",
        }
    ]

    config = RailsConfig.from_content(
        """
        define subflow validate tool parameters
          $valid = execute validate_tool_parameters(tool_calls=$tool_calls)

          if not $valid
            bot refuse dangerous tool parameters
            abort

        define bot refuse dangerous tool parameters
          "Tool blocked for security reasons."
        """,
        """
        models: []
        passthrough: true
        rails:
          tool_output:
            flows:
              - validate tool parameters
        """,
    )

    class MockLLMDangerousExec:
        def invoke(self, messages, **kwargs):
            return AIMessage(content="", tool_calls=dangerous_tool_calls)

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    rails = RunnableRails(config, llm=MockLLMDangerousExec())

    rails.rails.runtime.register_action(
        validate_tool_parameters, name="validate_tool_parameters"
    )
    rails.rails.runtime.register_action(
        self_check_tool_calls, name="self_check_tool_calls"
    )
    result = await rails.ainvoke(HumanMessage(content="Execute dangerous command"))

    assert "security reasons" in result.content
    assert result.tool_calls is not None
    assert result.tool_calls[0]["name"] == "dangerous_exec"
    assert "rm -rf" in result.tool_calls[0]["args"]["command"]


@pytest.mark.asyncio
async def test_complex_tool_calls_integration():
    complex_tool_calls = [
        {
            "name": "search_database",
            "args": {"table": "users", "query": "active=true"},
            "id": "call_db_search",
            "type": "tool_call",
        },
        {
            "name": "format_results",
            "args": {"format": "json", "limit": 10},
            "id": "call_format",
            "type": "tool_call",
        },
    ]

    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    class MockLLMComplexTools:
        def invoke(self, messages, **kwargs):
            return AIMessage(
                content="I'll search the database and format the results.",
                tool_calls=complex_tool_calls,
            )

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    rails = RunnableRails(config, llm=MockLLMComplexTools())
    result = await rails.ainvoke(
        HumanMessage(content="Find active users and format as JSON")
    )

    assert result.tool_calls is not None
    assert len(result.tool_calls) == 2

    db_call = result.tool_calls[0]
    assert db_call["name"] == "search_database"
    assert db_call["args"]["table"] == "users"
    assert db_call["args"]["query"] == "active=true"

    format_call = result.tool_calls[1]
    assert format_call["name"] == "format_results"
    assert format_call["args"]["format"] == "json"
    assert format_call["args"]["limit"] == 10

    assert result.content == ""

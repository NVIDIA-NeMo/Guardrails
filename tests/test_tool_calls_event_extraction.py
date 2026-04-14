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

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallFunction
from tests.utils import FakeLLMModel, TestChat


@action(is_system_action=True)
async def validate_tool_parameters(tool_calls, context=None, **kwargs):
    tool_calls = tool_calls or (context.get("tool_calls", []) if context else [])

    dangerous_patterns = ["eval", "exec", "system", "../", "rm -", "DROP", "DELETE"]

    for tool_call in tool_calls:
        func = tool_call.get("function", {})
        args = func.get("arguments", {})
        for param_value in args.values():
            if isinstance(param_value, str):
                if any(pattern.lower() in param_value.lower() for pattern in dangerous_patterns):
                    return False
    return True


@action(is_system_action=True)
async def self_check_tool_calls(tool_calls, context=None, **kwargs):
    tool_calls = tool_calls or (context.get("tool_calls", []) if context else [])

    return all(isinstance(call, dict) and "function" in call and "id" in call for call in tool_calls)


@pytest.mark.asyncio
async def test_tool_calls_preserved_when_rails_block():
    test_tool_calls = [
        ToolCall(
            id="call_dangerous",
            type="function",
            function=ToolCallFunction(name="dangerous_tool", arguments={"param": "eval('malicious code')"}),
        )
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

    fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="", tool_calls=test_tool_calls)])
    rails = LLMRails(config, llm=fake_llm)

    rails.runtime.register_action(validate_tool_parameters, name="validate_tool_parameters")
    rails.runtime.register_action(self_check_tool_calls, name="self_check_tool_calls")
    result = await rails.generate_async(messages=[{"role": "user", "content": "Execute dangerous tool"}])

    assert result["tool_calls"] is not None, "tool_calls should be preserved in final response"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "dangerous_tool"
    assert "cannot execute this tool request" in result["content"]


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

        result = await chat.app.generate_async(messages=[{"role": "user", "content": "Test"}])

        assert call_count >= 1, "get_and_clear_tool_calls_contextvar should be called"
        assert result["tool_calls"] is not None
        assert result["tool_calls"][0]["name"] == "test_tool"


@pytest.mark.asyncio
async def test_llmrails_extracts_tool_calls_from_events():
    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    test_tool_calls = [
        {
            "id": "call_extract",
            "type": "function",
            "function": {
                "name": "extract_test",
                "arguments": {"data": "test"},
            },
        }
    ]

    mock_events = [{"type": "BotToolCalls", "tool_calls": test_tool_calls, "uid": "test_uid"}]

    from nemoguardrails.actions.llm.utils import extract_tool_calls_from_events

    extracted_tool_calls = extract_tool_calls_from_events(mock_events)

    assert extracted_tool_calls is not None
    assert len(extracted_tool_calls) == 1
    assert extracted_tool_calls[0]["function"]["name"] == "extract_test"


@pytest.mark.asyncio
async def test_tool_rails_cannot_clear_context_variable():
    from nemoguardrails.context import tool_calls_var

    test_tool_calls = [
        {
            "id": "call_blocked",
            "type": "function",
            "function": {
                "name": "blocked_tool",
                "arguments": {"param": "rm -rf /"},
            },
        }
    ]

    tool_calls_var.set(test_tool_calls)

    context = {"tool_calls": test_tool_calls}
    result = await validate_tool_parameters(test_tool_calls, context=context)

    assert result is False
    assert tool_calls_var.get() is not None, "Context variable should not be cleared by tool rails"
    assert tool_calls_var.get()[0]["function"]["name"] == "blocked_tool"


@pytest.mark.asyncio
async def test_complete_fix_integration():
    dangerous_tool_calls = [
        ToolCall(
            id="call_dangerous_123",
            type="function",
            function=ToolCallFunction(name="dangerous_function", arguments={"code": "eval('malicious')"}),
        )
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

    fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="", tool_calls=dangerous_tool_calls)])
    rails = LLMRails(config, llm=fake_llm)

    rails.runtime.register_action(validate_tool_parameters, name="validate_tool_parameters")
    rails.runtime.register_action(self_check_tool_calls, name="self_check_tool_calls")
    result = await rails.generate_async(messages=[{"role": "user", "content": "Run dangerous code"}])

    assert "security concerns" in result["content"]

    assert result["tool_calls"] is not None, "tool_calls preserved despite being blocked"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "dangerous_function"


@pytest.mark.asyncio
async def test_passthrough_mode_with_multiple_tool_calls():
    test_tool_calls = [
        ToolCall(
            id="call_123",
            type="function",
            function=ToolCallFunction(name="get_weather", arguments={"location": "NYC"}),
        ),
        ToolCall(
            id="call_456",
            type="function",
            function=ToolCallFunction(name="calculate", arguments={"a": 2, "b": 2}),
        ),
    ]

    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    fake_llm = FakeLLMModel(
        llm_responses=[
            LLMResponse(
                content="I'll help you with the weather and calculation.",
                tool_calls=test_tool_calls,
            )
        ]
    )
    rails = LLMRails(config, llm=fake_llm)
    result = await rails.generate_async(
        messages=[{"role": "user", "content": "What's the weather in NYC and what's 2+2?"}]
    )

    assert result["tool_calls"] is not None
    assert len(result["tool_calls"]) == 2
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"
    assert result["tool_calls"][1]["function"]["name"] == "calculate"
    assert result["content"] == ""


@pytest.mark.asyncio
async def test_passthrough_mode_no_tool_calls():
    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="I can help with general questions.")])
    rails = LLMRails(config, llm=fake_llm)
    result = await rails.generate_async(messages=[{"role": "user", "content": "Hello"}])

    assert result.get("tool_calls") is None or result.get("tool_calls") == []
    assert result["content"] == "I can help with general questions."


@pytest.mark.asyncio
async def test_passthrough_mode_empty_tool_calls():
    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="No tools needed.", tool_calls=[])])
    rails = LLMRails(config, llm=fake_llm)
    result = await rails.generate_async(messages=[{"role": "user", "content": "Simple question"}])

    assert result.get("tool_calls") == [] or result.get("tool_calls") is None
    assert result["content"] == "No tools needed."


@pytest.mark.asyncio
async def test_tool_calls_with_prompt_response():
    test_tool_calls = [
        ToolCall(
            id="call_prompt",
            type="function",
            function=ToolCallFunction(name="search", arguments={"query": "latest news"}),
        )
    ]

    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="", tool_calls=test_tool_calls)])
    rails = LLMRails(config, llm=fake_llm)
    result = await rails.generate_async(messages=[{"role": "user", "content": "Get me the latest news"}])

    assert result["tool_calls"] is not None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "search"
    assert result["tool_calls"][0]["function"]["arguments"]["query"] == "latest news"


@pytest.mark.asyncio
async def test_tool_calls_preserve_metadata():
    test_tool_calls = [
        ToolCall(
            id="call_preserve",
            type="function",
            function=ToolCallFunction(name="preserve_test", arguments={"data": "preserved"}),
        )
    ]

    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    fake_llm = FakeLLMModel(
        llm_responses=[
            LLMResponse(
                content="Processing with metadata.",
                tool_calls=test_tool_calls,
                provider_metadata={"model": "test-model", "usage": {"tokens": 50}},
            )
        ]
    )
    rails = LLMRails(config, llm=fake_llm)
    result = await rails.generate_async(messages=[{"role": "user", "content": "Process with metadata"}])

    assert result["tool_calls"] is not None
    assert result["tool_calls"][0]["function"]["name"] == "preserve_test"
    assert result["content"] == ""


@pytest.mark.asyncio
async def test_tool_output_rails_blocking_behavior():
    dangerous_tool_calls = [
        ToolCall(
            id="call_dangerous_exec",
            type="function",
            function=ToolCallFunction(name="dangerous_exec", arguments={"command": "rm -rf /"}),
        )
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

    fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="", tool_calls=dangerous_tool_calls)])
    rails = LLMRails(config, llm=fake_llm)

    rails.runtime.register_action(validate_tool_parameters, name="validate_tool_parameters")
    rails.runtime.register_action(self_check_tool_calls, name="self_check_tool_calls")
    result = await rails.generate_async(messages=[{"role": "user", "content": "Execute dangerous command"}])

    assert "security reasons" in result["content"]
    assert result["tool_calls"] is not None
    assert result["tool_calls"][0]["function"]["name"] == "dangerous_exec"
    assert "rm -rf" in result["tool_calls"][0]["function"]["arguments"]["command"]


@pytest.mark.asyncio
async def test_complex_tool_calls_integration():
    complex_tool_calls = [
        ToolCall(
            id="call_db_search",
            type="function",
            function=ToolCallFunction(name="search_database", arguments={"table": "users", "query": "active=true"}),
        ),
        ToolCall(
            id="call_format",
            type="function",
            function=ToolCallFunction(name="format_results", arguments={"format": "json", "limit": 10}),
        ),
    ]

    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    fake_llm = FakeLLMModel(
        llm_responses=[
            LLMResponse(
                content="I'll search the database and format the results.",
                tool_calls=complex_tool_calls,
            )
        ]
    )
    rails = LLMRails(config, llm=fake_llm)
    result = await rails.generate_async(messages=[{"role": "user", "content": "Find active users and format as JSON"}])

    assert result["tool_calls"] is not None
    assert len(result["tool_calls"]) == 2

    db_call = result["tool_calls"][0]
    assert db_call["function"]["name"] == "search_database"
    assert db_call["function"]["arguments"]["table"] == "users"
    assert db_call["function"]["arguments"]["query"] == "active=true"

    format_call = result["tool_calls"][1]
    assert format_call["function"]["name"] == "format_results"
    assert format_call["function"]["arguments"]["format"] == "json"
    assert format_call["function"]["arguments"]["limit"] == 10

    assert result["content"] == ""

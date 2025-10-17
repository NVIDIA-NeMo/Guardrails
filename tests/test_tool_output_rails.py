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

"""Tests for tool output rails (Phase 2) functionality."""

from unittest.mock import patch

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
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
async def test_tool_output_rails_basic():
    """Test basic tool output rails functionality."""

    test_tool_calls = [
        {
            "name": "allowed_tool",
            "args": {"param": "safe_value"},
            "id": "call_safe",
            "type": "tool_call",
        }
    ]

    # Config with tool output rails
    config = RailsConfig.from_content(
        """
        define subflow self check tool calls
          $allowed = execute self_check_tool_calls(tool_calls=$tool_calls)

          if not $allowed
            bot refuse tool execution
            abort

        define bot refuse tool execution
          "I cannot execute this tool request due to policy restrictions."
        """,
        """
        models: []
        passthrough: true
        rails:
          tool_output:
            flows:
              - self check tool calls
        """,
    )

    with patch(
        "nemoguardrails.actions.llm.utils.get_and_clear_tool_calls_contextvar"
    ) as mock_get_clear:
        mock_get_clear.return_value = test_tool_calls

        chat = TestChat(config, llm_completions=[""])

        chat.app.runtime.register_action(
            validate_tool_parameters, name="validate_tool_parameters"
        )
        chat.app.runtime.register_action(
            self_check_tool_calls, name="self_check_tool_calls"
        )

        result = await chat.app.generate_async(
            messages=[{"role": "user", "content": "Use allowed tool"}]
        )

        # Tool should be allowed through
        assert result["tool_calls"] is not None
        assert result["tool_calls"][0]["name"] == "allowed_tool"


@pytest.mark.asyncio
async def test_tool_output_rails_blocking():
    """Test that tool output rails can block dangerous tools."""

    test_tool_calls = [
        {
            "name": "dangerous_tool",
            "args": {"param": "eval('malicious code')"},
            "id": "call_bad",
            "type": "tool_call",
        }
    ]

    # Config with tool parameter validation
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

    # Create a mock LLM that returns tool calls
    class MockLLMWithDangerousTool:
        def invoke(self, messages, **kwargs):
            from langchain_core.messages import AIMessage

            return AIMessage(content="", tool_calls=test_tool_calls)

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    from langchain_core.messages import HumanMessage

    from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

    rails = RunnableRails(config, llm=MockLLMWithDangerousTool())

    rails.rails.runtime.register_action(
        validate_tool_parameters, name="validate_tool_parameters"
    )
    rails.rails.runtime.register_action(
        self_check_tool_calls, name="self_check_tool_calls"
    )

    result = await rails.ainvoke(HumanMessage(content="Use dangerous tool"))

    assert "parameters may be unsafe" in result.content


@pytest.mark.asyncio
async def test_multiple_tool_output_rails():
    """Test multiple tool output rails working together."""

    test_tool_calls = [
        {
            "name": "test_tool",
            "args": {"param": "safe"},
            "id": "call_test",
            "type": "tool_call",
        }
    ]

    config = RailsConfig.from_content(
        """
        define subflow self check tool calls
          $allowed = execute self_check_tool_calls(tool_calls=$tool_calls)
          if not $allowed
            bot refuse tool execution
            abort

        define subflow validate tool parameters
          $valid = execute validate_tool_parameters(tool_calls=$tool_calls)
          if not $valid
            bot refuse dangerous tool parameters
            abort

        define bot refuse tool execution
          "Tool not allowed."

        define bot refuse dangerous tool parameters
          "Parameters unsafe."
        """,
        """
        models: []
        passthrough: true
        rails:
          tool_output:
            flows:
              - self check tool calls
              - validate tool parameters
        """,
    )

    with patch(
        "nemoguardrails.actions.llm.utils.get_and_clear_tool_calls_contextvar"
    ) as mock_get_clear:
        mock_get_clear.return_value = test_tool_calls

        chat = TestChat(config, llm_completions=[""])

        chat.app.runtime.register_action(
            validate_tool_parameters, name="validate_tool_parameters"
        )
        chat.app.runtime.register_action(
            self_check_tool_calls, name="self_check_tool_calls"
        )

        result = await chat.app.generate_async(
            messages=[{"role": "user", "content": "Use test tool"}]
        )

        assert result["tool_calls"] is not None
        assert result["tool_calls"][0]["name"] == "test_tool"

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

from unittest.mock import patch

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
async def test_tool_output_rails_basic():
    test_tool_calls = [
        {
            "name": "allowed_tool",
            "args": {"param": "safe_value"},
            "id": "call_safe",
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

    with patch("nemoguardrails.actions.llm.utils.get_and_clear_tool_calls_contextvar") as mock_get_clear:
        mock_get_clear.return_value = test_tool_calls

        chat = TestChat(config, llm_completions=[""])

        chat.app.runtime.register_action(validate_tool_parameters, name="validate_tool_parameters")
        chat.app.runtime.register_action(self_check_tool_calls, name="self_check_tool_calls")

        result = await chat.app.generate_async(messages=[{"role": "user", "content": "Use allowed tool"}])

        assert result["tool_calls"] is not None
        assert result["tool_calls"][0]["name"] == "allowed_tool"


@pytest.mark.asyncio
async def test_tool_output_rails_blocking():
    dangerous_tool_call_objects = [
        ToolCall(
            id="call_bad",
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

    fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="", tool_calls=dangerous_tool_call_objects)])
    rails = LLMRails(config, llm=fake_llm)

    rails.runtime.register_action(validate_tool_parameters, name="validate_tool_parameters")
    rails.runtime.register_action(self_check_tool_calls, name="self_check_tool_calls")

    result = await rails.generate_async(messages=[{"role": "user", "content": "Use dangerous tool"}])

    assert "parameters may be unsafe" in result["content"]


@pytest.mark.asyncio
async def test_multiple_tool_output_rails():
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

    with patch("nemoguardrails.actions.llm.utils.get_and_clear_tool_calls_contextvar") as mock_get_clear:
        mock_get_clear.return_value = test_tool_calls

        chat = TestChat(config, llm_completions=[""])

        chat.app.runtime.register_action(validate_tool_parameters, name="validate_tool_parameters")
        chat.app.runtime.register_action(self_check_tool_calls, name="self_check_tool_calls")

        result = await chat.app.generate_async(messages=[{"role": "user", "content": "Use test tool"}])

        assert result["tool_calls"] is not None
        assert result["tool_calls"][0]["name"] == "test_tool"

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

"""Tests for BotToolCalls event handling in NeMo Guardrails."""

from unittest.mock import patch

import pytest

from nemoguardrails import RailsConfig
from tests.utils import TestChat


@pytest.mark.asyncio
async def test_bot_tool_call_event_creation():
    """Test that BotToolCalls events are created when tool_calls are present."""

    test_tool_calls = [
        {
            "name": "test_function",
            "args": {"param1": "value1"},
            "id": "call_12345",
            "type": "tool_call",
        }
    ]

    with patch(
        "nemoguardrails.actions.llm.utils.get_and_clear_tool_calls_contextvar"
    ) as mock_get_clear:
        mock_get_clear.return_value = test_tool_calls

        config = RailsConfig.from_content(config={"models": [], "passthrough": True})
        chat = TestChat(config, llm_completions=[""])

        result = await chat.app.generate_async(
            messages=[{"role": "user", "content": "Test"}]
        )

        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "test_function"


@pytest.mark.asyncio
async def test_bot_message_vs_bot_tool_call_event():
    """Test that regular text creates BotMessage, tool calls create BotToolCalls."""

    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    with patch(
        "nemoguardrails.actions.llm.utils.get_and_clear_tool_calls_contextvar"
    ) as mock_get_clear:
        mock_get_clear.return_value = None

        chat_text = TestChat(config, llm_completions=["Regular text response"])
        result_text = await chat_text.app.generate_async(
            messages=[{"role": "user", "content": "Hello"}]
        )

        assert result_text["content"] == "Regular text response"
        assert (
            result_text.get("tool_calls") is None or result_text.get("tool_calls") == []
        )

    test_tool_calls = [
        {
            "name": "toggle_tool",
            "args": {},
            "id": "call_toggle",
            "type": "tool_call",
        }
    ]

    with patch(
        "nemoguardrails.actions.llm.utils.get_and_clear_tool_calls_contextvar"
    ) as mock_get_clear:
        mock_get_clear.return_value = test_tool_calls

        chat_tools = TestChat(config, llm_completions=[""])
        result_tools = await chat_tools.app.generate_async(
            messages=[{"role": "user", "content": "Use tool"}]
        )

        assert result_tools["tool_calls"] is not None
        assert result_tools["tool_calls"][0]["name"] == "toggle_tool"


@pytest.mark.asyncio
async def test_tool_calls_bypass_output_rails():
    """Test that tool calls bypass output rails in passthrough mode."""

    test_tool_calls = [
        {
            "name": "critical_tool",
            "args": {"action": "execute"},
            "id": "call_critical",
            "type": "tool_call",
        }
    ]

    config = RailsConfig.from_content(
        """
        define flow block_empty_content
          if $bot_message == ""
            bot refuse to respond
            stop
        """,
        """
        models: []
        passthrough: true
        rails:
          output:
            flows:
              - block_empty_content
        """,
    )

    with patch(
        "nemoguardrails.actions.llm.utils.get_and_clear_tool_calls_contextvar"
    ) as mock_get_clear:
        mock_get_clear.return_value = test_tool_calls

        chat = TestChat(config, llm_completions=[""])
        result = await chat.app.generate_async(
            messages=[{"role": "user", "content": "Execute"}]
        )

        assert result["tool_calls"] is not None
        assert result["tool_calls"][0]["name"] == "critical_tool"


@pytest.mark.asyncio
async def test_mixed_content_and_tool_calls():
    """Test responses that have both content and tool calls."""

    test_tool_calls = [
        {
            "name": "transmit_data",
            "args": {"info": "user_data"},
            "id": "call_transmit",
            "type": "tool_call",
        }
    ]

    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    with patch(
        "nemoguardrails.actions.llm.utils.get_and_clear_tool_calls_contextvar"
    ) as mock_get_clear:
        mock_get_clear.return_value = test_tool_calls

        chat = TestChat(
            config,
            llm_completions=["I found the information and will now transmit it."],
        )
        result = await chat.app.generate_async(
            messages=[{"role": "user", "content": "Process data"}]
        )

        assert result["tool_calls"] is not None
        assert result["tool_calls"][0]["name"] == "transmit_data"


@pytest.mark.asyncio
async def test_multiple_tool_calls():
    """Test handling of multiple tool calls in a single response."""

    test_tool_calls = [
        {
            "name": "tool_one",
            "args": {"param": "first"},
            "id": "call_one",
            "type": "tool_call",
        },
        {
            "name": "tool_two",
            "args": {"param": "second"},
            "id": "call_two",
            "type": "tool_call",
        },
    ]

    config = RailsConfig.from_content(config={"models": [], "passthrough": True})

    with patch(
        "nemoguardrails.actions.llm.utils.get_and_clear_tool_calls_contextvar"
    ) as mock_get_clear:
        mock_get_clear.return_value = test_tool_calls

        chat = TestChat(config, llm_completions=[""])
        result = await chat.app.generate_async(
            messages=[{"role": "user", "content": "Execute multiple tools"}]
        )

        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["name"] == "tool_one"
        assert result["tool_calls"][1]["name"] == "tool_two"


@pytest.mark.asyncio
async def test_regular_text_still_goes_through_output_rails():
    """Test that regular text responses still go through output rails."""

    config = RailsConfig.from_content(
        """
        define flow add_prefix
          $bot_message = "PREFIX: " + $bot_message
        """,
        """
        rails:
          output:
            flows:
              - add_prefix
        """,
    )

    with patch(
        "nemoguardrails.actions.llm.utils.get_and_clear_tool_calls_contextvar"
    ) as mock_get_clear:
        mock_get_clear.return_value = None

        chat = TestChat(config, llm_completions=["This is a regular response"])
        result = await chat.app.generate_async(
            messages=[{"role": "user", "content": "Say something"}]
        )

        assert "PREFIX: This is a regular response" in result["content"]
        assert result.get("tool_calls") is None or result.get("tool_calls") == []


@pytest.mark.asyncio
async def test_empty_text_without_tool_calls_still_blocked():
    """Test that empty text without tool_calls is still blocked by output rails."""

    config = RailsConfig.from_content(
        """
        define flow block_empty
          if $bot_message == ""
            bot refuse to respond
            stop
        """,
        """
        rails:
          output:
            flows:
              - block_empty
        """,
    )

    with patch(
        "nemoguardrails.actions.llm.utils.get_and_clear_tool_calls_contextvar"
    ) as mock_get_clear:
        mock_get_clear.return_value = None

        chat = TestChat(config, llm_completions=[""])
        result = await chat.app.generate_async(
            messages=[{"role": "user", "content": "Say something"}]
        )

        assert "I'm sorry, I can't respond to that." in result["content"]
        assert result.get("tool_calls") is None or result.get("tool_calls") == []

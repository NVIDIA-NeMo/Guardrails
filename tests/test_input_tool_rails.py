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

"""Tests for input tool rails functionality.

This module tests the input tool rails functionality introduced to validate and secure
tool inputs/results before they are processed by the LLM. Since the @nemoguardrails/library/tool_check/
actions and flows will be removed, this test file implements similar test actions and flows
to ensure input tool rails work as expected.
"""

import logging
from unittest.mock import patch

import pytest

from nemoguardrails import RailsConfig
from tests.input_tool_rails_actions import (
    sanitize_tool_input,
    self_check_tool_input,
    validate_tool_input_safety,
)
from tests.utils import TestChat

log = logging.getLogger(__name__)


class TestInputToolRails:
    """Test class for input tool rails functionality."""

    @pytest.mark.asyncio
    async def test_user_tool_messages_event_direct_processing(self):
        """Test that UserToolMessages events work correctly when created directly.

        This tests the core tool input rails functionality by creating UserToolMessages
        events directly, which should work according to the commit changes.
        """
        config = RailsConfig.from_content(
            """
            define flow self check tool input
              $allowed = execute test_self_check_tool_input(tool_message=$tool_message, tool_name=$tool_name, tool_call_id=$tool_call_id)

              if not $allowed
                bot refuse tool input
                abort

            define bot refuse tool input
              "Tool input validation failed via direct event."
            """,
            """
            models: []
            passthrough: true
            rails:
              tool_input:
                flows:
                  - self check tool input
            """,
        )

        chat = TestChat(config, llm_completions=["Should not be reached"])

        chat.app.runtime.register_action(self_check_tool_input, name="test_self_check_tool_input")

        from nemoguardrails.utils import new_event_dict

        tool_messages = [
            {
                "content": "Sunny, 22°C",
                "name": "get_weather",
                "tool_call_id": "call_weather_001",
            }
        ]

        events = [
            new_event_dict(
                "UserToolMessages",
                tool_messages=tool_messages,
            )
        ]
        result_events = await chat.app.runtime.generate_events(events)

        tool_input_rails_finished = any(event.get("type") == "ToolInputRailsFinished" for event in result_events)
        assert tool_input_rails_finished, (
            "Expected ToolInputRailsFinished event to be generated after successful tool input validation"
        )

        invalid_tool_messages = [
            {
                "content": "Sunny, 22°C",
                "name": "get_weather",
            }
        ]

        invalid_events = [
            new_event_dict(
                "UserToolMessages",
                tool_messages=invalid_tool_messages,
            )
        ]
        invalid_result_events = await chat.app.runtime.generate_events(invalid_events)

        blocked_found = any(
            event.get("type") == "BotMessage" and "validation failed" in event.get("text", "")
            for event in invalid_result_events
        )
        assert blocked_found, f"Expected tool input to be blocked, got events: {invalid_result_events}"

    @pytest.mark.asyncio
    async def test_message_to_event_conversion_fixed(self):
        """Test that message-to-event conversion for tool messages now works correctly.

        This test verifies that the automatic conversion from conversation messages
        to UserToolMessages events is working correctly after the fix.
        """
        config = RailsConfig.from_content(
            """
            define flow self check tool input
              $allowed = execute test_self_check_tool_input(tool_message=$tool_message, tool_name=$tool_name, tool_call_id=$tool_call_id)

              if not $allowed
                bot refuse tool input
                abort

            define bot refuse tool input
              "Tool input blocked via message processing."
            """,
            """
            models: []
            passthrough: true
            rails:
              tool_input:
                flows:
                  - self check tool input
            """,
        )

        chat = TestChat(config, llm_completions=["Normal LLM response"])

        chat.app.runtime.register_action(self_check_tool_input, name="test_self_check_tool_input")

        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                        "id": "call_weather_001",
                        "type": "tool_call",
                    }
                ],
            },
            {
                "role": "tool",
                "content": "",
                "name": "get_weather",
                "tool_call_id": "call_weather_001",
            },
        ]

        result = await chat.app.generate_async(messages=messages)

        assert "Tool input blocked" in result["content"], f"Expected tool input to be blocked, got: {result['content']}"

    @pytest.mark.asyncio
    async def test_tool_input_validation_blocking(self):
        """Test that tool input validation can block invalid tool responses."""
        config = RailsConfig.from_content(
            """
            define flow self check tool input
              $allowed = execute test_self_check_tool_input(tool_message=$tool_message, tool_name=$tool_name, tool_call_id=$tool_call_id)

              if not $allowed
                bot refuse tool input
                abort

            define bot refuse tool input
              "I cannot process this tool response due to validation issues."
            """,
            """
            models: []
            passthrough: true
            rails:
              tool_input:
                flows:
                  - self check tool input
            """,
        )

        chat = TestChat(config, llm_completions=[""])

        chat.app.runtime.register_action(self_check_tool_input, name="test_self_check_tool_input")

        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                        "id": "call_weather_001",
                        "type": "tool_call",
                    }
                ],
            },
            {
                "role": "tool",
                "content": "Sunny, 22°C",
                "name": "get_weather",
            },
        ]

        result = await chat.app.generate_async(messages=messages)

        assert "validation issues" in result["content"], (
            f"Expected validation to block missing tool_call_id, got: {result['content']}"
        )

    @pytest.mark.asyncio
    async def test_tool_input_safety_validation(self):
        """Test tool input safety validation blocks dangerous content."""
        config = RailsConfig.from_content(
            """
            define flow validate tool input safety
              $safe = execute test_validate_tool_input_safety(tool_message=$tool_message, tool_name=$tool_name)

              if not $safe
                bot refuse unsafe tool input
                abort

            define bot refuse unsafe tool input
              "I cannot process this tool response due to safety concerns."
            """,
            """
            models: []
            passthrough: true
            rails:
              tool_input:
                flows:
                  - validate tool input safety
            """,
        )

        chat = TestChat(config, llm_completions=[""])

        chat.app.runtime.register_action(validate_tool_input_safety, name="test_validate_tool_input_safety")

        messages = [
            {"role": "user", "content": "Get my credentials"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_credentials",
                        "args": {},
                        "id": "call_creds_001",
                        "type": "tool_call",
                    }
                ],
            },
            {
                "role": "tool",
                "content": "Your api_key is sk-1234567890abcdef and password is secret123",
                "name": "get_credentials",
                "tool_call_id": "call_creds_001",
            },
        ]

        result = await chat.app.generate_async(messages=messages)

        assert "safety concerns" in result["content"]

    @pytest.mark.asyncio
    async def test_tool_input_sanitization(self):
        """Test tool input sanitization processes sensitive information without blocking.

        This test verifies that the sanitization rail runs on tool inputs containing
        sensitive data and processes them appropriately without blocking the conversation.
        """
        config = RailsConfig.from_content(
            """
            define flow sanitize tool input
              $sanitized = execute test_sanitize_tool_input(tool_message=$tool_message, tool_name=$tool_name)
              $tool_message = $sanitized

            define flow self check tool input
              $allowed = execute test_self_check_tool_input(tool_message=$tool_message, tool_name=$tool_name, tool_call_id=$tool_call_id)
              if not $allowed
                bot refuse tool input
                abort

            define bot refuse tool input
              "I cannot process this tool response."
            """,
            """
            models: []
            passthrough: true
            rails:
              tool_input:
                flows:
                  - sanitize tool input
                  - self check tool input
            """,
        )

        chat = TestChat(
            config,
            llm_completions=["I found your account information from the database."],
        )

        chat.app.runtime.register_action(sanitize_tool_input, name="test_sanitize_tool_input")
        chat.app.runtime.register_action(self_check_tool_input, name="test_self_check_tool_input")

        messages = [
            {"role": "user", "content": "Look up my account"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "lookup_account",
                        "args": {},
                        "id": "call_lookup_001",
                        "type": "tool_call",
                    }
                ],
            },
            {
                "role": "tool",
                "content": "User email: john.doe@example.com, API token = abcd1234567890xyzABC",
                "name": "lookup_account",
                "tool_call_id": "call_lookup_001",
            },
        ]

        sanitized_result = await sanitize_tool_input(
            tool_message="User email: john.doe@example.com, API token = abcd1234567890xyzABC",
            tool_name="lookup_account",
        )

        assert "[USER]@example.com" in sanitized_result, f"Email not sanitized: {sanitized_result}"
        assert "[REDACTED]" in sanitized_result, f"API token not sanitized: {sanitized_result}"
        assert "john.doe" not in sanitized_result, f"Username not masked: {sanitized_result}"
        assert "abcd1234567890xyzABC" not in sanitized_result, f"API token not masked: {sanitized_result}"

        result = await chat.app.generate_async(messages=messages)

        assert "cannot process" not in result["content"].lower(), f"Unexpected blocking: {result['content']}"

    @pytest.mark.asyncio
    async def test_multiple_tool_input_rails(self):
        """Test multiple tool input rails working together."""
        config = RailsConfig.from_content(
            """
            define flow self check tool input
              $allowed = execute test_self_check_tool_input(tool_message=$tool_message, tool_name=$tool_name, tool_call_id=$tool_call_id)
              if not $allowed
                bot refuse tool input
                abort

            define flow validate tool input safety
              $safe = execute test_validate_tool_input_safety(tool_message=$tool_message, tool_name=$tool_name)
              if not $safe
                bot refuse unsafe tool input
                abort

            define bot refuse tool input
              "Tool validation failed."

            define bot refuse unsafe tool input
              "Tool safety check failed."
            """,
            """
            models: []
            passthrough: true
            rails:
              tool_input:
                flows:
                  - self check tool input
                  - validate tool input safety
            """,
        )

        chat = TestChat(
            config,
            llm_completions=["The weather information shows it's sunny."],
        )

        chat.app.runtime.register_action(self_check_tool_input, name="test_self_check_tool_input")
        chat.app.runtime.register_action(validate_tool_input_safety, name="test_validate_tool_input_safety")

        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                        "id": "call_weather_001",
                        "type": "tool_call",
                    }
                ],
            },
            {
                "role": "tool",
                "content": "Sunny, 22°C in Paris",
                "name": "get_weather",
                "tool_call_id": "call_weather_001",
            },
        ]

        from nemoguardrails.utils import new_event_dict

        events = [
            new_event_dict(
                "UserToolMessages",
                tool_messages=[
                    {
                        "content": "Sunny, 22°C",
                        "name": "get_weather",
                        "tool_call_id": "call_weather_001",
                    }
                ],
            )
        ]

        result_events = await chat.app.runtime.generate_events(events)

        safety_rail_finished = any(
            event.get("type") == "ToolInputRailFinished" and event.get("flow_id") == "validate tool input safety"
            for event in result_events
        )
        validation_rail_finished = any(
            event.get("type") == "ToolInputRailFinished" and event.get("flow_id") == "self check tool input"
            for event in result_events
        )

        assert safety_rail_finished, "Safety rail should have completed"
        assert validation_rail_finished, "Validation rail should have completed"

    @pytest.mark.asyncio
    async def test_multiple_tool_messages_processing(self):
        """Test processing multiple tool messages in UserToolMessages event."""
        config = RailsConfig.from_content(
            """
            define flow self check tool input
              $allowed = execute test_self_check_tool_input(tool_message=$tool_message, tool_name=$tool_name, tool_call_id=$tool_call_id)
              if not $allowed
                bot refuse tool input
                abort

            define bot refuse tool input
              "Tool validation failed."
            """,
            """
            models:
              - type: main
                engine: mock
                model: test-model
            rails:
              tool_input:
                flows:
                  - self check tool input
            """,
        )

        chat = TestChat(
            config,
            llm_completions=["The weather is sunny in Paris and AAPL stock is at $150.25."],
        )

        chat.app.runtime.register_action(self_check_tool_input, name="test_self_check_tool_input")

        messages = [
            {
                "role": "user",
                "content": "Get weather for Paris and stock price for AAPL",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                        "id": "call_weather_001",
                        "type": "tool_call",
                    },
                    {
                        "name": "get_stock_price",
                        "args": {"symbol": "AAPL"},
                        "id": "call_stock_001",
                        "type": "tool_call",
                    },
                ],
            },
            {
                "role": "tool",
                "content": "Sunny, 22°C",
                "name": "get_weather",
                "tool_call_id": "call_weather_001",
            },
            {
                "role": "tool",
                "content": "$150.25",
                "name": "get_stock_price",
                "tool_call_id": "call_stock_001",
            },
        ]

        result = await chat.app.generate_async(messages=messages)

        assert "validation issues" not in result["content"], f"Unexpected validation block: {result['content']}"

    @pytest.mark.asyncio
    async def test_tool_input_rails_with_allowed_tools_config(self):
        """Test tool input rails respecting allowed tools configuration."""

        class CustomConfig:
            def __init__(self):
                self.allowed_tools = ["get_weather", "get_time"]
                self.max_tool_message_length = 10000

        config = RailsConfig.from_content(
            """
            define flow self check tool input
              $allowed = execute test_self_check_tool_input(tool_message=$tool_message, tool_name=$tool_name, tool_call_id=$tool_call_id)
              if not $allowed
                bot refuse tool input
                abort

            define bot refuse tool input
              "Tool not allowed."
            """,
            """
            models:
              - type: main
                engine: mock
                model: test-model
            rails:
              tool_input:
                flows:
                  - self check tool input
            """,
        )

        chat = TestChat(config, llm_completions=[""])

        async def patched_self_check_tool_input(*args, **kwargs):
            context = kwargs.get("context", {})
            context["config"] = CustomConfig()
            kwargs["context"] = context
            return await self_check_tool_input(*args, **kwargs)

        chat.app.runtime.register_action(patched_self_check_tool_input, name="test_self_check_tool_input")

        messages = [
            {"role": "user", "content": "Execute dangerous operation"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "dangerous_tool",
                        "args": {},
                        "id": "call_danger_001",
                        "type": "tool_call",
                    }
                ],
            },
            {
                "role": "tool",
                "content": "Operation completed",
                "name": "dangerous_tool",
                "tool_call_id": "call_danger_001",
            },
        ]

        result = await chat.app.generate_async(messages=messages)

        assert "not allowed" in result["content"]

    @pytest.mark.asyncio
    async def test_oversized_tool_message_blocked(self):
        """Test that oversized tool messages are blocked by validation."""

        class CustomConfig:
            def __init__(self):
                self.max_tool_message_length = 50

        config = RailsConfig.from_content(
            """
            define flow self check tool input
              $allowed = execute test_self_check_tool_input(tool_message=$tool_message, tool_name=$tool_name, tool_call_id=$tool_call_id)
              if not $allowed
                bot refuse tool input
                abort

            define bot refuse tool input
              "Tool response too long."
            """,
            """
            models:
              - type: main
                engine: mock
                model: test-model
            rails:
              tool_input:
                flows:
                  - self check tool input
            """,
        )

        chat = TestChat(config, llm_completions=[""])

        async def patched_self_check_tool_input(*args, **kwargs):
            context = kwargs.get("context", {})
            context["config"] = CustomConfig()
            kwargs["context"] = context
            return await self_check_tool_input(*args, **kwargs)

        chat.app.runtime.register_action(patched_self_check_tool_input, name="test_self_check_tool_input")

        large_message = "This is a very long tool response that exceeds the maximum allowed length and should be blocked by the validation"

        messages = [
            {"role": "user", "content": "Get large data"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_large_data",
                        "args": {},
                        "id": "call_large_001",
                        "type": "tool_call",
                    }
                ],
            },
            {
                "role": "tool",
                "content": large_message,
                "name": "get_large_data",
                "tool_call_id": "call_large_001",
            },
        ]

        result = await chat.app.generate_async(messages=messages)

        assert "too long" in result["content"]


class TestBotToolCallsEventChanges:
    """Test the changes from BotToolCall to BotToolCalls event."""

    @pytest.mark.asyncio
    async def test_bot_tool_calls_event_generated(self):
        """Test that BotToolCalls events are generated (not BotToolCall)."""
        test_tool_calls = [
            {
                "name": "test_function",
                "args": {"param1": "value1"},
                "id": "call_12345",
                "type": "tool_call",
            }
        ]

        with patch("nemoguardrails.actions.llm.utils.get_and_clear_tool_calls_contextvar") as mock_get_clear:
            mock_get_clear.return_value = test_tool_calls

            config = RailsConfig.from_content(config={"models": [], "passthrough": True})
            chat = TestChat(config, llm_completions=[""])

            result = await chat.app.generate_async(messages=[{"role": "user", "content": "Test"}])

            assert result["tool_calls"] is not None
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["name"] == "test_function"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_bot_tool_calls_event(self):
        """Test that multiple tool calls are handled in BotToolCalls event."""
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

        with patch("nemoguardrails.actions.llm.utils.get_and_clear_tool_calls_contextvar") as mock_get_clear:
            mock_get_clear.return_value = test_tool_calls

            config = RailsConfig.from_content(config={"models": [], "passthrough": True})
            chat = TestChat(config, llm_completions=[""])

            result = await chat.app.generate_async(messages=[{"role": "user", "content": "Execute multiple tools"}])

            assert result["tool_calls"] is not None
            assert len(result["tool_calls"]) == 2
            assert result["tool_calls"][0]["name"] == "tool_one"
            assert result["tool_calls"][1]["name"] == "tool_two"


class TestUserToolMessagesEventProcessing:
    """Test the new UserToolMessages event processing."""

    @pytest.mark.asyncio
    async def test_user_tool_messages_validation_failure(self):
        """Test that UserToolMessages processing can fail validation."""
        config = RailsConfig.from_content(
            """
            define flow self check tool input
              $allowed = execute test_self_check_tool_input(tool_message=$tool_message, tool_name=$tool_name, tool_call_id=$tool_call_id)
              if not $allowed
                bot refuse tool input
                abort

            define bot refuse tool input
              "Tool input validation failed."
            """,
            """
            models:
              - type: main
                engine: mock
                model: test-model
            rails:
              tool_input:
                flows:
                  - self check tool input
            """,
        )

        chat = TestChat(config, llm_completions=[""])

        chat.app.runtime.register_action(self_check_tool_input, name="test_self_check_tool_input")

        messages = [
            {"role": "user", "content": "Get weather and stock data"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                        "id": "call_weather_001",
                        "type": "tool_call",
                    },
                    {
                        "name": "get_stock_price",
                        "args": {"symbol": "AAPL"},
                        "id": "call_stock_001",
                        "type": "tool_call",
                    },
                ],
            },
            {
                "role": "tool",
                "content": "Sunny, 22°C",
                "name": "get_weather",
                "tool_call_id": "call_weather_001",
            },
            {
                "role": "tool",
                "content": "$150.25",
                "name": "get_stock_price",
            },
        ]

        result = await chat.app.generate_async(messages=messages)

        from nemoguardrails.utils import new_event_dict

        invalid_events = [
            new_event_dict(
                "UserToolMessages",
                tool_messages=[
                    {
                        "content": "$150.25",
                        "name": "get_stock_price",
                    }
                ],
            )
        ]

        invalid_result_events = await chat.app.runtime.generate_events(invalid_events)

        blocked_found = any(
            event.get("type") == "BotMessage" and "validation failed" in event.get("text", "")
            for event in invalid_result_events
        )
        assert blocked_found, f"Expected tool input to be blocked, got events: {invalid_result_events}"


class TestInputToolRailsIntegration:
    """Integration tests for input tool rails with the broader system."""

    @pytest.mark.asyncio
    async def test_input_tool_rails_disabled_generation_options(self):
        """Test input tool rails can be disabled via generation options."""
        config = RailsConfig.from_content(
            """
            define flow self check tool input
              $allowed = execute test_self_check_tool_input(tool_message=$tool_message, tool_name=$tool_name, tool_call_id=$tool_call_id)
              if not $allowed
                bot refuse tool input
                abort

            define bot refuse tool input
              "Input validation blocked this."
            """,
            """
            models: []
            passthrough: true
            rails:
              tool_input:
                flows:
                  - self check tool input
            """,
        )

        chat = TestChat(
            config,
            llm_completions=["Weather processed without validation."],
        )

        chat.app.runtime.register_action(self_check_tool_input, name="test_self_check_tool_input")

        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                        "id": "call_weather_001",
                        "type": "tool_call",
                    }
                ],
            },
            {
                "role": "tool",
                "content": "",
                "name": "get_weather",
                "tool_call_id": "call_weather_001",
            },
        ]

        result = await chat.app.generate_async(messages=messages, options={"rails": {"tool_input": False}})

        content = result.response[0]["content"] if result.response else ""
        assert "Input validation blocked" not in content, (
            f"Tool input rails should be disabled but got blocking: {content}"
        )

        assert "Weather processed without validation" in content, (
            f"Expected LLM completion when tool input rails disabled: {content}"
        )

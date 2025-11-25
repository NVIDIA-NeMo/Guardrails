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

"""Test that tool calling ONLY works in passthrough mode."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions.llm.generation import LLMGenerationActions
from nemoguardrails.context import tool_calls_var


@pytest.fixture
def mock_llm_with_tool_calls():
    """Mock LLM that returns tool calls."""
    llm = AsyncMock()

    mock_response = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_123",
                "type": "tool_call",
                "name": "test_tool",
                "args": {"param": "value"},
            }
        ],
    )
    llm.ainvoke.return_value = mock_response
    llm.invoke.return_value = mock_response
    return llm


@pytest.fixture
def config_passthrough():
    """Config with passthrough enabled."""
    return RailsConfig.from_content(
        colang_content="",
        yaml_content="""
        models:
          - type: main
            engine: mock
            model: test-model

        rails:
          input:
            flows: []
          dialog:
            flows: []
          output:
            flows: []

        passthrough: true
        """,
    )


@pytest.fixture
def config_no_passthrough():
    """Config with passthrough disabled."""
    return RailsConfig.from_content(
        colang_content="",
        yaml_content="""
        models:
          - type: main
            engine: mock
            model: test-model

        rails:
          input:
            flows: []
          dialog:
            flows: []
          output:
            flows: []

        passthrough: false
        """,
    )


class TestToolCallingPassthroughOnly:
    """Test that tool calling only works in passthrough mode."""

    def test_config_passthrough_true(self, config_passthrough):
        """Test that passthrough config is correctly set."""
        assert config_passthrough.passthrough is True

    def test_config_passthrough_false(self, config_no_passthrough):
        """Test that non-passthrough config is correctly set."""
        assert config_no_passthrough.passthrough is False

    @pytest.mark.asyncio
    async def test_tool_calls_work_in_passthrough_mode(self, config_passthrough, mock_llm_with_tool_calls):
        """Test that tool calls create BotToolCalls events in passthrough mode."""
        # Set up context with tool calls
        tool_calls = [
            {
                "id": "call_123",
                "type": "tool_call",
                "name": "test_tool",
                "args": {"param": "value"},
            }
        ]
        tool_calls_var.set(tool_calls)

        generation_actions = LLMGenerationActions(
            config=config_passthrough,
            llm=mock_llm_with_tool_calls,
            llm_task_manager=MagicMock(),
            get_embedding_search_provider_instance=MagicMock(return_value=None),
        )

        events = [{"type": "UserMessage", "text": "test"}]
        context = {}

        result = await generation_actions.generate_user_intent(
            events=events, context=context, config=config_passthrough
        )

        assert len(result.events) == 1
        assert result.events[0]["type"] == "BotToolCalls"
        assert result.events[0]["tool_calls"] == tool_calls

    @pytest.mark.asyncio
    async def test_tool_calls_ignored_in_non_passthrough_mode(self, config_no_passthrough, mock_llm_with_tool_calls):
        """Test that tool calls are ignored when not in passthrough mode."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "tool_call",
                "name": "test_tool",
                "args": {"param": "value"},
            }
        ]
        tool_calls_var.set(tool_calls)

        generation_actions = LLMGenerationActions(
            config=config_no_passthrough,
            llm=mock_llm_with_tool_calls,
            llm_task_manager=MagicMock(),
            get_embedding_search_provider_instance=MagicMock(return_value=None),
        )

        events = [{"type": "UserMessage", "text": "test"}]
        context = {}

        result = await generation_actions.generate_user_intent(
            events=events, context=context, config=config_no_passthrough
        )

        assert len(result.events) == 1
        assert result.events[0]["type"] == "BotMessage"
        assert "tool_calls" not in result.events[0]

    @pytest.mark.asyncio
    async def test_no_tool_calls_creates_bot_message_in_passthrough(self, config_passthrough, mock_llm_with_tool_calls):
        """Test that no tool calls creates BotMessage event even in passthrough mode."""
        tool_calls_var.set(None)

        mock_response_no_tools = AIMessage(content="Regular text response")
        mock_llm_with_tool_calls.ainvoke.return_value = mock_response_no_tools
        mock_llm_with_tool_calls.invoke.return_value = mock_response_no_tools

        generation_actions = LLMGenerationActions(
            config=config_passthrough,
            llm=mock_llm_with_tool_calls,
            llm_task_manager=MagicMock(),
            get_embedding_search_provider_instance=MagicMock(return_value=None),
        )

        events = [{"type": "UserMessage", "text": "test"}]
        context = {}

        result = await generation_actions.generate_user_intent(
            events=events, context=context, config=config_passthrough
        )

        assert len(result.events) == 1
        assert result.events[0]["type"] == "BotMessage"

    def test_llm_rails_integration_passthrough_mode(self, config_passthrough, mock_llm_with_tool_calls):
        """Test LLMRails with passthrough mode allows tool calls."""
        rails = LLMRails(config=config_passthrough, llm=mock_llm_with_tool_calls)

        assert rails.config.passthrough is True

    def test_llm_rails_integration_non_passthrough_mode(self, config_no_passthrough, mock_llm_with_tool_calls):
        """Test LLMRails without passthrough mode."""
        rails = LLMRails(config=config_no_passthrough, llm=mock_llm_with_tool_calls)

        assert rails.config.passthrough is False

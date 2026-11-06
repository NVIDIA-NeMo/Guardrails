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

"""Test tool-calling behavior across passthrough and non-dialog-flow general modes.

Tool calls surface as ``BotToolCalls`` in any config that reaches
``_emit_general_bot_turn`` -- i.e. any config with no dialog flows (no
``user_messages`` and no ``single_call`` dialog rails) -- whether or not
``passthrough`` is set. Only dialog-flow configs (covered separately in
``tests/test_generation_equivalence.py``) drop tool calls.
"""

from unittest.mock import MagicMock

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions.llm.generation import LLMGenerationActions
from nemoguardrails.context import tool_calls_var
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallFunction
from tests.utils import FakeLLMModel


@pytest.fixture
def mock_llm_with_tool_calls():
    return FakeLLMModel(
        llm_responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_123",
                        type="function",
                        function=ToolCallFunction(
                            name="test_tool",
                            arguments={"param": "value"},
                        ),
                    )
                ],
            )
        ]
    )


@pytest.fixture
def config_passthrough():
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
    """Tool calls surface in any config without dialog flows, passthrough or not."""

    def test_config_passthrough_true(self, config_passthrough):
        assert config_passthrough.passthrough is True

    def test_config_passthrough_false(self, config_no_passthrough):
        assert config_no_passthrough.passthrough is False

    @pytest.mark.asyncio
    async def test_tool_calls_work_in_passthrough_mode(self, config_passthrough, mock_llm_with_tool_calls):
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "arguments": {"param": "value"},
                },
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
        stored = result.events[0]["tool_calls"]
        assert len(stored) == 1
        assert stored[0]["function"]["name"] == "test_tool"
        assert stored[0]["function"]["arguments"] == {"param": "value"}
        assert stored[0]["id"] == "call_123"

    @pytest.mark.asyncio
    async def test_tool_calls_work_in_non_passthrough_mode_without_dialog_flows(
        self, config_no_passthrough, mock_llm_with_tool_calls
    ):
        """Tool calls still surface as ``BotToolCalls`` without passthrough.

        ``config_no_passthrough`` has no dialog flows either (no
        ``user_messages``, ``single_call`` disabled), so ``generate_user_intent``
        still reaches ``_emit_general_bot_turn``, which reads ``tool_calls_var``
        regardless of ``passthrough``.
        """
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "arguments": {"param": "value"},
                },
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
        assert result.events[0]["type"] == "BotToolCalls"
        stored = result.events[0]["tool_calls"]
        assert len(stored) == 1
        assert stored[0]["function"]["name"] == "test_tool"
        assert stored[0]["function"]["arguments"] == {"param": "value"}
        assert stored[0]["id"] == "call_123"

    @pytest.mark.asyncio
    async def test_no_tool_calls_creates_bot_message_in_passthrough(self, config_passthrough):
        tool_calls_var.set(None)

        fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="Regular text response")])

        generation_actions = LLMGenerationActions(
            config=config_passthrough,
            llm=fake_llm,
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
        rails = LLMRails(config=config_passthrough, llm=mock_llm_with_tool_calls)

        assert rails.config.passthrough is True

    def test_llm_rails_integration_non_passthrough_mode(self, config_no_passthrough, mock_llm_with_tool_calls):
        rails = LLMRails(config=config_no_passthrough, llm=mock_llm_with_tool_calls)

        assert rails.config.passthrough is False

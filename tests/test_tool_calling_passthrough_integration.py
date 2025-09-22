# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemoguardrails import RailsConfig
from nemoguardrails.context import tool_calls_var
from nemoguardrails.rails.llm.llmrails import GenerationOptions, GenerationResponse
from tests.utils import TestChat


class TestToolCallingPassthroughIntegration:
    def setup_method(self):
        self.passthrough_config = RailsConfig.from_content(
            colang_content="",
            yaml_content="""
            models: []
            passthrough: true
            """,
        )

    @pytest.mark.asyncio
    async def test_tool_calls_work_in_passthrough_mode_with_options(self):
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

        with patch(
            "nemoguardrails.rails.llm.llmrails.get_and_clear_tool_calls_contextvar"
        ) as mock_get_clear:
            mock_get_clear.return_value = test_tool_calls

            chat = TestChat(
                self.passthrough_config,
                llm_completions=["I'll help you with the weather and calculation."],
            )

            result = await chat.app.generate_async(
                messages=[
                    {
                        "role": "user",
                        "content": "What's the weather in NYC and what's 2+2?",
                    }
                ],
                options=GenerationOptions(),
            )

            assert isinstance(result, GenerationResponse)
            assert result.tool_calls == test_tool_calls
            assert len(result.tool_calls) == 2
            assert isinstance(result.response, list)
            assert result.response[0]["role"] == "assistant"
            assert "help you" in result.response[0]["content"]

    @pytest.mark.asyncio
    async def test_tool_calls_work_in_passthrough_mode_dict_response(self):
        test_tool_calls = [
            {
                "name": "get_weather",
                "args": {"location": "London"},
                "id": "call_weather",
                "type": "tool_call",
            }
        ]

        with patch(
            "nemoguardrails.rails.llm.llmrails.get_and_clear_tool_calls_contextvar"
        ) as mock_get_clear:
            mock_get_clear.return_value = test_tool_calls

            chat = TestChat(
                self.passthrough_config,
                llm_completions=["I'll check the weather for you."],
            )

            result = await chat.app.generate_async(
                messages=[{"role": "user", "content": "What's the weather like?"}]
            )

            assert isinstance(result, dict)
            assert "tool_calls" in result
            assert result["tool_calls"] == test_tool_calls
            assert result["role"] == "assistant"
            assert "check the weather" in result["content"]

    @pytest.mark.asyncio
    async def test_no_tool_calls_in_passthrough_mode(self):
        with patch(
            "nemoguardrails.rails.llm.llmrails.get_and_clear_tool_calls_contextvar"
        ) as mock_get_clear:
            mock_get_clear.return_value = None

            chat = TestChat(
                self.passthrough_config,
                llm_completions=["Hello! How can I help you today?"],
            )

            result = await chat.app.generate_async(
                messages=[{"role": "user", "content": "Hello"}],
                options=GenerationOptions(),
            )

            assert isinstance(result, GenerationResponse)
            assert result.tool_calls is None
            assert "Hello! How can I help" in result.response[0]["content"]

    @pytest.mark.asyncio
    async def test_empty_tool_calls_in_passthrough_mode(self):
        with patch(
            "nemoguardrails.rails.llm.llmrails.get_and_clear_tool_calls_contextvar"
        ) as mock_get_clear:
            mock_get_clear.return_value = []

            chat = TestChat(
                self.passthrough_config, llm_completions=["I understand your request."]
            )

            result = await chat.app.generate_async(
                messages=[{"role": "user", "content": "Tell me a joke"}]
            )

            assert isinstance(result, dict)
            assert "tool_calls" not in result
            assert "understand your request" in result["content"]

    @pytest.mark.asyncio
    async def test_tool_calls_with_prompt_mode_passthrough(self):
        test_tool_calls = [
            {
                "name": "search",
                "args": {"query": "latest news"},
                "id": "call_prompt",
                "type": "tool_call",
            }
        ]

        with patch(
            "nemoguardrails.rails.llm.llmrails.get_and_clear_tool_calls_contextvar"
        ) as mock_get_clear:
            mock_get_clear.return_value = test_tool_calls

            chat = TestChat(
                self.passthrough_config,
                llm_completions=["I'll search for that information."],
            )

            result = await chat.app.generate_async(
                prompt="Search for the latest news", options=GenerationOptions()
            )

            assert isinstance(result, GenerationResponse)
            assert result.tool_calls == test_tool_calls
            assert isinstance(result.response, str)
            assert "search for that information" in result.response

    @pytest.mark.asyncio
    async def test_complex_tool_calls_passthrough_integration(self):
        complex_tool_calls = [
            {
                "name": "get_current_weather",
                "args": {"location": "San Francisco", "unit": "fahrenheit"},
                "id": "call_weather_001",
                "type": "tool_call",
            },
            {
                "name": "calculate_tip",
                "args": {"bill_amount": 85.50, "tip_percentage": 18},
                "id": "call_calc_002",
                "type": "tool_call",
            },
            {
                "name": "web_search",
                "args": {"query": "best restaurants near me", "limit": 5},
                "id": "call_search_003",
                "type": "tool_call",
            },
        ]

        with patch(
            "nemoguardrails.rails.llm.llmrails.get_and_clear_tool_calls_contextvar"
        ) as mock_get_clear:
            mock_get_clear.return_value = complex_tool_calls

            chat = TestChat(
                self.passthrough_config,
                llm_completions=[
                    "I'll help you with the weather, calculate the tip, and find restaurants."
                ],
            )

            result = await chat.app.generate_async(
                messages=[
                    {
                        "role": "user",
                        "content": "I need weather, tip calculation, and restaurant search",
                    }
                ],
                options=GenerationOptions(),
            )

            assert isinstance(result, GenerationResponse)
            assert result.tool_calls == complex_tool_calls
            assert len(result.tool_calls) == 3

            weather_call = result.tool_calls[0]
            assert weather_call["name"] == "get_current_weather"
            assert weather_call["args"]["location"] == "San Francisco"
            assert weather_call["args"]["unit"] == "fahrenheit"
            assert weather_call["id"] == "call_weather_001"
            assert weather_call["type"] == "tool_call"

            tip_call = result.tool_calls[1]
            assert tip_call["name"] == "calculate_tip"
            assert tip_call["args"]["bill_amount"] == 85.50
            assert tip_call["args"]["tip_percentage"] == 18
            assert tip_call["id"] == "call_calc_002"

            search_call = result.tool_calls[2]
            assert search_call["name"] == "web_search"
            assert search_call["args"]["query"] == "best restaurants near me"
            assert search_call["args"]["limit"] == 5
            assert search_call["id"] == "call_search_003"

    def test_get_and_clear_tool_calls_called_correctly(self):
        test_tool_calls = [
            {
                "name": "test_func",
                "args": {"param": "value"},
                "id": "call_test",
                "type": "tool_call",
            }
        ]

        tool_calls_var.set(test_tool_calls)

        from nemoguardrails.actions.llm.utils import get_and_clear_tool_calls_contextvar

        result = get_and_clear_tool_calls_contextvar()

        assert result == test_tool_calls
        assert tool_calls_var.get() is None

    @pytest.mark.asyncio
    async def test_tool_calls_integration_preserves_other_response_data(self):
        test_tool_calls = [
            {
                "name": "preserve_test",
                "args": {"data": "preserved"},
                "id": "call_preserve",
                "type": "tool_call",
            }
        ]

        with patch(
            "nemoguardrails.rails.llm.llmrails.get_and_clear_tool_calls_contextvar"
        ) as mock_get_clear:
            mock_get_clear.return_value = test_tool_calls

            chat = TestChat(
                self.passthrough_config,
                llm_completions=["Response with preserved data."],
            )

            result = await chat.app.generate_async(
                messages=[{"role": "user", "content": "Test message"}],
                options=GenerationOptions(),
            )

            assert isinstance(result, GenerationResponse)
            assert result.tool_calls == test_tool_calls
            assert result.response is not None
            assert result.llm_output is None
            assert result.state is None
            assert isinstance(result.response, list)
            assert len(result.response) == 1
            assert result.response[0]["role"] == "assistant"
            assert result.response[0]["content"] == "Response with preserved data."

    @pytest.mark.asyncio
    async def test_tool_calls_with_real_world_examples(self):
        realistic_tool_calls = [
            {
                "name": "get_weather",
                "args": {"location": "London"},
                "id": "call_JMTxzsfy21izMf248MHZvj3G",
                "type": "tool_call",
            },
            {
                "name": "add",
                "args": {"a": 15, "b": 27},
                "id": "call_INoaqHesFOrZdjHynU78qjX4",
                "type": "tool_call",
            },
        ]

        with patch(
            "nemoguardrails.rails.llm.llmrails.get_and_clear_tool_calls_contextvar"
        ) as mock_get_clear:
            mock_get_clear.return_value = realistic_tool_calls

            chat = TestChat(
                self.passthrough_config,
                llm_completions=[
                    "I'll get the weather in London and add 15 + 27 for you."
                ],
            )

            result = await chat.app.generate_async(
                messages=[
                    {
                        "role": "user",
                        "content": "What's the weather in London and what's 15 + 27?",
                    }
                ],
                options=GenerationOptions(),
            )

            assert isinstance(result, GenerationResponse)
            assert result.tool_calls == realistic_tool_calls

            weather_call = result.tool_calls[0]
            assert weather_call["name"] == "get_weather"
            assert weather_call["args"] == {"location": "London"}
            assert weather_call["id"] == "call_JMTxzsfy21izMf248MHZvj3G"
            assert weather_call["type"] == "tool_call"

            add_call = result.tool_calls[1]
            assert add_call["name"] == "add"
            assert add_call["args"] == {"a": 15, "b": 27}
            assert add_call["id"] == "call_INoaqHesFOrZdjHynU78qjX4"
            assert add_call["type"] == "tool_call"

    @pytest.mark.asyncio
    async def test_passthrough_config_requirement(self):
        assert self.passthrough_config.passthrough is True

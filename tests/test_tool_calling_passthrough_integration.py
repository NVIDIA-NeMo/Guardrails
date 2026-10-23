# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy
import json
from unittest.mock import AsyncMock, patch

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.context import tool_calls_var
from nemoguardrails.exceptions import ToolConversationTooLongError
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.llm.types import Task
from nemoguardrails.rails.llm.llmrails import GenerationOptions, GenerationResponse
from nemoguardrails.rails.llm.utils import get_history_cache_key
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallFunction
from tests.utils import FakeLLMModel, TestChat


def _tool_prompt_config(chat_prompt, max_length):
    """Build a non-passthrough string or chat prompt with the requested character limit."""
    template = (
        {"messages": [{"type": "system", "content": "Rules"}, "{{ history | to_chat_messages }}"]}
        if chat_prompt
        else {"content": "Rules\n{{ history | user_assistant_sequence }}"}
    )
    return RailsConfig.from_content(
        yaml_content=json.dumps(
            {
                "models": [],
                "passthrough": False,
                "prompts": [{"task": "general", "models": ["unknown"], "max_length": max_length, **template}],
            }
        )
    )


@pytest.fixture
def tool_history():
    """Provide a two-city tool exchange with parallel calls and matching results."""
    return [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": f"call_{city}", "type": "function", "function": {"name": "weather", "arguments": {"city": city}}}
                for city in ("Paris", "Berlin")
            ],
        },
        {"role": "tool", "tool_call_id": "call_Paris", "content": "Paris: sunny"},
        {"role": "tool", "tool_call_id": "call_Berlin", "content": "Berlin: rainy"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_prompt", [False, True])
@pytest.mark.parametrize("oversized_field", ["result", "arguments", "history", "call_id"])
async def test_oversized_tool_history_rejected_before_model_call(chat_prompt, oversized_field, tool_history, caplog):
    """Reject oversized history or tool metadata without mutation, private-data logging, or model invocation."""
    sensitive_text = "private-tool-data-" * 150
    if oversized_field == "result":
        tool_history[-1]["content"] = sensitive_text
    elif oversized_field == "arguments":
        tool_history[1]["tool_calls"][0]["function"]["arguments"]["city"] = sensitive_text
    elif oversized_field == "call_id":
        tool_history[1]["tool_calls"][0]["id"] = sensitive_text
        tool_history[2]["tool_call_id"] = sensitive_text
    else:
        tool_history[0]["content"] = sensitive_text
    original = copy.deepcopy(tool_history)
    model = FakeLLMModel(responses=["Must not be called"])
    rails = LLMRails(_tool_prompt_config(chat_prompt, 1000), llm=model)

    with pytest.raises(ToolConversationTooLongError, match="Tool conversation.*max_length") as error:
        await rails.generate_async(messages=tool_history)

    assert model.inference_count == 0
    assert error.value.max_length == 1000
    assert error.value.actual_length > 1000
    assert tool_history == original
    assert sensitive_text not in str(error.value)
    assert "private-tool-data" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_prompt", [False, True])
@pytest.mark.parametrize("max_length", [1000, None])
async def test_tool_history_within_budget_preserves_all_calls_and_results(chat_prompt, max_length, tool_history):
    """Preserve complete native exchanges when the prompt fits or its size limit is disabled."""
    model = FakeLLMModel(responses=["Paris is sunny; Berlin is rainy."])
    model.generate_async = AsyncMock(wraps=model.generate_async)
    rails = LLMRails(_tool_prompt_config(chat_prompt, max_length), llm=model)
    if max_length is None:
        tool_history[-1]["content"] *= 2000

    result = await rails.generate_async(messages=tool_history)

    assert result["content"] == "Paris is sunny; Berlin is rainy."
    sent = [message.to_dict() for message in model.generate_async.await_args.args[0]]
    assert sent[-3:] == tool_history[1:]
    assert model.inference_count == 1


@pytest.mark.parametrize("chat_prompt", [False, True])
def test_tool_prompt_budget_boundary_counts_serialized_messages(chat_prompt, tool_history):
    """Accept the exact serialized character limit and reject one character less for both prompt forms."""
    tool_history[-1]["content"] = "Berlin: 雨 ☔"
    expected = (
        [{"type": "system", "content": "Rules"}, tool_history[0]]
        if chat_prompt
        else [{"role": "user", "content": "Rules\nUser: Weather?"}]
    ) + [{**tool_history[1], "content": None}, *tool_history[2:]]
    limit = len(json.dumps(expected, ensure_ascii=False))
    events = [
        {"type": "UserMessage", "text": "Weather?"},
        {"type": "BotToolCalls", "tool_calls": tool_history[1]["tool_calls"]},
        *[{"type": "ToolMessage", "message": message} for message in tool_history[2:]],
    ]
    manager = LLMTaskManager(_tool_prompt_config(chat_prompt, limit))
    assert manager.render_task_prompt(Task.GENERAL, events=events) == expected
    manager = LLMTaskManager(_tool_prompt_config(chat_prompt, limit - 1))
    with pytest.raises(ToolConversationTooLongError, match="Tool conversation.*max_length"):
        manager.render_task_prompt("general", events=events)


def test_tool_history_limit_applies_to_sync_generation(tool_history):
    """Propagate oversized-tool-history validation through the synchronous API before invoking the model."""
    tool_history[-1]["content"] *= 2000
    model = FakeLLMModel(responses=["Must not be called"])
    rails = LLMRails(_tool_prompt_config(False, 1000), llm=model)
    with pytest.raises(ToolConversationTooLongError, match="Tool conversation.*max_length"):
        rails.generate(messages=tool_history)
    assert model.inference_count == 0


@pytest.mark.asyncio
async def test_passthrough_tool_history_does_not_use_general_prompt_limit(tool_history):
    """Keep passthrough generation independent of the unused general prompt size limit."""
    config = _tool_prompt_config(False, 1)
    config.passthrough = True
    tool_history[-1]["content"] *= 2000
    model = FakeLLMModel(responses=["Done"])
    model.generate_async = AsyncMock(wraps=model.generate_async)
    rails = LLMRails(config, llm=model)

    await rails.generate_async(messages=tool_history)

    assert [message.to_dict() for message in model.generate_async.await_args.args[0]] == tool_history


@pytest.mark.parametrize("chat_prompt", [False, True])
def test_text_only_general_prompt_still_trims_history(chat_prompt):
    """Preserve existing oldest-event trimming for general prompts without tool calls."""
    manager = LLMTaskManager(_tool_prompt_config(chat_prompt, 50))
    prompt = manager.render_task_prompt(
        Task.GENERAL,
        events=[{"type": "UserMessage", "text": "old history" * 100}, {"type": "UserMessage", "text": "Hi"}],
    )
    assert "old history" not in str(prompt)
    assert "Hi" in str(prompt)


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("chat_prompt", [False, True])
@pytest.mark.parametrize("passthrough", [False, True])
async def test_tool_round_trip_preserves_history(cached, chat_prompt, passthrough):
    """Preserve tool calls, results, and later turns across caching, prompt formats, and passthrough modes."""
    config = RailsConfig.from_content(
        yaml_content=f"models: []\npassthrough: {str(passthrough).lower()}\n"
        + (
            """
prompts:
  - task: general
    models: [unknown]
    messages:
      - type: system
        content: "{{ general_instructions }}"
      - "{{ history | to_chat_messages }}"
"""
            if chat_prompt
            else ""
        )
    )
    tool_calls = [
        ToolCall(id=f"call_{city}", function=ToolCallFunction(name="get_weather", arguments={"city": city}))
        for city in ("Paris", "Berlin")
    ]
    model = FakeLLMModel(
        llm_responses=[
            LLMResponse(content="", tool_calls=tool_calls),
            LLMResponse(content="Paris is sunny; Berlin is rainy."),
            LLMResponse(content="Take an umbrella in Berlin."),
        ]
    )
    model.generate_async = AsyncMock(wraps=model.generate_async)
    rails = LLMRails(config, llm=model)
    messages = [{"role": "user", "content": "Compare the weather in Paris and Berlin."}]
    first = await rails.generate_async(messages=messages)
    assert first["tool_calls"] == [call.to_dict() for call in tool_calls]
    if chat_prompt:
        first.pop("content")
    results = [
        {"role": "tool", "tool_call_id": "call_Paris", "name": "get_weather", "content": "Paris: sunny, 22 C"},
        {"role": "tool", "tool_call_id": "call_Berlin", "name": "get_weather", "content": "Berlin: rainy, 15 C"},
    ]
    messages.extend([first, *results])
    if not cached:
        rails.events_history_cache.clear()

    second = await rails.generate_async(messages=messages)
    assert second == {"role": "assistant", "content": "Paris is sunny; Berlin is rainy."}
    messages.extend([second, {"role": "user", "content": "Should I take an umbrella?"}])
    if not cached:
        rails.events_history_cache.clear()
    await rails.generate_async(messages=messages)

    for call in model.generate_async.await_args_list[1:]:
        prompt = call.args[0]
        history = [message.to_dict() for message in prompt if message.role != "system"]
        if not passthrough and not chat_prompt:
            assert history[0]["content"].count(messages[0]["content"]) == 1
            assert "helpful AI assistant" in history[0]["content"]
            assert "Tool:" not in history[0]["content"]
        else:
            assert history[0] == messages[0]
        assistant = first if passthrough else {"role": "assistant", "tool_calls": first["tool_calls"]}
        assert history[1:4] == [assistant, *results]
    assert model.inference_count == 3


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

        with patch("nemoguardrails.actions.llm.generation.get_and_clear_tool_calls_contextvar") as mock_get_clear:
            mock_get_clear.return_value = test_tool_calls

            chat = TestChat(
                self.passthrough_config,
                llm_completions=[""],
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
            assert result.response[0]["content"] == ""

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

        with patch("nemoguardrails.actions.llm.generation.get_and_clear_tool_calls_contextvar") as mock_get_clear:
            mock_get_clear.return_value = test_tool_calls

            chat = TestChat(
                self.passthrough_config,
                llm_completions=["I'll check the weather for you."],
            )

            result = await chat.app.generate_async(messages=[{"role": "user", "content": "What's the weather like?"}])

            assert isinstance(result, dict)
            assert "tool_calls" in result
            assert result["tool_calls"] == test_tool_calls
            assert result["role"] == "assistant"
            assert result["content"] == ""

    @pytest.mark.asyncio
    async def test_no_tool_calls_in_passthrough_mode(self):
        with patch("nemoguardrails.actions.llm.generation.get_and_clear_tool_calls_contextvar") as mock_get_clear:
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
        with patch("nemoguardrails.actions.llm.generation.get_and_clear_tool_calls_contextvar") as mock_get_clear:
            mock_get_clear.return_value = []

            chat = TestChat(self.passthrough_config, llm_completions=["I understand your request."])

            result = await chat.app.generate_async(messages=[{"role": "user", "content": "Tell me a joke"}])

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

        with patch("nemoguardrails.actions.llm.generation.get_and_clear_tool_calls_contextvar") as mock_get_clear:
            mock_get_clear.return_value = test_tool_calls

            chat = TestChat(
                self.passthrough_config,
                # note that llm would not generate any content when tool calls are present
                # this is here just to show the underlying behavior
                llm_completions=["I'll search for that information."],
            )

            result = await chat.app.generate_async(prompt="Search for the latest news", options=GenerationOptions())

            assert isinstance(result, GenerationResponse)
            assert result.tool_calls == test_tool_calls
            assert isinstance(result.response, str)
            assert result.response == ""

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

        with patch("nemoguardrails.actions.llm.generation.get_and_clear_tool_calls_contextvar") as mock_get_clear:
            mock_get_clear.return_value = complex_tool_calls

            chat = TestChat(
                self.passthrough_config,
                llm_completions=["I'll help you with the weather, calculate the tip, and find restaurants."],
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

        with patch("nemoguardrails.actions.llm.generation.get_and_clear_tool_calls_contextvar") as mock_get_clear:
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
            assert result.response[0]["content"] == ""

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

        with patch("nemoguardrails.actions.llm.generation.get_and_clear_tool_calls_contextvar") as mock_get_clear:
            mock_get_clear.return_value = realistic_tool_calls

            chat = TestChat(
                self.passthrough_config,
                llm_completions=["I'll get the weather in London and add 15 + 27 for you."],
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


class TestToolCallingNonPassthroughIntegration:
    """Exercise tool-call response shapes and metadata when passthrough is disabled."""

    def setup_method(self):
        """Create a fresh configuration without dialog flows and with passthrough disabled."""
        self.non_passthrough_config = RailsConfig.from_content(
            colang_content="",
            yaml_content="""
            models: []
            passthrough: false
            """,
        )

    @pytest.mark.asyncio
    async def test_tool_calls_work_in_non_passthrough_mode_with_options(self):
        """Return tool calls on GenerationResponse when generation options are supplied."""
        test_tool_calls = [
            {
                "name": "get_weather",
                "args": {"location": "NYC"},
                "id": "call_123",
                "type": "tool_call",
            },
        ]

        with patch("nemoguardrails.actions.llm.generation.get_and_clear_tool_calls_contextvar") as mock_get_clear:
            mock_get_clear.return_value = test_tool_calls

            chat = TestChat(
                self.non_passthrough_config,
                llm_completions=[""],
            )

            result = await chat.app.generate_async(
                messages=[
                    {
                        "role": "user",
                        "content": "What's the weather in NYC?",
                    }
                ],
                options=GenerationOptions(),
            )

            assert isinstance(result, GenerationResponse)
            assert result.tool_calls == test_tool_calls
            assert len(result.tool_calls) == 1
            assert isinstance(result.response, list)
            assert result.response[0]["role"] == "assistant"
            assert result.response[0]["content"] == ""

    @pytest.mark.asyncio
    async def test_tool_calls_work_in_non_passthrough_mode_dict_response(self):
        """Return tool calls on a plain assistant dictionary when generation options are omitted."""
        test_tool_calls = [
            {
                "name": "get_weather",
                "args": {"location": "London"},
                "id": "call_weather",
                "type": "tool_call",
            }
        ]

        with patch("nemoguardrails.actions.llm.generation.get_and_clear_tool_calls_contextvar") as mock_get_clear:
            mock_get_clear.return_value = test_tool_calls

            chat = TestChat(
                self.non_passthrough_config,
                llm_completions=["I'll check the weather for you."],
            )

            result = await chat.app.generate_async(messages=[{"role": "user", "content": "What's the weather like?"}])

            assert isinstance(result, dict)
            assert "tool_calls" in result
            assert result["tool_calls"] == test_tool_calls
            assert result["role"] == "assistant"
            assert result["content"] == ""

    @pytest.mark.asyncio
    async def test_no_tool_calls_in_non_passthrough_mode(self):
        """Preserve an ordinary text response when the provider returns no tool calls."""
        with patch("nemoguardrails.actions.llm.generation.get_and_clear_tool_calls_contextvar") as mock_get_clear:
            mock_get_clear.return_value = None

            chat = TestChat(
                self.non_passthrough_config,
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
    async def test_empty_tool_calls_in_non_passthrough_mode(self):
        """Omit the tool_calls key and preserve text when the provider returns an empty call list."""
        with patch("nemoguardrails.actions.llm.generation.get_and_clear_tool_calls_contextvar") as mock_get_clear:
            mock_get_clear.return_value = []

            chat = TestChat(self.non_passthrough_config, llm_completions=["I understand your request."])

            result = await chat.app.generate_async(messages=[{"role": "user", "content": "Tell me a joke"}])

            assert isinstance(result, dict)
            assert "tool_calls" not in result
            assert "understand your request" in result["content"]

    @pytest.mark.asyncio
    async def test_tool_calls_with_prompt_mode_non_passthrough(self):
        """Expose tool metadata alongside an empty string response for prompt-based generation."""
        test_tool_calls = [
            {
                "name": "search",
                "args": {"query": "latest news"},
                "id": "call_prompt",
                "type": "tool_call",
            }
        ]

        with patch("nemoguardrails.actions.llm.generation.get_and_clear_tool_calls_contextvar") as mock_get_clear:
            mock_get_clear.return_value = test_tool_calls

            chat = TestChat(
                self.non_passthrough_config,
                llm_completions=["I'll search for that information."],
            )

            result = await chat.app.generate_async(prompt="Search for the latest news", options=GenerationOptions())

            assert isinstance(result, GenerationResponse)
            assert result.tool_calls == test_tool_calls
            assert isinstance(result.response, str)
            assert result.response == ""

    @pytest.mark.asyncio
    async def test_complex_tool_calls_non_passthrough_integration(self):
        """Preserve all names, arguments, IDs, and types in a response containing multiple tool calls."""
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

        with patch("nemoguardrails.actions.llm.generation.get_and_clear_tool_calls_contextvar") as mock_get_clear:
            mock_get_clear.return_value = complex_tool_calls

            chat = TestChat(
                self.non_passthrough_config,
                llm_completions=["I'll help you with the weather, calculate the tip, and find restaurants."],
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
            assert tip_call["type"] == "tool_call"

            search_call = result.tool_calls[2]
            assert search_call["name"] == "web_search"
            assert search_call["args"]["query"] == "best restaurants near me"
            assert search_call["args"]["limit"] == 5
            assert search_call["id"] == "call_search_003"
            assert search_call["type"] == "tool_call"

    @pytest.mark.asyncio
    async def test_tool_calls_integration_preserves_other_response_data_non_passthrough(self):
        """Keep the GenerationResponse envelope intact when it carries tool calls."""
        test_tool_calls = [
            {
                "name": "preserve_test",
                "args": {"data": "preserved"},
                "id": "call_preserve",
                "type": "tool_call",
            }
        ]

        with patch("nemoguardrails.actions.llm.generation.get_and_clear_tool_calls_contextvar") as mock_get_clear:
            mock_get_clear.return_value = test_tool_calls

            chat = TestChat(
                self.non_passthrough_config,
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
            assert result.response[0]["content"] == ""

    @pytest.mark.asyncio
    async def test_tool_calls_with_real_world_examples_non_passthrough(self):
        """Preserve weather and arithmetic tool calls together in a non-passthrough response."""
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

        with patch("nemoguardrails.actions.llm.generation.get_and_clear_tool_calls_contextvar") as mock_get_clear:
            mock_get_clear.return_value = realistic_tool_calls

            chat = TestChat(
                self.non_passthrough_config,
                llm_completions=["I'll get the weather in London and add 15 + 27 for you."],
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
    async def test_passthrough_config_requirement_non_passthrough(self):
        """Verify this test class exercises a configuration with passthrough disabled."""
        assert self.non_passthrough_config.passthrough is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("single_call", [False, True])
    async def test_tool_calls_fail_in_dialog_flows_non_passthrough(self, single_call):
        """Drop actual provider tool calls in both single-call and multi-call dialog generation."""
        config = RailsConfig.from_content(
            colang_content="""
define user ask about weather
  "What is the weather?"

define flow weather
  user ask about weather
  bot report weather
            """,
            yaml_content=json.dumps(
                {"models": [], "passthrough": False, "rails": {"dialog": {"single_call": {"enabled": single_call}}}}
            ),
        )
        assert config.user_messages == {"ask about weather": ["What is the weather?"]}
        assert config.rails.dialog.single_call.enabled is single_call
        tool_calls = [
            ToolCall(id="call_weather", function=ToolCallFunction(name="weather", arguments={"city": "Paris"}))
        ]
        completions = (
            ['  ask about weather\nbot report weather\n  "Weather unavailable."']
            if single_call
            else ["ask about weather", '  "Weather unavailable."']
        )
        model = FakeLLMModel(
            llm_responses=[LLMResponse(content=content, tool_calls=tool_calls) for content in completions]
        )
        rails = LLMRails(config, llm=model)
        result = await rails.generate_async(messages=[{"role": "user", "content": "What is the weather in Paris?"}])
        assert result == {"role": "assistant", "content": "Weather unavailable."}
        assert model.inference_count == len(completions)


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [False, True])
@pytest.mark.parametrize("wire_format", [False, True])
async def test_input_rail_rewrite_survives_cached_tool_round_trip(options, wire_format):
    """Reuse sanitized history across tool-choice changes and equivalent wire-format call arguments."""
    config = RailsConfig.from_content(
        yaml_content="""
models: []
passthrough: false
rails:
  input:
    flows: [rewrite query]
""",
        colang_content="""
define flow rewrite query
  $user_message = "REWRITTEN query"
""",
    )
    model = FakeLLMModel(
        llm_responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_weather", function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"})
                    )
                ],
            ),
            LLMResponse(content="Final answer."),
        ]
    )
    model.generate_async = AsyncMock(wraps=model.generate_async)
    rails = LLMRails(config, llm=model)
    messages = [{"role": "user", "content": "ORIGINAL query"}]
    first = await rails.generate_async(
        messages=messages, options={"llm_params": {"tool_choice": "required"}} if options else None
    )
    assistant = {**first.response[0], "tool_calls": first.tool_calls} if options else first
    if wire_format:
        assistant = copy.deepcopy(assistant)
        assistant.pop("content", None)
        for call in assistant["tool_calls"]:
            call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
    messages.append(assistant)
    assert get_history_cache_key(messages) in rails.events_history_cache
    messages.append({"role": "tool", "tool_call_id": "call_weather", "content": "Sunny"})
    await rails.generate_async(messages=messages, options={"llm_params": {"tool_choice": "auto"}} if options else None)
    prompt = model.generate_async.await_args.args[0]
    assert "REWRITTEN query" in prompt[0].content
    assert "ORIGINAL query" not in prompt[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize("block", [False, True])
async def test_cached_history_contains_only_approved_tool_calls(block):
    """Keep rejected calls out of cached prompts and preserve rail-approved argument rewrites."""
    config = RailsConfig.from_content(
        yaml_content=json.dumps(
            {"models": [], "passthrough": False, "rails": {"tool_call": {"flows": ["check calls"]}}}
        ),
        colang_content="""
define flow check calls
"""
        + (
            """  bot refuse call
  abort

define bot refuse call
  "Tool call rejected."
"""
            if block
            else """  $tool_calls = execute approve_calls(tool_calls=$tool_calls)
"""
        ),
    )
    model = FakeLLMModel(
        llm_responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_weather", function=ToolCallFunction(name="get_weather", arguments={"city": "ORIGINAL"})
                    )
                ],
            ),
            LLMResponse(content="Final answer."),
        ]
    )
    model.generate_async = AsyncMock(wraps=model.generate_async)
    rails = LLMRails(config, llm=model)

    async def approve_calls(tool_calls):
        """Copy the proposed calls and replace the first call arguments with the approved city."""
        approved = copy.deepcopy(tool_calls)
        approved[0]["function"]["arguments"] = {"city": "APPROVED"}
        return approved

    rails.register_action(approve_calls, name="approve_calls")
    messages = [{"role": "user", "content": "Weather?"}]
    first = await rails.generate_async(messages=messages)
    messages.append(first)
    if block:
        assert "tool_calls" not in first
        messages.append({"role": "user", "content": "Hello"})
    else:
        assert first["tool_calls"][0]["function"]["arguments"] == {"city": "APPROVED"}
        messages.append({"role": "tool", "tool_call_id": "call_weather", "content": "Sunny"})
    await rails.generate_async(messages=messages)
    prompt = model.generate_async.await_args.args[0]
    sent = (
        [{"role": "user", "content": prompt}] if isinstance(prompt, str) else [message.to_dict() for message in prompt]
    )
    assert "ORIGINAL" not in json.dumps(sent)
    if block:
        assert not any(message.get("tool_calls") for message in sent)
    else:
        assert next(message["tool_calls"] for message in sent if message.get("tool_calls")) == first["tool_calls"]


@pytest.mark.asyncio
@pytest.mark.parametrize("image_position", ["initial", "later"])
async def test_tool_prompt_budget_discounts_inline_image_data_without_changing_it(tool_history, image_position):
    """Exclude structured inline image payloads from the size budget while sending their original content."""
    image_content = [
        {"type": "text", "text": "Inspect this image."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 20000, "detail": "low"}},
    ]
    if image_position == "initial":
        tool_history[0]["content"] = image_content
    else:
        tool_history.extend(
            [
                {"role": "user", "content": image_content},
                {"role": "assistant", "content": "An image."},
                {"role": "user", "content": "Recheck the image."},
            ]
        )
    original = copy.deepcopy(tool_history)
    model = FakeLLMModel(responses=["Answer"])
    model.generate_async = AsyncMock(wraps=model.generate_async)
    rails = LLMRails(_tool_prompt_config(True, 1000), llm=model)
    result = await rails.generate_async(messages=tool_history)
    assert result == {"role": "assistant", "content": "Answer"}
    sent = [message.to_dict() for message in model.generate_async.await_args.args[0]]
    assert next(message["content"] for message in sent if isinstance(message.get("content"), list)) == image_content
    assert tool_history == original
    assert model.inference_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized_field", ["arguments", "result", "caption"])
async def test_tool_prompt_budget_counts_text_even_when_it_contains_image_data(tool_history, oversized_field):
    """Count data URLs fully when they occur as tool arguments, results, or text captions."""
    data_url = "data:image/png;base64," + "A" * 20000
    if oversized_field == "arguments":
        tool_history[1]["tool_calls"][0]["function"]["arguments"]["city"] = data_url
    elif oversized_field == "result":
        tool_history[-1]["content"] = data_url
    else:
        tool_history[0]["content"] = [
            {"type": "text", "text": data_url},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    model = FakeLLMModel(responses=["Must not be called"])
    rails = LLMRails(_tool_prompt_config(True, 1000), llm=model)
    with pytest.raises(ToolConversationTooLongError):
        await rails.generate_async(messages=tool_history)
    assert model.inference_count == 0

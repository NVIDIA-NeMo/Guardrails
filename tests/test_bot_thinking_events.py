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

from unittest.mock import patch

import pytest

from nemoguardrails import RailsConfig
from tests.conftest import REASONING_TRACE_MOCK_PATH
from tests.utils import TestChat


@pytest.mark.asyncio
async def test_bot_thinking_event_creation_passthrough():
    test_reasoning_trace = "Let me think about this step by step..."

    with patch(REASONING_TRACE_MOCK_PATH) as mock_get_reasoning:
        mock_get_reasoning.return_value = test_reasoning_trace

        config = RailsConfig.from_content(config={"models": [], "passthrough": True})
        chat = TestChat(config, llm_completions=["The answer is 42"])

        events = await chat.app.generate_events_async([{"type": "UserMessage", "text": "What is the answer?"}])

        bot_thinking_events = [e for e in events if e["type"] == "BotThinking"]
        assert len(bot_thinking_events) == 1
        assert bot_thinking_events[0]["content"] == test_reasoning_trace


@pytest.mark.asyncio
async def test_bot_thinking_event_creation_non_passthrough():
    test_reasoning_trace = "Analyzing the user's request..."

    with patch(REASONING_TRACE_MOCK_PATH) as mock_get_reasoning:
        mock_get_reasoning.return_value = test_reasoning_trace

        config = RailsConfig.from_content(
            colang_content="""
                define user ask question
                  "what is"

                define bot provide answer
                  "The answer is"

                define flow
                  user ask question
                  bot provide answer
            """,
        )
        chat = TestChat(
            config,
            llm_completions=[
                "  ask question",
                "  provide answer",
                '  "The answer is 42"',
            ],
        )

        events = await chat.app.generate_events_async([{"type": "UserMessage", "text": "what is the answer"}])

        bot_thinking_events = [e for e in events if e["type"] == "BotThinking"]
        assert len(bot_thinking_events) == 1
        assert bot_thinking_events[0]["content"] == test_reasoning_trace


@pytest.mark.asyncio
async def test_no_bot_thinking_event_when_no_reasoning_trace():
    with patch(REASONING_TRACE_MOCK_PATH) as mock_get_reasoning:
        mock_get_reasoning.return_value = None

        config = RailsConfig.from_content(config={"models": [], "passthrough": True})
        chat = TestChat(config, llm_completions=["Regular response"])

        events = await chat.app.generate_events_async([{"type": "UserMessage", "text": "Hello"}])

        bot_thinking_events = [e for e in events if e["type"] == "BotThinking"]
        assert len(bot_thinking_events) == 0


@pytest.mark.asyncio
async def test_bot_thinking_before_bot_message():
    test_reasoning_trace = "Step 1: Understand the question\nStep 2: Formulate answer"

    with patch(REASONING_TRACE_MOCK_PATH) as mock_get_reasoning:
        mock_get_reasoning.return_value = test_reasoning_trace

        config = RailsConfig.from_content(config={"models": [], "passthrough": True})
        chat = TestChat(config, llm_completions=["Response"])

        events = await chat.app.generate_events_async([{"type": "UserMessage", "text": "Test"}])

        bot_thinking_idx = None
        bot_message_idx = None

        for idx, event in enumerate(events):
            if event["type"] == "BotThinking":
                bot_thinking_idx = idx
            elif event["type"] == "BotMessage":
                bot_message_idx = idx

        assert bot_thinking_idx is not None
        assert bot_message_idx is not None
        assert bot_thinking_idx < bot_message_idx


@pytest.mark.asyncio
async def test_bot_thinking_accessible_in_output_rails():
    test_reasoning_trace = "Thinking: This requires careful consideration"

    with patch(REASONING_TRACE_MOCK_PATH) as mock_get_reasoning:
        mock_get_reasoning.return_value = test_reasoning_trace

        config = RailsConfig.from_content(
            colang_content="""
                define flow check_thinking_exists
                  if $bot_thinking
                    $thinking_was_present = True
                  else
                    $thinking_was_present = False
            """,
            yaml_content="""
                models: []
                passthrough: true
                rails:
                  output:
                    flows:
                      - check_thinking_exists
            """,
        )

        chat = TestChat(config, llm_completions=["Answer"])

        result = await chat.app.generate_async(
            messages=[{"role": "user", "content": "test"}],
            options={"output_vars": True},
        )

        assert result.output_data["thinking_was_present"] is True


@pytest.mark.asyncio
async def test_bot_thinking_matches_in_output_rails():
    test_reasoning_trace = "Let me analyze: step 1, step 2, step 3"

    with patch(REASONING_TRACE_MOCK_PATH) as mock_get_reasoning:
        mock_get_reasoning.return_value = test_reasoning_trace

        config = RailsConfig.from_content(
            colang_content="""
                define flow capture_thinking
                  $captured_thinking = $bot_thinking
            """,
            yaml_content="""
                models: []
                passthrough: true
                rails:
                  output:
                    flows:
                      - capture_thinking
            """,
        )

        chat = TestChat(config, llm_completions=["Response text"])

        result = await chat.app.generate_async(
            messages=[{"role": "user", "content": "query"}],
            options={"output_vars": True},
        )

        assert result.output_data["captured_thinking"] == test_reasoning_trace


@pytest.mark.asyncio
async def test_bot_thinking_none_when_no_reasoning():
    with patch(REASONING_TRACE_MOCK_PATH) as mock_get_reasoning:
        mock_get_reasoning.return_value = None

        config = RailsConfig.from_content(
            colang_content="""
                define flow check_no_thinking
                  if $bot_thinking
                    $thinking_exists = True
                  else
                    $thinking_exists = False
            """,
            yaml_content="""
                models: []
                passthrough: true
                rails:
                  output:
                    flows:
                      - check_no_thinking
            """,
        )

        chat = TestChat(config, llm_completions=["Response"])

        result = await chat.app.generate_async(
            messages=[{"role": "user", "content": "test"}],
            options={"output_vars": True},
        )

        assert result.output_data["thinking_exists"] is False


@pytest.mark.asyncio
async def test_bot_thinking_usable_in_output_rail_logic():
    test_reasoning_trace = "This contains sensitive information"

    with patch(REASONING_TRACE_MOCK_PATH) as mock_get_reasoning:
        mock_get_reasoning.return_value = test_reasoning_trace

        config = RailsConfig.from_content(
            colang_content="""
                define flow block_sensitive_thinking
                  if "sensitive" in $bot_thinking
                    bot refuse to respond
                    stop
            """,
            yaml_content="""
                models: []
                passthrough: true
                rails:
                  output:
                    flows:
                      - block_sensitive_thinking
            """,
        )

        chat = TestChat(config, llm_completions=["This is my response"])

        result = await chat.app.generate_async(
            messages=[{"role": "user", "content": "question"}],
            options={"output_vars": False},
        )

        assert isinstance(result.response, list)
        assert result.reasoning_content == test_reasoning_trace
        assert result.response[0]["content"] == "I'm sorry, I can't respond to that."
        assert test_reasoning_trace not in result.response[0]["content"]


@pytest.mark.asyncio
async def test_bot_message_accessible_in_output_rails_sanity_check():
    config = RailsConfig.from_content(
        colang_content="""
            define flow check_bot_message_exists
              if $bot_message
                $bot_message_was_present = True
              else
                $bot_message_was_present = False
        """,
        yaml_content="""
            models: []
            passthrough: true
            rails:
              output:
                flows:
                  - check_bot_message_exists
        """,
    )

    chat = TestChat(config, llm_completions=["Answer"])

    result = await chat.app.generate_async(
        messages=[{"role": "user", "content": "test"}], options={"output_vars": True}
    )

    assert result.output_data["bot_message_was_present"] is True


@pytest.mark.asyncio
async def test_extract_bot_thinking_from_events_util():
    from nemoguardrails.actions.llm.utils import extract_bot_thinking_from_events

    test_thinking = "Analysis of the situation"

    events = [
        {"type": "UserMessage", "text": "Hello"},
        {"type": "BotThinking", "content": test_thinking},
        {"type": "BotMessage", "text": "Response"},
    ]

    result = extract_bot_thinking_from_events(events)
    assert result == test_thinking


@pytest.mark.asyncio
async def test_extract_bot_thinking_returns_none_when_not_present():
    from nemoguardrails.actions.llm.utils import extract_bot_thinking_from_events

    events = [
        {"type": "UserMessage", "text": "Hello"},
        {"type": "BotMessage", "text": "Response"},
    ]

    result = extract_bot_thinking_from_events(events)
    assert result is None


@pytest.mark.asyncio
async def test_extract_bot_thinking_with_empty_events_list():
    from nemoguardrails.actions.llm.utils import extract_bot_thinking_from_events

    events = []

    result = extract_bot_thinking_from_events(events)
    assert result is None


@pytest.mark.asyncio
async def test_extract_bot_thinking_returns_first_when_multiple():
    from nemoguardrails.actions.llm.utils import extract_bot_thinking_from_events

    first_thinking = "First thought process"
    second_thinking = "Second thought process"

    events = [
        {"type": "UserMessage", "text": "Hello"},
        {"type": "BotThinking", "content": first_thinking},
        {"type": "BotMessage", "text": "Response"},
        {"type": "BotThinking", "content": second_thinking},
    ]

    result = extract_bot_thinking_from_events(events)
    assert result == first_thinking


@pytest.mark.asyncio
async def test_extract_bot_thinking_with_empty_content():
    from nemoguardrails.actions.llm.utils import extract_bot_thinking_from_events

    events = [
        {"type": "UserMessage", "text": "Hello"},
        {"type": "BotThinking", "content": ""},
        {"type": "BotMessage", "text": "Response"},
    ]

    result = extract_bot_thinking_from_events(events)
    assert result == ""


@pytest.mark.asyncio
async def test_extract_bot_thinking_with_none_content():
    from nemoguardrails.actions.llm.utils import extract_bot_thinking_from_events

    events = [
        {"type": "UserMessage", "text": "Hello"},
        {"type": "BotThinking", "content": None},
        {"type": "BotMessage", "text": "Response"},
    ]

    result = extract_bot_thinking_from_events(events)
    assert result is None


@pytest.mark.asyncio
async def test_extract_bot_thinking_without_content_field():
    from nemoguardrails.actions.llm.utils import extract_bot_thinking_from_events

    events = [
        {"type": "UserMessage", "text": "Hello"},
        {"type": "BotThinking"},
        {"type": "BotMessage", "text": "Response"},
    ]

    result = extract_bot_thinking_from_events(events)
    assert result is None


@pytest.mark.asyncio
async def test_extract_bot_thinking_with_multiline_content():
    from nemoguardrails.actions.llm.utils import extract_bot_thinking_from_events

    multiline_thinking = """Step 1: Analyze the request
Step 2: Consider the context
Step 3: Formulate response"""

    events = [
        {"type": "UserMessage", "text": "Hello"},
        {"type": "BotThinking", "content": multiline_thinking},
        {"type": "BotMessage", "text": "Response"},
    ]

    result = extract_bot_thinking_from_events(events)
    assert result == multiline_thinking


@pytest.mark.asyncio
async def test_extract_bot_thinking_with_special_characters():
    from nemoguardrails.actions.llm.utils import extract_bot_thinking_from_events

    special_thinking = "Thinking: <test> \"quoted\" 'text' & symbols!"

    events = [
        {"type": "UserMessage", "text": "Hello"},
        {"type": "BotThinking", "content": special_thinking},
        {"type": "BotMessage", "text": "Response"},
    ]

    result = extract_bot_thinking_from_events(events)
    assert result == special_thinking


@pytest.mark.asyncio
async def test_extract_bot_thinking_with_numeric_content():
    from nemoguardrails.actions.llm.utils import extract_bot_thinking_from_events

    events = [
        {"type": "UserMessage", "text": "Hello"},
        {"type": "BotThinking", "content": 12345},
        {"type": "BotMessage", "text": "Response"},
    ]

    result = extract_bot_thinking_from_events(events)
    assert result == 12345


@pytest.mark.asyncio
async def test_extract_bot_thinking_with_only_bot_thinking_event():
    from nemoguardrails.actions.llm.utils import extract_bot_thinking_from_events

    thinking = "Isolated thought"

    events = [{"type": "BotThinking", "content": thinking}]

    result = extract_bot_thinking_from_events(events)
    assert result == thinking


@pytest.mark.asyncio
async def test_extract_bot_thinking_ignores_similar_event_types():
    from nemoguardrails.actions.llm.utils import extract_bot_thinking_from_events

    events = [
        {"type": "UserMessage", "text": "Hello"},
        {"type": "BotThinkingStarted", "content": "Should be ignored"},
        {"type": "PreBotThinking", "content": "Should be ignored"},
        {"type": "BotMessage", "text": "Response"},
    ]

    result = extract_bot_thinking_from_events(events)
    assert result is None

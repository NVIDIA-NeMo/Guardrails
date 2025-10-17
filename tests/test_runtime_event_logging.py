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

import logging
from unittest.mock import patch

import pytest

from nemoguardrails import RailsConfig
from tests.utils import TestChat


@pytest.mark.asyncio
async def test_bot_thinking_event_logged_in_runtime(caplog):
    test_reasoning_trace = "Let me think about this step by step..."

    caplog.set_level(logging.INFO)

    with patch(
        "nemoguardrails.actions.llm.generation.get_and_clear_reasoning_trace_contextvar"
    ) as mock_get_reasoning:
        mock_get_reasoning.return_value = test_reasoning_trace

        config = RailsConfig.from_content(config={"models": [], "passthrough": True})
        chat = TestChat(config, llm_completions=["The answer is 42"])

        await chat.app.generate_events_async(
            [{"type": "UserMessage", "text": "What is the answer?"}]
        )

        bot_thinking_logs = [
            record
            for record in caplog.records
            if "Event :: BotThinking" in record.message
        ]
        assert len(bot_thinking_logs) >= 1


@pytest.mark.asyncio
async def test_bot_message_event_logged_in_runtime(caplog):
    caplog.set_level(logging.INFO)

    config = RailsConfig.from_content(config={"models": [], "passthrough": True})
    chat = TestChat(config, llm_completions=["The answer is 42"])

    await chat.app.generate_events_async(
        [{"type": "UserMessage", "text": "What is the answer?"}]
    )

    bot_message_logs = [
        record for record in caplog.records if "Event :: BotMessage" in record.message
    ]
    assert len(bot_message_logs) >= 1


@pytest.mark.asyncio
async def test_context_update_event_logged_in_runtime(caplog):
    caplog.set_level(logging.INFO)

    config = RailsConfig.from_content(config={"models": [], "passthrough": True})
    chat = TestChat(config, llm_completions=["Response"])

    await chat.app.generate_events_async([{"type": "UserMessage", "text": "Hello"}])

    context_update_logs = [
        record
        for record in caplog.records
        if "Event :: ContextUpdate" in record.message
        or "Event ContextUpdate" in record.message
    ]
    assert len(context_update_logs) >= 1


@pytest.mark.asyncio
async def test_all_events_logged_when_multiple_events_generated(caplog):
    test_reasoning_trace = "Analyzing..."

    caplog.set_level(logging.INFO)

    with patch(
        "nemoguardrails.actions.llm.generation.get_and_clear_reasoning_trace_contextvar"
    ) as mock_get_reasoning:
        mock_get_reasoning.return_value = test_reasoning_trace

        config = RailsConfig.from_content(config={"models": [], "passthrough": True})
        chat = TestChat(config, llm_completions=["Response"])

        await chat.app.generate_events_async([{"type": "UserMessage", "text": "Test"}])

        bot_thinking_found = any(
            "Event :: BotThinking" in record.message for record in caplog.records
        )
        bot_message_found = any(
            "Event :: BotMessage" in record.message for record in caplog.records
        )

        assert bot_thinking_found
        assert bot_message_found


@pytest.mark.asyncio
async def test_bot_thinking_event_logged_before_bot_message(caplog):
    test_reasoning_trace = "Step 1: Understand\nStep 2: Respond"

    caplog.set_level(logging.INFO)

    with patch(
        "nemoguardrails.actions.llm.generation.get_and_clear_reasoning_trace_contextvar"
    ) as mock_get_reasoning:
        mock_get_reasoning.return_value = test_reasoning_trace

        config = RailsConfig.from_content(config={"models": [], "passthrough": True})
        chat = TestChat(config, llm_completions=["Answer"])

        await chat.app.generate_events_async(
            [{"type": "UserMessage", "text": "Question"}]
        )

        bot_thinking_idx = None
        bot_message_idx = None

        for idx, record in enumerate(caplog.records):
            if "Event :: BotThinking" in record.message and bot_thinking_idx is None:
                bot_thinking_idx = idx
            if "Event :: BotMessage" in record.message and bot_message_idx is None:
                bot_message_idx = idx

        assert bot_thinking_idx is not None
        assert bot_message_idx is not None
        assert bot_thinking_idx < bot_message_idx


@pytest.mark.asyncio
async def test_event_history_update_not_logged(caplog):
    caplog.set_level(logging.INFO)

    config = RailsConfig.from_content(config={"models": [], "passthrough": True})
    chat = TestChat(config, llm_completions=["Response"])

    await chat.app.generate_events_async([{"type": "UserMessage", "text": "Test"}])

    event_history_update_logs = [
        record
        for record in caplog.records
        if "Event :: EventHistoryUpdate" in record.message
        or "Event EventHistoryUpdate" in record.message
    ]
    assert len(event_history_update_logs) == 0

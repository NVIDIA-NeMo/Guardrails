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

import os
from unittest.mock import AsyncMock, patch

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.imports import check_optional_dependency
from nemoguardrails.rails.llm.options import GenerationOptions
from tests.utils import TestChat

_has_langchain_openai = check_optional_dependency("langchain_openai")

_has_openai_key = bool(os.getenv("OPENAI_API_KEY"))

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")

OPTIONS = GenerationOptions(
    log={
        "activated_rails": True,
        "llm_calls": True,
        "internal_events": True,
        "colang_history": True,
    }
)


@pytest.mark.asyncio
async def test_internal_error_stops_execution():
    """Test that internal errors trigger stop execution to prevent further LLM generation."""
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails"))

    # mock the render_task_prompt method to raise an exception (simulating missing prompt)
    with patch("nemoguardrails.llm.taskmanager.LLMTaskManager.render_task_prompt") as mock_render:
        mock_render.side_effect = Exception("Missing prompt for task: self_check_input")

        chat = TestChat(config, llm_completions=["Hello!"])
        chat >> "hi"

        result = await chat.app.generate_async(messages=chat.history, options=OPTIONS)

        # should get an internal error response
        assert result is not None
        assert "internal error" in result.response[0]["content"].lower()

        # check that a BotIntent stop event was generated in the internal events
        stop_events = [
            event
            for event in result.log.internal_events
            if event.get("type") == "BotIntent" and event.get("intent") == "stop"
        ]
        assert len(stop_events) > 0, "Expected BotIntent stop event after internal error"


@pytest.mark.skipif(
    not _has_langchain_openai or not _has_openai_key,
    reason="langchain-openai not available",
)
@pytest.mark.asyncio
async def test_no_app_llm_request_on_internal_error():
    """Test that App LLM request is not sent when internal error occurs."""
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails"))

    # mock the render_task_prompt method to raise an exception
    with patch("nemoguardrails.llm.taskmanager.LLMTaskManager.render_task_prompt") as mock_render:
        mock_render.side_effect = Exception("Missing prompt for task: self_check_input")

        with patch("nemoguardrails.actions.llm.utils.llm_call", new_callable=AsyncMock) as mock_llm_call:
            mock_llm_call.return_value = "Mocked response"

            chat = TestChat(config, llm_completions=["Test response"])
            chat >> "test"

            result = await chat.app.generate_async(messages=chat.history, options=OPTIONS)

            # should get internal error response
            assert result is not None
            assert "internal error" in result.response[0]["content"].lower()

            # verify that the main LLM was NOT called (no App LLM request sent)
            # The LLM call should be 0 because execution stopped after internal error
            assert mock_llm_call.call_count == 0, f"Expected 0 LLM calls, but got {mock_llm_call.call_count}"

            # verify BotIntent stop event was generated
            stop_events = [
                event
                for event in result.log.internal_events
                if event.get("type") == "BotIntent" and event.get("intent") == "stop"
            ]
            assert len(stop_events) > 0, "Expected BotIntent stop event after internal error"


@pytest.mark.asyncio
async def test_parallel_rails_partial_failure():
    """Test that partial failure in parallel rails is handled properly."""
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails"))

    chat = TestChat(
        config,
        llm_completions=[
            "No",  # self check input
            "Hi there! How can I help?",  # main response
            "No",  # self check output
        ],
    )
    chat >> "hi"

    result = await chat.app.generate_async(messages=chat.history, options=OPTIONS)

    # should get successful response (not internal error)
    assert result is not None
    assert "internal error" not in result.response[0]["content"].lower()
    assert "Hi there! How can I help?" in result.response[0]["content"]


@pytest.mark.asyncio
async def test_no_stop_event_without_error():
    """Test that normal execution doesn't generate unnecessary stop events."""
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails"))

    chat = TestChat(
        config,
        llm_completions=[
            "No",  # self check input passes
            "Hi there! How can I help?",  # main response
            "No",  # self check output passes
        ],
    )

    chat >> "hi"
    result = await chat.app.generate_async(messages=chat.history, options=OPTIONS)

    assert result is not None
    assert "Hi there! How can I help?" in result.response[0]["content"]

    # should not contain "internal error" in normal execution
    assert "internal error" not in result.response[0]["content"].lower()


@pytest.mark.asyncio
async def test_internal_error_adds_three_specific_events():
    """Minimal test to verify the exact events added by the fix.

    The fix in runtime.py adds these events when an internal error occurs:
    1. BotIntent with intent="inform internal error occurred"
    2. StartUtteranceBotAction with error message
    3. hide_prev_turn
    4. BotIntent with intent="stop"
    """
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails"))

    # mock render_task_prompt to trigger an internal error
    with patch("nemoguardrails.llm.taskmanager.LLMTaskManager.render_task_prompt") as mock_render:
        mock_render.side_effect = Exception("Test internal error")

        chat = TestChat(config, llm_completions=["Test response"])
        chat >> "test"

        result = await chat.app.generate_async(messages=chat.history, options=OPTIONS)

        # find the BotIntent with "inform internal error occurred"
        error_event_index = None
        for i, event in enumerate(result.log.internal_events):
            if event.get("type") == "BotIntent" and event.get("intent") == "inform internal error occurred":
                error_event_index = i
                break

        assert error_event_index is not None, "Expected BotIntent with intent='inform internal error occurred'"

        assert error_event_index + 3 < len(result.log.internal_events), (
            "Expected at least 4 events total for error handling"
        )

        utterance_event = result.log.internal_events[error_event_index + 1]
        assert utterance_event.get("type") == "StartUtteranceBotAction", (
            f"Expected StartUtteranceBotAction after error, got {utterance_event.get('type')}"
        )

        hide_event = result.log.internal_events[error_event_index + 2]
        assert hide_event.get("type") == "hide_prev_turn", (
            f"Expected hide_prev_turn after utterance, got {hide_event.get('type')}"
        )

        stop_event = result.log.internal_events[error_event_index + 3]
        assert stop_event.get("type") == "BotIntent", (
            f"Expected BotIntent after hide_prev_turn, got {stop_event.get('type')}"
        )
        assert stop_event.get("intent") == "stop", f"Expected intent='stop', got {stop_event.get('intent')}"


@pytest.mark.asyncio
async def test_action_execution_returns_failed():
    """Test that when an action returns 'failed' status, BotIntent stop event is generated."""
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "parallel_rails"))

    # mock execute_action to return failed status
    with patch(
        "nemoguardrails.actions.action_dispatcher.ActionDispatcher.execute_action",
        return_value=(None, "failed"),
    ):
        chat = TestChat(config, llm_completions=["Test response"])
        chat >> "test"

        result = await chat.app.generate_async(messages=chat.history, options=OPTIONS)

        # should get internal error response
        assert result is not None
        assert "internal error" in result.response[0]["content"].lower()

        # verify BotIntent stop event is generated (the key fix)
        stop_events = [
            event
            for event in result.log.internal_events
            if event.get("type") == "BotIntent" and event.get("intent") == "stop"
        ]
        assert len(stop_events) > 0, "Expected BotIntent stop event after action failure"


@pytest.mark.skipif(
    not _has_langchain_openai or not _has_openai_key,
    reason="langchain-openai not available",
)
@pytest.mark.asyncio
async def test_single_error_message_not_multiple():
    """Test that we get exactly one error message, not multiple for each failed rail.

    Before the fix, if we had multiple rails failing, we'd get multiple error messages.
    This test verifies we only get one error message even with multiple parallel rails.
    Now with config-time validation, we provide valid config and trigger runtime failures.
    """
    config_data = {
        "models": [
            {"type": "main", "engine": "openai", "model": "gpt-3.5-turbo"},
            {"type": "content_safety", "engine": "openai", "model": "gpt-3.5-turbo"},
        ],
        "rails": {
            "input": {
                "flows": [
                    "self check input",
                    "content safety check input $model=content_safety",
                ],
                "parallel": True,
            }
        },
        "prompts": [
            {
                "task": "self_check_input",
                "content": "Is the user input safe? Answer Yes or No.",
            },
            {
                "task": "content_safety_check_input $model=content_safety",
                "content": "Check content safety: {{ user_input }}",
            },
        ],
    }

    config = RailsConfig.from_content(config=config_data)

    with patch("nemoguardrails.llm.taskmanager.LLMTaskManager.render_task_prompt") as mock_render:
        mock_render.side_effect = Exception("Runtime error in multiple rails")

        chat = TestChat(config, llm_completions=["Test response"])
        chat >> "test message"

        result = await chat.app.generate_async(messages=chat.history, options=OPTIONS)

        # should get exactly one response, not multiple
        assert result is not None
        assert len(result.response) == 1, f"Expected 1 response, got {len(result.response)}"

        # that single response should be an internal error
        assert "internal error" in result.response[0]["content"].lower()

        # count how many times "internal error" appears in the response
        error_count = result.response[0]["content"].lower().count("internal error")
        assert error_count == 1, f"Expected 1 'internal error' message, found {error_count}"

        # verify stop event was generated
        stop_events = [
            event
            for event in result.log.internal_events
            if event.get("type") == "BotIntent" and event.get("intent") == "stop"
        ]
        assert len(stop_events) >= 1, "Expected at least one BotIntent stop event"

        # verify we don't have multiple StartUtteranceBotAction events with error messages
        error_utterances = [
            event
            for event in result.log.internal_events
            if event.get("type") == "StartUtteranceBotAction" and "internal error" in event.get("script", "").lower()
        ]
        assert len(error_utterances) == 1, f"Expected 1 error utterance, found {len(error_utterances)}"

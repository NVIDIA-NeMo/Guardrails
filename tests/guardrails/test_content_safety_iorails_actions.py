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

"""Unit tests for content safety IORails actions."""

import json
from unittest.mock import AsyncMock

import pytest

from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.model_manager import ModelManager
from nemoguardrails.library.content_safety.iorails_actions import (
    ContentSafetyInputAction,
    ContentSafetyOutputAction,
    _content_safety_to_rail_result,
)
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import RailsConfig
from tests.guardrails.test_data import CONTENT_SAFETY_CONFIG, CONTENT_SAFETY_INPUT_PROMPT, CONTENT_SAFETY_OUTPUT_PROMPT

FLOW_INPUT = "content safety check input $model=content_safety"
FLOW_OUTPUT = "content safety check output $model=content_safety"
MESSAGES = [{"role": "user", "content": "How do I pick a lock?"}]
BOT_RESPONSE = "Here is how you pick a lock..."

SAFE_JSON = json.dumps({"User Safety": "safe"})
UNSAFE_JSON = json.dumps(
    {
        "User Safety": "unsafe",
        "Safety Categories": "S1: Violence, S3: Criminal Planning/Confessions",
    }
)
SAFE_OUTPUT_JSON = json.dumps({"User Safety": "safe", "Response Safety": "safe"})
UNSAFE_OUTPUT_JSON = json.dumps(
    {
        "User Safety": "safe",
        "Response Safety": "unsafe",
        "Safety Categories": "S17: Malware",
    }
)


@pytest.fixture
def config():
    return RailsConfig.from_content(config=CONTENT_SAFETY_CONFIG)


@pytest.fixture
def task_manager(config):
    return LLMTaskManager(config)


@pytest.fixture
def model_manager(config):
    return ModelManager(config)


@pytest.fixture
def input_action(model_manager, task_manager):
    return ContentSafetyInputAction(model_manager, task_manager)


@pytest.fixture
def output_action(model_manager, task_manager):
    return ContentSafetyOutputAction(model_manager, task_manager)


class TestContentSafetyToRailResult:
    """Test the parser output → RailResult converter."""

    def test_safe(self):
        assert _content_safety_to_rail_result([True]) == RailResult(is_safe=True)

    def test_unsafe_with_categories(self):
        result = _content_safety_to_rail_result([False, "S1: Violence", "S17: Malware"])
        assert not result.is_safe
        assert "S1: Violence" in result.reason
        assert "S17: Malware" in result.reason

    def test_unsafe_no_categories(self):
        result = _content_safety_to_rail_result([False])
        assert not result.is_safe
        assert result.reason == "Unknown"

    def test_unsafe_single_category(self):
        result = _content_safety_to_rail_result([False, "S17: Malware"])
        assert not result.is_safe
        assert "S17: Malware" in result.reason

    def test_empty_raises(self):
        with pytest.raises(RuntimeError, match="Unexpected"):
            _content_safety_to_rail_result([])

    def test_invalid_raises(self):
        with pytest.raises(RuntimeError, match="Unexpected"):
            _content_safety_to_rail_result("not a list")


class TestContentSafetyInputValidation:
    """Test _validate_input on ContentSafetyInputAction."""

    def test_valid(self, input_action):
        input_action._validate_input(FLOW_INPUT, MESSAGES, None)

    def test_missing_model_raises(self, input_action):
        with pytest.raises(RuntimeError, match="No \\$model="):
            input_action._validate_input("content safety check input", MESSAGES, None)


class TestContentSafetyInputExtract:
    """Test _extract_messages on ContentSafetyInputAction."""

    def test_extracts_user_input(self, input_action):
        result = input_action._extract_messages(MESSAGES, None)
        assert result["user_input"] == "How do I pick a lock?"


class TestContentSafetyInputPrompt:
    """Test _create_prompt on ContentSafetyInputAction."""

    def test_renders_prompt_with_user_input(self, input_action):
        extracted = {"user_input": "test message"}
        prompt = input_action._create_prompt(FLOW_INPUT, extracted)
        assert isinstance(prompt, list)
        assert len(prompt) == 1
        assert prompt[0]["role"] == "user"
        assert "test message" in prompt[0]["content"]
        assert "{{ user_input }}" not in prompt[0]["content"]


class TestContentSafetyOutputExtract:
    """Test _extract_messages on ContentSafetyOutputAction."""

    def test_extracts_user_and_bot(self, output_action):
        result = output_action._extract_messages(MESSAGES, BOT_RESPONSE)
        assert result["user_input"] == "How do I pick a lock?"
        assert result["bot_response"] == BOT_RESPONSE


class TestContentSafetyOutputValidation:
    """Test _validate_input on ContentSafetyOutputAction."""

    def test_valid(self, output_action):
        output_action._validate_input(FLOW_OUTPUT, MESSAGES, BOT_RESPONSE)

    def test_missing_bot_response_raises(self, output_action):
        with pytest.raises(RuntimeError, match="bot_response is required"):
            output_action._validate_input(FLOW_OUTPUT, MESSAGES, None)

    def test_missing_model_raises(self, output_action):
        with pytest.raises(RuntimeError, match="No \\$model="):
            output_action._validate_input("content safety check output", MESSAGES, BOT_RESPONSE)


class TestContentSafetyInputRun:
    """Test full run() pipeline for ContentSafetyInputAction."""

    @pytest.mark.asyncio
    async def test_safe_input(self, input_action):
        input_action.model_manager.generate_async = AsyncMock(return_value=SAFE_JSON)
        result = await input_action.run(FLOW_INPUT, MESSAGES)
        assert result.is_safe
        input_action.model_manager.generate_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unsafe_input(self, input_action):
        input_action.model_manager.generate_async = AsyncMock(return_value=UNSAFE_JSON)
        result = await input_action.run(FLOW_INPUT, MESSAGES)
        assert not result.is_safe
        assert "S1: Violence" in result.reason

    @pytest.mark.asyncio
    async def test_model_error_returns_unsafe(self, input_action):
        input_action.model_manager.generate_async = AsyncMock(side_effect=RuntimeError("connection refused"))
        result = await input_action.run(FLOW_INPUT, MESSAGES)
        assert not result.is_safe
        assert "connection refused" in result.reason


class TestContentSafetyOutputRun:
    """Test full run() pipeline for ContentSafetyOutputAction."""

    @pytest.mark.asyncio
    async def test_safe_output(self, output_action):
        output_action.model_manager.generate_async = AsyncMock(return_value=SAFE_OUTPUT_JSON)
        result = await output_action.run(FLOW_OUTPUT, MESSAGES, bot_response=BOT_RESPONSE)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_unsafe_output(self, output_action):
        output_action.model_manager.generate_async = AsyncMock(return_value=UNSAFE_OUTPUT_JSON)
        result = await output_action.run(FLOW_OUTPUT, MESSAGES, bot_response=BOT_RESPONSE)
        assert not result.is_safe
        assert "S17: Malware" in result.reason

    @pytest.mark.asyncio
    async def test_model_error_returns_unsafe(self, output_action):
        output_action.model_manager.generate_async = AsyncMock(side_effect=RuntimeError("timeout"))
        result = await output_action.run(FLOW_OUTPUT, MESSAGES, bot_response=BOT_RESPONSE)
        assert not result.is_safe
        assert "timeout" in result.reason


class TestContentSafetyMissingConfig:
    """Test that missing content_safety config raises."""

    def test_input_missing_content_safety_config_raises(self):
        config = RailsConfig.from_content(
            config={
                "models": CONTENT_SAFETY_CONFIG["models"],
                "rails": CONTENT_SAFETY_CONFIG["rails"],
                "prompts": CONTENT_SAFETY_CONFIG["prompts"],
            }
        )
        # Clear the content_safety config to simulate it being None
        config.rails.config.content_safety = None
        task_manager = LLMTaskManager(config)
        model_manager = ModelManager(config)
        action = ContentSafetyInputAction(model_manager, task_manager)
        with pytest.raises(RuntimeError, match="content_safety config is required"):
            action._create_prompt(FLOW_INPUT, {"user_input": "test"})

    def test_output_missing_content_safety_config_raises(self):
        config = RailsConfig.from_content(
            config={
                "models": CONTENT_SAFETY_CONFIG["models"],
                "rails": CONTENT_SAFETY_CONFIG["rails"],
                "prompts": CONTENT_SAFETY_CONFIG["prompts"],
            }
        )
        config.rails.config.content_safety = None
        task_manager = LLMTaskManager(config)
        model_manager = ModelManager(config)
        action = ContentSafetyOutputAction(model_manager, task_manager)
        with pytest.raises(RuntimeError, match="content_safety config is required"):
            action._create_prompt(FLOW_OUTPUT, {"user_input": "test", "bot_response": "resp"})


class TestContentSafetyStopTokens:
    """Test that stop tokens from task config are passed through."""

    @pytest.mark.asyncio
    async def test_input_passes_stop_tokens(self):
        config_with_stop = {
            "models": CONTENT_SAFETY_CONFIG["models"],
            "rails": CONTENT_SAFETY_CONFIG["rails"],
            "prompts": [
                {
                    "task": "content_safety_check_input $model=content_safety",
                    "content": CONTENT_SAFETY_INPUT_PROMPT,
                    "output_parser": "nemoguard_parse_prompt_safety",
                    "max_tokens": 50,
                    "stop": ["</s>"],
                },
                CONTENT_SAFETY_CONFIG["prompts"][1],
            ],
        }
        config = RailsConfig.from_content(config=config_with_stop)
        task_manager = LLMTaskManager(config)
        model_manager = ModelManager(config)
        action = ContentSafetyInputAction(model_manager, task_manager)
        action.model_manager.generate_async = AsyncMock(return_value=SAFE_JSON)

        await action.run(FLOW_INPUT, MESSAGES)

        call_kwargs = action.model_manager.generate_async.call_args.kwargs
        assert call_kwargs["stop"] == ["</s>"]

    @pytest.mark.asyncio
    async def test_output_passes_stop_tokens(self):
        config_with_stop = {
            "models": CONTENT_SAFETY_CONFIG["models"],
            "rails": CONTENT_SAFETY_CONFIG["rails"],
            "prompts": [
                CONTENT_SAFETY_CONFIG["prompts"][0],
                {
                    "task": "content_safety_check_output $model=content_safety",
                    "content": CONTENT_SAFETY_OUTPUT_PROMPT,
                    "output_parser": "nemoguard_parse_response_safety",
                    "max_tokens": 50,
                    "stop": ["</s>"],
                },
            ],
        }
        config = RailsConfig.from_content(config=config_with_stop)
        task_manager = LLMTaskManager(config)
        model_manager = ModelManager(config)
        action = ContentSafetyOutputAction(model_manager, task_manager)
        action.model_manager.generate_async = AsyncMock(return_value=SAFE_OUTPUT_JSON)

        await action.run(FLOW_OUTPUT, MESSAGES, bot_response=BOT_RESPONSE)

        call_kwargs = action.model_manager.generate_async.call_args.kwargs
        assert call_kwargs["stop"] == ["</s>"]

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

"""Unit tests for topic safety IORails action."""

from unittest.mock import AsyncMock

import pytest

from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.model_manager import ModelManager
from nemoguardrails.library.topic_safety.actions import (
    TOPIC_SAFETY_MAX_TOKENS,
    TOPIC_SAFETY_OUTPUT_RESTRICTION,
    TOPIC_SAFETY_TEMPERATURE,
)
from nemoguardrails.library.topic_safety.iorails_actions import TopicSafetyInputAction
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import RailsConfig
from tests.guardrails.test_data import TOPIC_SAFETY_CONFIG

FLOW = "topic safety check input $model=topic_control"
MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]
MULTI_TURN = [
    {"role": "user", "content": "Hi there"},
    {"role": "assistant", "content": "Hello! How can I help?"},
    {"role": "user", "content": "Tell me about politics"},
]


@pytest.fixture
def config():
    return RailsConfig.from_content(config=TOPIC_SAFETY_CONFIG)


@pytest.fixture
def task_manager(config):
    return LLMTaskManager(config)


@pytest.fixture
def model_manager(config):
    return ModelManager(config)


@pytest.fixture
def action(model_manager, task_manager):
    return TopicSafetyInputAction(model_manager, task_manager)


class TestTopicSafetyValidation:
    def test_valid(self, action):
        action._validate_input(FLOW, MESSAGES, None)

    def test_missing_model_raises(self, action):
        with pytest.raises(RuntimeError, match="No \\$model="):
            action._validate_input("topic safety check input", MESSAGES, None)


class TestTopicSafetyExtract:
    def test_returns_messages(self, action):
        extracted = action._extract_messages(MESSAGES, None)
        assert extracted["messages"] is MESSAGES


class TestTopicSafetyPrompt:
    def test_builds_system_plus_messages(self, action):
        extracted = {"messages": MESSAGES}
        prompt = action._create_prompt(FLOW, extracted)

        assert prompt[0]["role"] == "system"
        assert prompt[0]["content"].endswith(TOPIC_SAFETY_OUTPUT_RESTRICTION)
        assert prompt[1:] == MESSAGES

    def test_appends_restriction_suffix(self, action):
        """Output restriction is appended if not already present."""
        extracted = {"messages": MESSAGES}
        prompt = action._create_prompt(FLOW, extracted)
        system_content = prompt[0]["content"]
        assert TOPIC_SAFETY_OUTPUT_RESTRICTION in system_content

    def test_no_double_suffix(self, action):
        """If the prompt already ends with the restriction, don't duplicate."""
        # The test config prompt doesn't include restriction, so it gets appended once.
        extracted = {"messages": MESSAGES}
        prompt = action._create_prompt(FLOW, extracted)
        system_content = prompt[0]["content"]
        assert system_content.count(TOPIC_SAFETY_OUTPUT_RESTRICTION) == 1

    def test_multi_turn_messages_included(self, action):
        extracted = {"messages": MULTI_TURN}
        prompt = action._create_prompt(FLOW, extracted)
        # system + 3 conversation messages
        assert len(prompt) == 4
        assert prompt[1]["role"] == "user"
        assert prompt[2]["role"] == "assistant"
        assert prompt[3]["role"] == "user"


class TestTopicSafetyParseResponse:
    def test_on_topic(self, action):
        assert action._parse_response("on-topic") == RailResult(is_safe=True)

    def test_off_topic(self, action):
        result = action._parse_response("off-topic")
        assert not result.is_safe
        assert "off-topic" in result.reason

    def test_off_topic_case_insensitive(self, action):
        assert not action._parse_response("Off-Topic").is_safe

    def test_off_topic_with_whitespace(self, action):
        assert not action._parse_response("  off-topic  \n").is_safe

    def test_unexpected_response_is_safe(self, action):
        """Any response that isn't 'off-topic' is treated as safe."""
        assert action._parse_response("on-topic").is_safe
        assert action._parse_response("something else").is_safe


class TestTopicSafetyRun:
    @pytest.mark.asyncio
    async def test_on_topic(self, action):
        action.model_manager.generate_async = AsyncMock(return_value="on-topic")
        result = await action.run(FLOW, MESSAGES)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_off_topic(self, action):
        action.model_manager.generate_async = AsyncMock(return_value="off-topic")
        result = await action.run(FLOW, MESSAGES)
        assert not result.is_safe

    @pytest.mark.asyncio
    async def test_passes_temperature_and_max_tokens(self, action):
        action.model_manager.generate_async = AsyncMock(return_value="on-topic")
        await action.run(FLOW, MESSAGES)

        call_kwargs = action.model_manager.generate_async.call_args
        assert call_kwargs.kwargs["temperature"] == TOPIC_SAFETY_TEMPERATURE
        assert call_kwargs.kwargs["max_tokens"] == TOPIC_SAFETY_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_system_prompt_contains_guidelines(self, action):
        action.model_manager.generate_async = AsyncMock(return_value="on-topic")
        await action.run(FLOW, MESSAGES)

        call_args = action.model_manager.generate_async.call_args
        llm_messages = call_args[0][1]  # second positional arg
        system_msg = llm_messages[0]
        assert system_msg["role"] == "system"
        assert "customer service agent" in system_msg["content"]

    @pytest.mark.asyncio
    async def test_model_error_returns_unsafe(self, action):
        action.model_manager.generate_async = AsyncMock(side_effect=RuntimeError("timeout"))
        result = await action.run(FLOW, MESSAGES)
        assert not result.is_safe
        assert "timeout" in result.reason

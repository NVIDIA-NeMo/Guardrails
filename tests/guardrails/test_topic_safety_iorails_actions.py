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

from nemoguardrails.guardrails.actions.topic_safety_action import TopicSafetyInputAction
from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.library.topic_safety.actions import (
    TOPIC_SAFETY_MAX_TOKENS,
    TOPIC_SAFETY_OUTPUT_RESTRICTION,
    TOPIC_SAFETY_TEMPERATURE,
)
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.types import LLMResponse
from tests.guardrails.test_data import TOPIC_SAFETY_CONFIG, TOPIC_SAFETY_INPUT_PROMPT

FLOW = "topic safety check input $model=topic_control"
MODEL_TYPE = "topic_control"
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
def engine_registry(config):
    return EngineRegistry(config.models, config.rails.config)


@pytest.fixture
def action(engine_registry, task_manager):
    return TopicSafetyInputAction(engine_registry, task_manager)


class TestTopicSafetyMissingModel:
    @pytest.mark.asyncio
    async def test_missing_model_raises(self, action):
        with pytest.raises(RuntimeError, match="No \\$model="):
            await action.run("topic safety check input", MESSAGES)


class TestTopicSafetyExtract:
    def test_returns_messages(self, action):
        extracted = action._extract_messages(MESSAGES, None)
        assert extracted["messages"] is MESSAGES


class TestTopicSafetyPrompt:
    def test_builds_system_plus_messages(self, action):
        prompt = action._create_prompt(MODEL_TYPE, {"messages": MESSAGES})
        assert prompt[0]["role"] == "system"
        assert prompt[0]["content"].endswith(TOPIC_SAFETY_OUTPUT_RESTRICTION)
        assert prompt[0]["content"].count(TOPIC_SAFETY_OUTPUT_RESTRICTION) == 1
        assert prompt[1:] == MESSAGES

    def test_multi_turn_messages_included(self, action):
        prompt = action._create_prompt(MODEL_TYPE, {"messages": MULTI_TURN})
        assert len(prompt) == 4
        assert [m["role"] for m in prompt] == ["system", "user", "assistant", "user"]


class TestTopicSafetyParseResponse:
    def test_on_topic(self, action):
        assert action._parse_response("on-topic") == RailResult(is_safe=True)

    def test_off_topic(self, action):
        assert action._parse_response("off-topic") == RailResult(is_safe=False, reason="Topic safety: off-topic")

    @pytest.mark.parametrize("text", ["Off-Topic", "  off-topic  \n", "OFF-TOPIC"])
    def test_off_topic_variants(self, action, text):
        assert not action._parse_response(text).is_safe

    @pytest.mark.parametrize("text", ["on-topic", "something else", ""])
    def test_non_off_topic_is_safe(self, action, text):
        assert action._parse_response(text).is_safe


class TestTopicSafetyRun:
    @pytest.mark.asyncio
    async def test_on_topic(self, action):
        action.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="on-topic"))
        result = await action.run(FLOW, MESSAGES)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_off_topic(self, action):
        action.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="off-topic"))
        result = await action.run(FLOW, MESSAGES)
        assert not result.is_safe

    @pytest.mark.asyncio
    async def test_passes_temperature_and_max_tokens(self, action):
        action.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="on-topic"))
        await action.run(FLOW, MESSAGES)

        call_kwargs = action.engine_registry.model_call.call_args
        assert call_kwargs.kwargs["temperature"] == TOPIC_SAFETY_TEMPERATURE
        assert call_kwargs.kwargs["max_tokens"] == TOPIC_SAFETY_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_system_prompt_contains_guidelines(self, action):
        action.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="on-topic"))
        await action.run(FLOW, MESSAGES)

        call_args = action.engine_registry.model_call.call_args
        llm_messages = call_args[0][1]  # second positional arg
        system_msg = llm_messages[0]
        assert system_msg["role"] == "system"
        assert "customer service agent" in system_msg["content"]

    @pytest.mark.asyncio
    async def test_model_error_returns_unsafe(self, action):
        action.engine_registry.model_call = AsyncMock(side_effect=RuntimeError("timeout"))
        result = await action.run(FLOW, MESSAGES)
        assert not result.is_safe
        assert "timeout" in result.reason


class TestTopicSafetyPromptIsList:
    """Test that a list-type prompt raises."""

    def test_list_prompt_raises(self):
        config = RailsConfig.from_content(
            config={
                **TOPIC_SAFETY_CONFIG,
                "prompts": [
                    {
                        "task": "topic_safety_check_input $model=topic_control",
                        "messages": [{"type": "system", "content": "guidelines"}],
                    },
                ],
            }
        )
        task_manager = LLMTaskManager(config)
        engine_registry = EngineRegistry(config.models, config.rails.config)
        action = TopicSafetyInputAction(engine_registry, task_manager)
        with pytest.raises(RuntimeError, match="must be a string template"):
            action._create_prompt(MODEL_TYPE, {"messages": MESSAGES})


class TestTopicSafetyStopTokens:
    """Test that stop tokens from task config are passed through."""

    @pytest.mark.asyncio
    async def test_passes_stop_tokens(self):
        config = RailsConfig.from_content(
            config={
                **TOPIC_SAFETY_CONFIG,
                "prompts": [
                    {
                        "task": "topic_safety_check_input $model=topic_control",
                        "content": TOPIC_SAFETY_INPUT_PROMPT,
                        "stop": ["</s>"],
                    },
                ],
            }
        )
        task_manager = LLMTaskManager(config)
        engine_registry = EngineRegistry(config.models, config.rails.config)
        action = TopicSafetyInputAction(engine_registry, task_manager)
        action.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="on-topic"))

        await action.run(FLOW, MESSAGES)

        call_kwargs = action.engine_registry.model_call.call_args.kwargs
        assert call_kwargs["stop"] == ["</s>"]


class TestTopicSafetyEndToEnd:
    """Full chain: raw HTTP response dict -> ModelEngine._parse_chat_completion -> LLMResponse
    -> EngineRegistry.model_call -> RailAction._get_llm_response -> .content -> _parse_response
    -> RailResult. Mocks at the HTTP boundary so the dict-parsing path is exercised.
    """

    @pytest.mark.asyncio
    async def test_off_topic_full_chain(self, action):
        engine = action.engine_registry._get_engine(MODEL_TYPE, ModelEngine)
        engine.call = AsyncMock(
            return_value={
                "id": "chatcmpl-1",
                "model": "topic-control-model",
                "choices": [{"message": {"role": "assistant", "content": "off-topic"}, "finish_reason": "stop"}],
            }
        )

        result = await action.run(FLOW, MESSAGES)

        assert result == RailResult(is_safe=False, reason="Topic safety: off-topic")
        engine.call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_topic_full_chain(self, action):
        engine = action.engine_registry._get_engine(MODEL_TYPE, ModelEngine)
        engine.call = AsyncMock(return_value={"choices": [{"message": {"role": "assistant", "content": "on-topic"}}]})

        result = await action.run(FLOW, MESSAGES)

        assert result.is_safe

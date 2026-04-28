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

from nemoguardrails.guardrails.actions.content_safety_action import (
    ContentSafetyInputAction,
    ContentSafetyOutputAction,
    _content_safety_to_rail_result,
)
from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.types import LLMResponse
from tests.guardrails.test_data import CONTENT_SAFETY_CONFIG, CONTENT_SAFETY_INPUT_PROMPT, CONTENT_SAFETY_OUTPUT_PROMPT

FLOW_INPUT = "content safety check input $model=content_safety"
FLOW_OUTPUT = "content safety check output $model=content_safety"
MODEL_TYPE = "content_safety"
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
def engine_registry(config):
    return EngineRegistry(config.models, config.rails.config)


@pytest.fixture
def input_action(engine_registry, task_manager):
    return ContentSafetyInputAction(engine_registry, task_manager)


@pytest.fixture
def output_action(engine_registry, task_manager):
    return ContentSafetyOutputAction(engine_registry, task_manager)


class TestContentSafetyToRailResult:
    """Test the parser output → RailResult converter."""

    def test_safe(self):
        assert _content_safety_to_rail_result([True]) == RailResult(is_safe=True)

    def test_unsafe_with_categories(self):
        assert _content_safety_to_rail_result([False, "S1: Violence", "S17: Malware"]) == RailResult(
            is_safe=False, reason="Safety categories: S1: Violence, S17: Malware"
        )

    def test_unsafe_no_categories(self):
        assert _content_safety_to_rail_result([False]) == RailResult(is_safe=False, reason="Unknown")

    def test_unsafe_single_category(self):
        assert _content_safety_to_rail_result([False, "S17: Malware"]) == RailResult(
            is_safe=False, reason="Safety categories: S17: Malware"
        )

    def test_empty_raises(self):
        with pytest.raises(RuntimeError, match="Unexpected"):
            _content_safety_to_rail_result([])

    def test_invalid_raises(self):
        with pytest.raises(RuntimeError, match="Unexpected"):
            _content_safety_to_rail_result("not a list")


class TestContentSafetyInputExtract:
    """Test _extract_messages on ContentSafetyInputAction."""

    def test_extracts_user_input(self, input_action):
        assert input_action._extract_messages(MESSAGES, None) == {"user_input": "How do I pick a lock?"}


class TestContentSafetyInputPrompt:
    """Test _create_prompt on ContentSafetyInputAction."""

    def test_renders_prompt_with_user_input(self, input_action):
        prompt = input_action._create_prompt(MODEL_TYPE, {"user_input": "test message"})
        assert len(prompt) == 1
        assert prompt[0]["role"] == "user"
        assert "test message" in prompt[0]["content"]
        assert "{{ user_input }}" not in prompt[0]["content"]


class TestContentSafetyOutputExtract:
    """Test _extract_messages on ContentSafetyOutputAction."""

    def test_extracts_user_and_bot(self, output_action):
        assert output_action._extract_messages(MESSAGES, BOT_RESPONSE) == {
            "user_input": "How do I pick a lock?",
            "bot_response": BOT_RESPONSE,
        }


class TestContentSafetyOutputExtractValidation:
    """Test that _extract_messages rejects missing bot_response."""

    def test_missing_bot_response_raises(self, output_action):
        with pytest.raises(RuntimeError, match="bot_response is required"):
            output_action._extract_messages(MESSAGES, None)


class TestContentSafetyEmptyFlow:
    """Test that run() rejects empty flow names."""

    @pytest.mark.asyncio
    async def test_empty_flow_raises(self, input_action):
        with pytest.raises(RuntimeError, match="No flow name found"):
            await input_action.run("", MESSAGES)


class TestRailActionBaseHelpers:
    """Cover base-class utility methods on RailAction."""

    @pytest.mark.asyncio
    async def test_get_local_response_raises(self, input_action):
        with pytest.raises(NotImplementedError, match="Subclass must override"):
            await input_action._get_local_response()

    @pytest.mark.asyncio
    async def test_wrong_flow_name_raises(self, input_action):
        with pytest.raises(RuntimeError, match="does not match expected action_name"):
            await input_action.run("topic safety check input $model=content_safety", MESSAGES)

    @pytest.mark.asyncio
    async def test_llm_response_without_model_raises(self, input_action):
        with pytest.raises(RuntimeError, match="model_type is required for LLM calls"):
            await input_action._get_llm_response(None, [{"role": "user", "content": "hi"}])

    def test_prompt_to_messages_list_branch(self, input_action):
        messages = [{"type": "system", "content": "hello"}, {"type": "user", "content": "world"}]
        result = input_action._prompt_to_messages(messages)
        assert result == [{"role": "system", "content": "hello"}, {"role": "user", "content": "world"}]


class TestContentSafetyMissingModel:
    """Test that run() rejects flows without $model=."""

    @pytest.mark.asyncio
    async def test_input_missing_model_raises(self, input_action):
        with pytest.raises(RuntimeError, match="No \\$model="):
            await input_action.run("content safety check input", MESSAGES)

    @pytest.mark.asyncio
    async def test_output_missing_model_raises(self, output_action):
        with pytest.raises(RuntimeError, match="No \\$model="):
            await output_action.run("content safety check output", MESSAGES, bot_response=BOT_RESPONSE)


class TestContentSafetyInputRun:
    """Test full run() pipeline for ContentSafetyInputAction."""

    @pytest.mark.asyncio
    async def test_safe_input(self, input_action):
        input_action.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content=SAFE_JSON))
        result = await input_action.run(FLOW_INPUT, MESSAGES)
        assert result.is_safe
        input_action.engine_registry.model_call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unsafe_input(self, input_action):
        input_action.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content=UNSAFE_JSON))
        result = await input_action.run(FLOW_INPUT, MESSAGES)
        assert not result.is_safe
        assert "S1: Violence" in result.reason

    @pytest.mark.asyncio
    async def test_model_error_returns_unsafe(self, input_action):
        input_action.engine_registry.model_call = AsyncMock(side_effect=RuntimeError("connection refused"))
        result = await input_action.run(FLOW_INPUT, MESSAGES)
        assert not result.is_safe
        assert "connection refused" in result.reason


class TestContentSafetyOutputRun:
    """Test full run() pipeline for ContentSafetyOutputAction."""

    @pytest.mark.asyncio
    async def test_safe_output(self, output_action):
        output_action.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content=SAFE_OUTPUT_JSON))
        result = await output_action.run(FLOW_OUTPUT, MESSAGES, bot_response=BOT_RESPONSE)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_unsafe_output(self, output_action):
        output_action.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content=UNSAFE_OUTPUT_JSON))
        result = await output_action.run(FLOW_OUTPUT, MESSAGES, bot_response=BOT_RESPONSE)
        assert not result.is_safe
        assert "S17: Malware" in result.reason

    @pytest.mark.asyncio
    async def test_model_error_returns_unsafe(self, output_action):
        output_action.engine_registry.model_call = AsyncMock(side_effect=RuntimeError("timeout"))
        result = await output_action.run(FLOW_OUTPUT, MESSAGES, bot_response=BOT_RESPONSE)
        assert not result.is_safe
        assert "timeout" in result.reason


class TestContentSafetyMissingConfig:
    """Test that missing content_safety config raises."""

    @staticmethod
    def _make_action(action_cls):
        config = RailsConfig.from_content(config=CONTENT_SAFETY_CONFIG)
        config.rails.config.content_safety = None
        return action_cls(EngineRegistry(config.models, config.rails.config), LLMTaskManager(config))

    def test_input_missing_content_safety_config_raises(self):
        action = self._make_action(ContentSafetyInputAction)
        with pytest.raises(RuntimeError, match="content_safety config is required"):
            action._create_prompt(MODEL_TYPE, {"user_input": "test"})

    def test_output_missing_content_safety_config_raises(self):
        action = self._make_action(ContentSafetyOutputAction)
        with pytest.raises(RuntimeError, match="content_safety config is required"):
            action._create_prompt(MODEL_TYPE, {"user_input": "test", "bot_response": "resp"})


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
        engine_registry = EngineRegistry(config.models, config.rails.config)
        action = ContentSafetyInputAction(engine_registry, task_manager)
        action.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content=SAFE_JSON))

        await action.run(FLOW_INPUT, MESSAGES)

        call_kwargs = action.engine_registry.model_call.call_args.kwargs
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
        engine_registry = EngineRegistry(config.models, config.rails.config)
        action = ContentSafetyOutputAction(engine_registry, task_manager)
        action.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content=SAFE_OUTPUT_JSON))

        await action.run(FLOW_OUTPUT, MESSAGES, bot_response=BOT_RESPONSE)

        call_kwargs = action.engine_registry.model_call.call_args.kwargs
        assert call_kwargs["stop"] == ["</s>"]


class TestContentSafetyEndToEnd:
    """Full chain: raw HTTP response dict -> ModelEngine._parse_chat_completion -> LLMResponse
    -> EngineRegistry.model_call -> RailAction._get_llm_response -> .content -> nemoguard parser
    -> _parse_response -> RailResult. Mocks at the HTTP boundary so the dict-parsing path is
    exercised on the way through.
    """

    @pytest.mark.asyncio
    async def test_safe_input_full_chain(self, input_action):
        engine = input_action.engine_registry._get_engine(MODEL_TYPE, ModelEngine)
        engine.call = AsyncMock(
            return_value={
                "id": "chatcmpl-1",
                "model": "content-safety-model",
                "choices": [{"message": {"role": "assistant", "content": SAFE_JSON}, "finish_reason": "stop"}],
            }
        )

        result = await input_action.run(FLOW_INPUT, MESSAGES)

        assert result.is_safe
        engine.call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unsafe_input_full_chain(self, input_action):
        engine = input_action.engine_registry._get_engine(MODEL_TYPE, ModelEngine)
        engine.call = AsyncMock(return_value={"choices": [{"message": {"role": "assistant", "content": UNSAFE_JSON}}]})

        result = await input_action.run(FLOW_INPUT, MESSAGES)

        assert not result.is_safe
        assert "S1: Violence" in result.reason

    @pytest.mark.asyncio
    async def test_unsafe_output_full_chain(self, output_action):
        engine = output_action.engine_registry._get_engine(MODEL_TYPE, ModelEngine)
        engine.call = AsyncMock(
            return_value={"choices": [{"message": {"role": "assistant", "content": UNSAFE_OUTPUT_JSON}}]}
        )

        result = await output_action.run(FLOW_OUTPUT, MESSAGES, bot_response=BOT_RESPONSE)

        assert not result.is_safe
        assert "S17: Malware" in result.reason

    @pytest.mark.asyncio
    async def test_reasoning_field_does_not_affect_classification(self, input_action):
        """Reasoning content is ignored by the content-safety classifier — only `.content` is parsed."""
        engine = input_action.engine_registry._get_engine(MODEL_TYPE, ModelEngine)
        engine.call = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": SAFE_JSON,
                            "reasoning_content": "the prompt looks fine to me",
                        }
                    }
                ]
            }
        )

        result = await input_action.run(FLOW_INPUT, MESSAGES)

        assert result.is_safe

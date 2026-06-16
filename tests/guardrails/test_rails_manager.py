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

"""Unit tests for rails_manager module.

Tests the RailsManager orchestration layer: init, sequential/parallel
execution, and integration with RailAction subclasses via model mocks.
Rail-specific logic (prompt rendering, parsing) is tested in the
individual iorails_actions test files.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.rails_manager import RailsManager
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.types import LLMResponse
from tests.guardrails.test_data import (
    CONTENT_SAFETY_CONFIG,
    NEMOGUARDS_CONFIG,
    NEMOGUARDS_PARALLEL_CONFIG,
    NEMOGUARDS_PARALLEL_INPUT_CONFIG,
    NEMOGUARDS_PARALLEL_OUTPUT_CONFIG,
    TOPIC_SAFETY_CONFIG,
)

SAFE_INPUT_JSON = json.dumps({"User Safety": "safe"})
UNSAFE_INPUT_JSON = json.dumps({"User Safety": "unsafe", "Safety Categories": "S1: Violence"})
SAFE_OUTPUT_JSON = json.dumps({"User Safety": "safe", "Response Safety": "safe"})
UNSAFE_OUTPUT_JSON = json.dumps(
    {
        "User Safety": "safe",
        "Response Safety": "unsafe",
        "Safety Categories": "S17: Malware",
    }
)
MESSAGES = [{"role": "user", "content": "hello"}]


def _make_rails_manager(config: RailsConfig, engine_registry: EngineRegistry | None = None) -> RailsManager:
    """Build a RailsManager from a RailsConfig, extracting the narrow params."""
    if engine_registry is None:
        engine_registry = EngineRegistry(config.models, config.rails.config)
    return RailsManager(
        engine_registry=engine_registry,
        task_manager=LLMTaskManager(config),
        input_flows=config.rails.input.flows,
        output_flows=config.rails.output.flows,
        input_parallel=config.rails.input.parallel or False,
        output_parallel=config.rails.output.parallel or False,
    )


@pytest.fixture
def content_safety_rails_config():
    return RailsConfig.from_content(config=CONTENT_SAFETY_CONFIG)


@pytest.fixture
def content_safety_engine_registry(content_safety_rails_config):
    return EngineRegistry(content_safety_rails_config.models, content_safety_rails_config.rails.config)


@pytest.fixture
def content_safety_rails_manager(content_safety_rails_config, content_safety_engine_registry):
    return _make_rails_manager(content_safety_rails_config, content_safety_engine_registry)


@pytest.fixture
def nemoguards_rails_config():
    return RailsConfig.from_content(config=NEMOGUARDS_CONFIG)


@pytest.fixture
def nemoguards_engine_registry(nemoguards_rails_config):
    return EngineRegistry(nemoguards_rails_config.models, nemoguards_rails_config.rails.config)


@pytest.fixture
def nemoguards_rails_manager(nemoguards_rails_config, nemoguards_engine_registry):
    return _make_rails_manager(nemoguards_rails_config, nemoguards_engine_registry)


@pytest.fixture
def topic_safety_rails_config():
    return RailsConfig.from_content(config=TOPIC_SAFETY_CONFIG)


@pytest.fixture
def topic_safety_engine_registry(topic_safety_rails_config):
    return EngineRegistry(topic_safety_rails_config.models, topic_safety_rails_config.rails.config)


@pytest.fixture
def topic_safety_rails_manager(topic_safety_rails_config, topic_safety_engine_registry):
    return _make_rails_manager(topic_safety_rails_config, topic_safety_engine_registry)


@pytest.fixture
def parallel_input_rails_manager():
    config = RailsConfig.from_content(config=NEMOGUARDS_PARALLEL_INPUT_CONFIG)
    return _make_rails_manager(config)


@pytest.fixture
def parallel_output_rails_manager():
    config = RailsConfig.from_content(config=NEMOGUARDS_PARALLEL_OUTPUT_CONFIG)
    return _make_rails_manager(config)


@pytest.fixture
def parallel_rails_manager():
    config = RailsConfig.from_content(config=NEMOGUARDS_PARALLEL_CONFIG)
    return _make_rails_manager(config)


# --- Init tests ---


class TestRailsManagerInit:
    """Test flows and actions are correctly set up from config."""

    def test_input_flows_populated(self, content_safety_rails_manager):
        assert "content safety check input $model=content_safety" in content_safety_rails_manager.input_flows

    def test_output_flows_populated(self, content_safety_rails_manager):
        assert "content safety check output $model=content_safety" in content_safety_rails_manager.output_flows

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    def test_empty_rails_config(self):
        config = RailsConfig.from_content(config={"models": []})
        mgr = _make_rails_manager(config)
        assert mgr.input_flows == []
        assert mgr.output_flows == []

    def test_unsupported_flow_raises(self):
        config_with_unknown = {
            **CONTENT_SAFETY_CONFIG,
            "rails": {"input": {"flows": ["unknown rail $model=content_safety"]}},
        }
        with pytest.raises(RuntimeError, match="not supported"):
            config = RailsConfig.from_content(config=config_with_unknown)
            _make_rails_manager(config)

    def test_actions_created_for_flows(self, content_safety_rails_manager):
        assert "content safety check input $model=content_safety" in content_safety_rails_manager._actions
        assert "content safety check output $model=content_safety" in content_safety_rails_manager._actions

    def test_nemoguards_actions_created(self, nemoguards_rails_manager):
        assert "content safety check input $model=content_safety" in nemoguards_rails_manager._actions
        assert "content safety check output $model=content_safety" in nemoguards_rails_manager._actions
        assert "topic safety check input $model=topic_control" in nemoguards_rails_manager._actions
        assert "jailbreak detection model" in nemoguards_rails_manager._actions


# --- Sequential input/output tests ---


class TestIsInputSafe:
    """Test is_input_safe with sequential execution."""

    @pytest.mark.asyncio
    async def test_safe(self, content_safety_rails_manager):
        content_safety_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=SAFE_INPUT_JSON)
        )
        result = await content_safety_rails_manager.is_input_safe(MESSAGES)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_unsafe(self, content_safety_rails_manager):
        content_safety_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=UNSAFE_INPUT_JSON)
        )
        result = await content_safety_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe
        assert "Violence" in result.reason

    @pytest.mark.asyncio
    async def test_no_flows_returns_safe(self, content_safety_rails_manager):
        content_safety_rails_manager.input_flows = []
        result = await content_safety_rails_manager.is_input_safe(MESSAGES)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_model_error_returns_unsafe(self, content_safety_rails_manager):
        content_safety_rails_manager.engine_registry.model_call = AsyncMock(side_effect=RuntimeError("timeout"))
        result = await content_safety_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe
        assert "error" in result.reason.lower()


class TestIsOutputSafe:
    """Test is_output_safe with sequential execution."""

    @pytest.mark.asyncio
    async def test_safe(self, content_safety_rails_manager):
        content_safety_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=SAFE_OUTPUT_JSON)
        )
        result = await content_safety_rails_manager.is_output_safe(MESSAGES, "response")
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_unsafe(self, content_safety_rails_manager):
        content_safety_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=UNSAFE_OUTPUT_JSON)
        )
        result = await content_safety_rails_manager.is_output_safe(MESSAGES, "bad response")
        assert not result.is_safe
        assert "S17: Malware" in result.reason

    @pytest.mark.asyncio
    async def test_no_flows_returns_safe(self, content_safety_rails_manager):
        content_safety_rails_manager.output_flows = []
        result = await content_safety_rails_manager.is_output_safe(MESSAGES, "response")
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_model_error_returns_unsafe(self, content_safety_rails_manager):
        content_safety_rails_manager.engine_registry.model_call = AsyncMock(side_effect=RuntimeError("fail"))
        result = await content_safety_rails_manager.is_output_safe(MESSAGES, "response")
        assert not result.is_safe


# --- Multi-rail sequential tests (nemoguards config: content + topic + jailbreak) ---


class TestSequentialMultiRail:
    """Test sequential execution with multiple rails."""

    @pytest.mark.asyncio
    async def test_all_safe(self, nemoguards_rails_manager):
        nemoguards_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=SAFE_INPUT_JSON)
        )
        nemoguards_rails_manager.engine_registry.api_call = AsyncMock(return_value={"jailbreak": False, "score": 0.01})
        result = await nemoguards_rails_manager.is_input_safe(MESSAGES)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_first_rail_blocks(self, nemoguards_rails_manager):
        """Content safety blocks -> topic safety and jailbreak never called."""
        nemoguards_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=UNSAFE_INPUT_JSON)
        )
        nemoguards_rails_manager.engine_registry.api_call = AsyncMock()
        result = await nemoguards_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe
        # Jailbreak API should not have been called (short-circuit)
        nemoguards_rails_manager.engine_registry.api_call.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_jailbreak_blocks(self, nemoguards_rails_manager):
        """Content and topic pass, jailbreak blocks."""
        nemoguards_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=SAFE_INPUT_JSON)
        )
        nemoguards_rails_manager.engine_registry.api_call = AsyncMock(return_value={"jailbreak": True, "score": 0.95})
        result = await nemoguards_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe
        assert "0.95" in result.reason


# --- Topic safety via is_input_safe ---


class TestTopicSafetyIsInputSafe:
    """Test topic safety via the public is_input_safe method."""

    @pytest.mark.asyncio
    async def test_on_topic(self, topic_safety_rails_manager):
        topic_safety_rails_manager.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="on-topic"))
        result = await topic_safety_rails_manager.is_input_safe(MESSAGES)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_off_topic(self, topic_safety_rails_manager):
        topic_safety_rails_manager.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="off-topic"))
        result = await topic_safety_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe
        assert "off-topic" in result.reason

    @pytest.mark.asyncio
    async def test_model_error(self, topic_safety_rails_manager):
        topic_safety_rails_manager.engine_registry.model_call = AsyncMock(side_effect=RuntimeError("timeout"))
        result = await topic_safety_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe


# --- Jailbreak detection via is_input_safe ---


class TestJailbreakDetectionIsInputSafe:
    """Test jailbreak detection via the public is_input_safe method (nemoguards config)."""

    @pytest.mark.asyncio
    async def test_safe(self, nemoguards_rails_manager):
        nemoguards_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=SAFE_INPUT_JSON)
        )
        nemoguards_rails_manager.engine_registry.api_call = AsyncMock(return_value={"jailbreak": False, "score": -0.99})
        result = await nemoguards_rails_manager.is_input_safe(MESSAGES)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_jailbreak_detected(self, nemoguards_rails_manager):
        nemoguards_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=SAFE_INPUT_JSON)
        )
        nemoguards_rails_manager.engine_registry.api_call = AsyncMock(return_value={"jailbreak": True, "score": 0.92})
        result = await nemoguards_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe

    @pytest.mark.asyncio
    async def test_api_error(self, nemoguards_rails_manager):
        nemoguards_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=SAFE_INPUT_JSON)
        )
        nemoguards_rails_manager.engine_registry.api_call = AsyncMock(side_effect=RuntimeError("connection refused"))
        result = await nemoguards_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe


# --- Parallel init ---


class TestParallelInit:
    """Test that parallel flags are correctly stored from config."""

    def test_parallel_false_by_default(self, content_safety_rails_manager):
        assert not content_safety_rails_manager.input_parallel
        assert not content_safety_rails_manager.output_parallel

    def test_parallel_input_true(self, parallel_input_rails_manager):
        assert parallel_input_rails_manager.input_parallel
        assert not parallel_input_rails_manager.output_parallel

    def test_parallel_output_true(self, parallel_output_rails_manager):
        assert not parallel_output_rails_manager.input_parallel
        assert parallel_output_rails_manager.output_parallel

    def test_parallel_both(self, parallel_rails_manager):
        assert parallel_rails_manager.input_parallel
        assert parallel_rails_manager.output_parallel


# --- Parallel input ---


class TestParallelIsInputSafe:
    """Test parallel input rail execution."""

    @pytest.mark.asyncio
    async def test_all_safe(self, parallel_input_rails_manager):
        parallel_input_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=SAFE_INPUT_JSON)
        )
        parallel_input_rails_manager.engine_registry.api_call = AsyncMock(
            return_value={"jailbreak": False, "score": 0.01}
        )
        result = await parallel_input_rails_manager.is_input_safe(MESSAGES)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_one_unsafe(self, parallel_input_rails_manager):
        parallel_input_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=UNSAFE_INPUT_JSON)
        )
        parallel_input_rails_manager.engine_registry.api_call = AsyncMock(
            return_value={"jailbreak": False, "score": 0.01}
        )
        result = await parallel_input_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe

    @pytest.mark.asyncio
    async def test_empty_flows(self, parallel_input_rails_manager):
        parallel_input_rails_manager.input_flows = []
        result = await parallel_input_rails_manager.is_input_safe(MESSAGES)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_model_error(self, parallel_input_rails_manager):
        parallel_input_rails_manager.engine_registry.model_call = AsyncMock(side_effect=RuntimeError("fail"))
        parallel_input_rails_manager.engine_registry.api_call = AsyncMock(
            return_value={"jailbreak": False, "score": 0.01}
        )
        result = await parallel_input_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe


# --- Parallel output ---


class TestParallelIsOutputSafe:
    """Test parallel output rail execution."""

    @pytest.mark.asyncio
    async def test_all_safe(self, parallel_output_rails_manager):
        parallel_output_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=SAFE_OUTPUT_JSON)
        )
        result = await parallel_output_rails_manager.is_output_safe(MESSAGES, "response")
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_one_unsafe(self, parallel_output_rails_manager):
        parallel_output_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=UNSAFE_OUTPUT_JSON)
        )
        result = await parallel_output_rails_manager.is_output_safe(MESSAGES, "bad response")
        assert not result.is_safe

    @pytest.mark.asyncio
    async def test_empty_flows(self, parallel_output_rails_manager):
        parallel_output_rails_manager.output_flows = []
        result = await parallel_output_rails_manager.is_output_safe(MESSAGES, "response")
        assert result.is_safe


# --- Parallel both directions ---


class TestParallelBothDirections:
    """Test with both input and output parallel enabled."""

    @pytest.mark.asyncio
    async def test_both_safe(self, parallel_rails_manager):
        parallel_rails_manager.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content=SAFE_INPUT_JSON))
        parallel_rails_manager.engine_registry.api_call = AsyncMock(return_value={"jailbreak": False, "score": 0.01})
        input_result = await parallel_rails_manager.is_input_safe(MESSAGES)
        assert input_result.is_safe

        parallel_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=SAFE_OUTPUT_JSON)
        )
        output_result = await parallel_rails_manager.is_output_safe(MESSAGES, "response")
        assert output_result.is_safe

    @pytest.mark.asyncio
    async def test_input_unsafe(self, parallel_rails_manager):
        parallel_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=UNSAFE_INPUT_JSON)
        )
        parallel_rails_manager.engine_registry.api_call = AsyncMock(return_value={"jailbreak": False, "score": 0.01})
        result = await parallel_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe

    @pytest.mark.asyncio
    async def test_output_unsafe(self, parallel_rails_manager):
        parallel_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=UNSAFE_OUTPUT_JSON)
        )
        result = await parallel_rails_manager.is_output_safe(MESSAGES, "response")
        assert not result.is_safe

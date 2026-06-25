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

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.guardrails_types import RailDirection, RailResult
from nemoguardrails.guardrails.rails_manager import RailsManager
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.tracing.constants import GuardrailsAttributes
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallFunction
from tests.guardrails.test_data import (
    CONTENT_SAFETY_CONFIG,
    NEMOGUARDS_CONFIG,
    NEMOGUARDS_PARALLEL_CONFIG,
    NEMOGUARDS_PARALLEL_INPUT_CONFIG,
    NEMOGUARDS_PARALLEL_OUTPUT_CONFIG,
    TOPIC_SAFETY_CONFIG,
)
from tests.guardrails.tool_helpers import (
    WEATHER_SCHEMA,
    assert_blocked,
    make_tool_conversation,
    malformed_prior_tool_call_messages,
    multi_turn_reused_call_id_messages,
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


class TestIsInputSafeToggle:
    """The per-request enabled toggle selects which input rails run (bool / list / normalized name)."""

    @pytest.mark.asyncio
    async def test_disabled_toggle_skips_input_rails(self, content_safety_rails_manager):
        """enabled=False runs no input rails, so an otherwise-unsafe input passes."""
        content_safety_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=UNSAFE_INPUT_JSON)
        )
        result = await content_safety_rails_manager.is_input_safe(MESSAGES, enabled=False)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_empty_list_toggle_skips_input_rails(self, content_safety_rails_manager):
        """enabled=[] selects no input rails, so an otherwise-unsafe input passes."""
        content_safety_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=UNSAFE_INPUT_JSON)
        )
        result = await content_safety_rails_manager.is_input_safe(MESSAGES, enabled=[])
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_normalized_name_list_runs_input_rail(self, content_safety_rails_manager):
        """A toggle listing the canonical rail name matches the configured $model=-suffixed flow and runs it."""
        content_safety_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=UNSAFE_INPUT_JSON)
        )
        result = await content_safety_rails_manager.is_input_safe(MESSAGES, enabled=["content safety check input"])
        assert not result.is_safe
        assert "Violence" in result.reason

    @pytest.mark.asyncio
    async def test_true_toggle_runs_input_rails(self, content_safety_rails_manager):
        """enabled=True runs every configured input rail, matching the default."""
        content_safety_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=UNSAFE_INPUT_JSON)
        )
        result = await content_safety_rails_manager.is_input_safe(MESSAGES, enabled=True)
        assert not result.is_safe


class TestIsOutputSafeToggle:
    """The per-request enabled toggle selects which output rails run (bool / list / normalized name)."""

    @pytest.mark.asyncio
    async def test_disabled_toggle_skips_output_rails(self, content_safety_rails_manager):
        """enabled=False runs no output rails, so an otherwise-unsafe response passes."""
        content_safety_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=UNSAFE_OUTPUT_JSON)
        )
        result = await content_safety_rails_manager.is_output_safe(MESSAGES, "bad response", enabled=False)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_normalized_name_list_runs_output_rail(self, content_safety_rails_manager):
        """A toggle listing the canonical rail name matches the configured $model=-suffixed flow and runs it."""
        content_safety_rails_manager.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content=UNSAFE_OUTPUT_JSON)
        )
        result = await content_safety_rails_manager.is_output_safe(
            MESSAGES, "bad response", enabled=["content safety check output"]
        )
        assert not result.is_safe
        assert "S17: Malware" in result.reason


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


def _tool_rails_manager(*, tool_call_flows=None, tool_result_flows=None) -> RailsManager:
    """Build a RailsManager with only tool rails wired (no LLM input/output flows)."""
    config = RailsConfig.from_content(config={"models": []})
    return RailsManager(
        engine_registry=EngineRegistry(config.models, config.rails.config),
        task_manager=LLMTaskManager(config),
        input_flows=[],
        output_flows=[],
        tool_call_flows=tool_call_flows or [],
        tool_result_flows=tool_result_flows or [],
    )


def _call(name: str, arguments: dict) -> ToolCall:
    return ToolCall(id="c1", function=ToolCallFunction(name=name, arguments=arguments))


class TestRailsManagerToolInit:
    def test_tool_flows_populated(self):
        mgr = _tool_rails_manager(
            tool_call_flows=["tool call validation"], tool_result_flows=["tool result validation"]
        )
        assert mgr.tool_call_flows == ["tool call validation"]
        assert mgr.tool_result_flows == ["tool result validation"]

    def test_no_tool_flows_by_default(self):
        mgr = _tool_rails_manager()
        assert mgr.tool_call_flows == []
        assert mgr.tool_result_flows == []

    def test_unknown_tool_flow_raises(self):
        with pytest.raises(RuntimeError, match="not supported"):
            _tool_rails_manager(tool_call_flows=["bogus tool rail"])

    def test_tool_call_flow_with_result_rail_raises(self):
        with pytest.raises(RuntimeError, match="expected ToolCallRailAction"):
            _tool_rails_manager(tool_call_flows=["tool result validation"])

    def test_tool_result_flow_with_call_rail_raises(self):
        with pytest.raises(RuntimeError, match="expected ToolResultRailAction"):
            _tool_rails_manager(tool_result_flows=["tool call validation"])

    def test_duplicate_tool_call_flow_raises(self):
        with pytest.raises(RuntimeError, match="Duplicate tool rail flow"):
            _tool_rails_manager(tool_call_flows=["tool call validation", "tool call validation"])

    def test_duplicate_tool_result_flow_raises(self):
        with pytest.raises(RuntimeError, match="Duplicate tool rail flow"):
            _tool_rails_manager(tool_result_flows=["tool result validation", "tool result validation"])


def _tool_rails_manager_with_main(*, tool_call_flows=None, tool_result_flows=None) -> RailsManager:
    """Like ``_tool_rails_manager`` but with a 'main' engine registered.

    The request-shaped ``are_tool_*_safe`` methods parse tools / extract results via the
    engine adapter, so they need a 'main' engine to delegate to. ``_tool_rails_manager``
    (no engine) is only enough for the disabled / no-flows early-outs that return before
    any engine call."""
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
        config = RailsConfig.from_content(
            config={"models": [{"type": "main", "engine": "nim", "model": "meta/llama-3.3-70b-instruct"}]}
        )
        engine_registry = EngineRegistry(config.models, config.rails.config)
    return RailsManager(
        engine_registry=engine_registry,
        task_manager=LLMTaskManager(config),
        input_flows=[],
        output_flows=[],
        tool_call_flows=tool_call_flows or [],
        tool_result_flows=tool_result_flows or [],
    )


WEATHER_TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "description": "Get weather", "parameters": WEATHER_SCHEMA},
}


class TestRailsManagerToolCalls:
    """The request-shaped ``are_tool_calls_safe``: parse the declared toolset, then validate."""

    @pytest.mark.asyncio
    async def test_no_flows_returns_safe_without_parsing(self):
        # No tool-call rails configured -> safe, and parse is never attempted. The
        # registry here has no engine that could parse, so a safe result proves the
        # early-out happens before any engine call.
        mgr = _tool_rails_manager()
        result = await mgr.are_tool_calls_safe([_call("rm_rf", {})], {"tools": [WEATHER_TOOL]})
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_allows_valid_call(self):
        mgr = _tool_rails_manager_with_main(tool_call_flows=["tool call validation"])
        result = await mgr.are_tool_calls_safe([_call("get_weather", {"city": "Paris"})], {"tools": [WEATHER_TOOL]})
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_blocks_undeclared_call(self):
        mgr = _tool_rails_manager_with_main(tool_call_flows=["tool call validation"])
        result = await mgr.are_tool_calls_safe([_call("rm_rf", {})], {"tools": [WEATHER_TOOL]})
        assert_blocked(result, "rm_rf")

    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_safe(self):
        mgr = _tool_rails_manager_with_main(tool_call_flows=["tool call validation"])
        result = await mgr.are_tool_calls_safe([], {"tools": [WEATHER_TOOL]})
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_fails_closed_on_duplicate_tool_definitions(self):
        # parse_tools raises ValueError on a duplicate tool name; the method must
        # convert that into a block, not propagate.
        mgr = _tool_rails_manager_with_main(tool_call_flows=["tool call validation"])
        result = await mgr.are_tool_calls_safe(
            [_call("get_weather", {"city": "Paris"})], {"tools": [WEATHER_TOOL, WEATHER_TOOL]}
        )
        assert_blocked(result, "tool parsing failed")

    @pytest.mark.asyncio
    async def test_disabled_toggle_skips_validation(self):
        mgr = _tool_rails_manager_with_main(tool_call_flows=["tool call validation"])
        result = await mgr.are_tool_calls_safe([_call("rm_rf", {})], {"tools": [WEATHER_TOOL]}, enabled=False)
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_list_toggle_selects_named_flow(self):
        mgr = _tool_rails_manager_with_main(tool_call_flows=["tool call validation"])
        result = await mgr.are_tool_calls_safe(
            [_call("rm_rf", {})], {"tools": [WEATHER_TOOL]}, enabled=["tool call validation"]
        )
        assert_blocked(result, "rm_rf")


class TestRailsManagerToolResults:
    """The request-shaped ``are_tool_results_safe``: extract results + prior calls, then validate."""

    @pytest.mark.asyncio
    async def test_no_flows_returns_safe_without_extracting(self):
        mgr = _tool_rails_manager()
        result = await mgr.are_tool_results_safe(make_tool_conversation(result_call_id="call_999"))
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_no_tool_results_returns_safe(self):
        mgr = _tool_rails_manager_with_main(tool_result_flows=["tool result validation"])
        result = await mgr.are_tool_results_safe([{"role": "user", "content": "hi"}])
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_allows_linked_result(self):
        mgr = _tool_rails_manager_with_main(tool_result_flows=["tool result validation"])
        result = await mgr.are_tool_results_safe(make_tool_conversation(result_call_id="call_1"))
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_blocks_unlinked_result(self):
        mgr = _tool_rails_manager_with_main(tool_result_flows=["tool result validation"])
        result = await mgr.are_tool_results_safe(make_tool_conversation(result_call_id="call_999"))
        assert_blocked(result, "call_999")

    @pytest.mark.asyncio
    async def test_recycled_call_ids_across_turns_are_safe(self):
        """Reuse the same call ID across turns, but within each turn the call ID is unique"""
        mgr = _tool_rails_manager_with_main(tool_result_flows=["tool result validation"])
        result = await mgr.are_tool_results_safe(multi_turn_reused_call_id_messages())
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_malformed_prior_tool_call_does_not_block_well_formed_results(self):
        """#14 (currently failing): a malformed historical tool-call must not block the request.

        The tool-result rail validates linkage (call_id + name), not the prior call's
        arguments, so a truncated/invalid argument JSON on one turn should not fail
        extraction for the whole conversation. Expected to FAIL until extraction tolerates
        a malformed historical call instead of raising and blocking the whole request.
        """
        mgr = _tool_rails_manager_with_main(tool_result_flows=["tool result validation"])
        result = await mgr.are_tool_results_safe(malformed_prior_tool_call_messages())
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_disabled_toggle_skips_validation(self):
        mgr = _tool_rails_manager_with_main(tool_result_flows=["tool result validation"])
        result = await mgr.are_tool_results_safe(make_tool_conversation(result_call_id="call_999"), enabled=False)
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_fails_closed_when_exchange_extraction_raises(self):
        # If the engine adapter's exchange extraction itself blows up, the method must
        # fail closed (block) rather than let the error escape.
        mgr = _tool_rails_manager_with_main(tool_result_flows=["tool result validation"])

        def _boom(*args, **kwargs):
            raise RuntimeError("extract boom")

        mgr.engine_registry.extract_tool_exchanges = _boom
        result = await mgr.are_tool_results_safe(make_tool_conversation())
        assert_blocked(result, "tool exchange extraction failed")


class TestRailsManagerToolToggleNormalization:
    """#15 (currently failing): a list-valued enable toggle must match configured flows
    by their normalized name, not by the raw flow string.

    A configured tool flow may carry a ``$model=`` or ``(...)`` suffix (accepted by
    config loading, which normalizes via ``_get_flow_name``), while a caller's per-request
    ``enabled`` list naturally carries the canonical rail name. The toggle currently
    compares the raw configured flow string against the requested names, so a suffixed
    configured flow never matches the canonical name, the rail is silently dropped, and
    tool calls/results go unvalidated (fail-open). These assert the rail still runs.
    """

    SUFFIXED_CALL_FLOWS = [
        "tool call validation $model=main",
        "tool call validation(main)",
    ]
    SUFFIXED_RESULT_FLOWS = [
        "tool result validation $model=main",
        "tool result validation(main)",
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("configured_flow", SUFFIXED_CALL_FLOWS)
    async def test_call_toggle_matches_normalized_name(self, configured_flow):
        # Configured flow carries a suffix; the request toggle uses the canonical name.
        # The call rail must still run and block the undeclared call.
        mgr = _tool_rails_manager_with_main(tool_call_flows=[configured_flow])
        result = await mgr.are_tool_calls_safe(
            [_call("rm_rf", {})], {"tools": [WEATHER_TOOL]}, enabled=["tool call validation"]
        )
        assert_blocked(result, "rm_rf")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("configured_flow", SUFFIXED_RESULT_FLOWS)
    async def test_result_toggle_matches_normalized_name(self, configured_flow):
        # Configured flow carries a suffix; the request toggle uses the canonical name.
        # The result rail must still run and block the unlinked result.
        mgr = _tool_rails_manager_with_main(tool_result_flows=[configured_flow])
        result = await mgr.are_tool_results_safe(
            make_tool_conversation(result_call_id="call_999"), enabled=["tool result validation"]
        )
        assert_blocked(result, "call_999")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("configured_flow", SUFFIXED_CALL_FLOWS)
    async def test_call_toggle_raw_flow_string_also_matches(self, configured_flow):
        # Passing the raw configured flow string (including suffix) must also select it,
        # so normalizing the comparison does not break exact-string callers.
        mgr = _tool_rails_manager_with_main(tool_call_flows=[configured_flow])
        result = await mgr.are_tool_calls_safe(
            [_call("rm_rf", {})], {"tools": [WEATHER_TOOL]}, enabled=[configured_flow]
        )
        assert_blocked(result, "rm_rf")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("configured_flow", SUFFIXED_RESULT_FLOWS)
    async def test_result_toggle_raw_flow_string_also_matches(self, configured_flow):
        mgr = _tool_rails_manager_with_main(tool_result_flows=[configured_flow])
        result = await mgr.are_tool_results_safe(
            make_tool_conversation(result_call_id="call_999"), enabled=[configured_flow]
        )
        assert_blocked(result, "call_999")


def _capture_tool_rails_manager():
    """Build (manager, exporter) with a real tracer + content capture on, both tool rails wired.

    Includes a 'main' engine so the request-shaped ``are_tool_*_safe`` can parse / extract."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
        config = RailsConfig.from_content(
            config={"models": [{"type": "main", "engine": "nim", "model": "meta/llama-3.3-70b-instruct"}]}
        )
        engine_registry = EngineRegistry(config.models, config.rails.config)
    manager = RailsManager(
        engine_registry=engine_registry,
        task_manager=LLMTaskManager(config),
        input_flows=[],
        output_flows=[],
        tool_call_flows=["tool call validation"],
        tool_result_flows=["tool result validation"],
        tracer=provider.get_tracer("test"),
        content_capture_enabled=True,
    )
    return manager, exporter


def _rail_span(exporter):
    """The single finished span that carries rail.input (the rail span, not the action span)."""
    spans = [s for s in exporter.get_finished_spans() if GuardrailsAttributes.RAIL_INPUT in s.attributes]
    assert len(spans) == 1
    return spans[0]


_UNLINKED_RESULT_MESSAGES = [
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
    },
    {"role": "tool", "tool_call_id": "c9", "name": "get_weather", "content": "x"},
]


class TestRailsManagerToolContentCapture:
    @pytest.mark.asyncio
    async def test_tool_call_span_captures_calls_and_reason_on_block(self):
        manager, exporter = _capture_tool_rails_manager()
        await manager.are_tool_calls_safe([_call("rm_rf", {})], {"tools": [WEATHER_TOOL]})
        attrs = _rail_span(exporter).attributes
        payload = json.loads(attrs[GuardrailsAttributes.RAIL_INPUT])
        assert payload["tool_calls"][0]["function"]["name"] == "rm_rf"
        assert "rm_rf" in attrs[GuardrailsAttributes.RAIL_REASON]

    @pytest.mark.asyncio
    async def test_tool_call_span_omits_reason_when_safe(self):
        manager, exporter = _capture_tool_rails_manager()
        await manager.are_tool_calls_safe([_call("get_weather", {"city": "Paris"})], {"tools": [WEATHER_TOOL]})
        attrs = _rail_span(exporter).attributes
        assert "tool_calls" in json.loads(attrs[GuardrailsAttributes.RAIL_INPUT])
        assert GuardrailsAttributes.RAIL_REASON not in attrs

    @pytest.mark.asyncio
    async def test_tool_result_span_captures_linkage_and_reason_on_block(self):
        manager, exporter = _capture_tool_rails_manager()
        await manager.are_tool_results_safe(_UNLINKED_RESULT_MESSAGES)
        attrs = _rail_span(exporter).attributes
        payload = json.loads(attrs[GuardrailsAttributes.RAIL_INPUT])
        assert payload["tool_results"][0] == {"call_id": "c9", "name": "get_weather", "is_error": False}
        assert "c9" in attrs[GuardrailsAttributes.RAIL_REASON]


class TestRunRailsParallel:
    """Direct tests for the parallel runner's cancel-on-block and error-cleanup paths.

    These exercise ``_run_rails_parallel`` (used by ``is_input_safe`` / ``is_output_safe``
    when ``parallel`` is enabled) with hand-built coroutines so a rail can stay pending /
    raise deterministically -- the mock-fast rails in the config-driven parallel tests
    above all resolve in the first batch, leaving the cancel/except branches uncovered.
    """

    @pytest.mark.asyncio
    async def test_first_unsafe_result_cancels_pending_rails(self):
        mgr = _tool_rails_manager()
        cancelled = asyncio.Event()

        async def slow_safe():
            try:
                await asyncio.sleep(5)
                return RailResult(is_safe=True)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def fast_unsafe():
            return RailResult(is_safe=False, reason="blocked fast")

        rails = {"slow": slow_safe(), "fast": fast_unsafe()}
        result = await mgr._run_rails_parallel(rails, RailDirection.INPUT)

        assert_blocked(result, "blocked fast")
        assert cancelled.is_set(), "the still-pending rail should have been cancelled"

    @pytest.mark.asyncio
    async def test_rail_exception_cancels_all_and_propagates(self):
        mgr = _tool_rails_manager()
        cancelled = asyncio.Event()

        async def slow_safe():
            try:
                await asyncio.sleep(5)
                return RailResult(is_safe=True)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def raises():
            raise RuntimeError("rail boom")

        rails = {"slow": slow_safe(), "boom": raises()}
        with pytest.raises(RuntimeError, match="rail boom"):
            await mgr._run_rails_parallel(rails, RailDirection.INPUT)

        assert cancelled.is_set(), "remaining rails should be cancelled on a rail error"

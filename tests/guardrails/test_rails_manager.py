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

import asyncio
import copy
import gc
import inspect
import json
import logging
import warnings
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.guardrails.compiled_rail import RailCompilationError, unservable_reason
from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.guardrails_types import RailCallRecord, RailDirection, RailResult, serialize_prompt
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.guardrails.rails_manager import (
    _HTTP_CLIENT_SURFACE_NAMES,
    RailsManager,
    _checked_text,
    _rail_call_record,
    _rail_result,
    _refuse_concurrent_rewrite,
    _result_after_rewrites,
    _rewriting_flows,
    _rewritten_text,
    _tool_rail_result,
    _transforms_first,
)
from nemoguardrails.http.retry import RetryingHTTPClient
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.manifests import RailDirection as SurfaceDirection
from nemoguardrails.manifests import default_rail_catalog, resolve_import_ref
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.tracing.constants import GuardrailsAttributes
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallFunction
from tests.guardrails.async_helpers import (
    mock_jailbreak_nim,
    mock_jailbreak_nim_failure,
    mock_rail_http_response,
    mock_rail_model,
)
from tests.guardrails.rail_stubs import StubRail, declared_rewriter, rails_compiled_as, rewriting_stub
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
    assert_result_blocked,
    make_tool_conversation,
    malformed_prior_tool_call_messages,
    multi_turn_reused_call_id_messages,
)


def _coroutine_returning(result: RailResult):
    """A coroutine handing back *result*, standing in for one rail's pending run."""

    async def run() -> RailResult:
        return result

    return run()


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
        engine_registry = EngineRegistry(config.models)
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
    return EngineRegistry(content_safety_rails_config.models)


@pytest.fixture
def content_safety_rails_manager(content_safety_rails_config, content_safety_engine_registry):
    return _make_rails_manager(content_safety_rails_config, content_safety_engine_registry)


@pytest.fixture
def nemoguards_rails_config():
    return RailsConfig.from_content(config=NEMOGUARDS_CONFIG)


@pytest.fixture
def nemoguards_engine_registry(nemoguards_rails_config):
    return EngineRegistry(nemoguards_rails_config.models)


@pytest.fixture
def nemoguards_rails_manager(nemoguards_rails_config, nemoguards_engine_registry):
    return _make_rails_manager(nemoguards_rails_config, nemoguards_engine_registry)


@pytest.fixture
def topic_safety_rails_config():
    return RailsConfig.from_content(config=TOPIC_SAFETY_CONFIG)


@pytest.fixture
def topic_safety_engine_registry(topic_safety_rails_config):
    return EngineRegistry(topic_safety_rails_config.models)


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
        with pytest.raises(RailCompilationError, match="no surface named"):
            config = RailsConfig.from_content(config=config_with_unknown)
            _make_rails_manager(config)

    def test_unparseable_flow_raises(self, content_safety_rails_config, content_safety_engine_registry):
        """A flow the surface parser rejects fails compilation instead of raising a raw ValueError."""
        # The HTTP-client lookup parses each flow before compiling it, so an unparseable flow
        # must fall through to compile_rail rather than escaping as the parser's ValueError.
        with pytest.raises(RailCompilationError, match="not a valid flow reference"):
            RailsManager(
                engine_registry=content_safety_engine_registry,
                task_manager=LLMTaskManager(content_safety_rails_config),
                input_flows=["$model=content_safety"],
                output_flows=[],
            )

    def test_rails_compiled_for_flows(self, content_safety_rails_manager):
        assert set(content_safety_rails_manager._rails) == {
            (RailDirection.INPUT, "content safety check input $model=content_safety"),
            (RailDirection.OUTPUT, "content safety check output $model=content_safety"),
        }

    def test_nemoguards_rails_compiled(self, nemoguards_rails_manager):
        assert set(nemoguards_rails_manager._rails) == {
            (RailDirection.INPUT, "content safety check input $model=content_safety"),
            (RailDirection.INPUT, "topic safety check input $model=topic_control"),
            (RailDirection.INPUT, "jailbreak detection model"),
            (RailDirection.OUTPUT, "content safety check output $model=content_safety"),
        }

    @pytest.mark.asyncio
    async def test_a_rail_runs_the_compilation_for_its_own_direction(self, content_safety_rails_manager):
        """Dispatch keys on direction, so one flow name cannot resolve to the other direction."""
        # No catalog surface is offered in both directions today, so this pins the lookup
        # rather than a reachable collision.
        flow = "content safety check input $model=content_safety"
        content_safety_rails_manager._rails[(RailDirection.INPUT, flow)] = StubRail(RailOutcome.allow())
        content_safety_rails_manager._rails[(RailDirection.OUTPUT, flow)] = StubRail(RailOutcome.block())

        allowed = await content_safety_rails_manager._run_rail(flow, RailDirection.INPUT, MESSAGES)
        blocked = await content_safety_rails_manager._run_rail(flow, RailDirection.OUTPUT, MESSAGES)

        assert allowed.is_safe is True
        assert blocked.is_safe is False


class _RecordingClient:
    """Minimal ``ClosableHTTPClient`` stand-in that records ``close()`` calls."""

    def __init__(self, close_error: Exception | None = None) -> None:
        self.close_count = 0
        self.close_error = close_error

    async def request(self, method, url, **kwargs):
        raise NotImplementedError("the recording client does not send requests")

    async def close(self) -> None:
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


def _give_every_rail_a_client(manager: RailsManager) -> list[_RecordingClient]:
    """Swap in a recording client for every compiled rail, returned in dispatch order."""
    clients = [_RecordingClient() for _ in manager._rails]
    for rail, client in zip(manager._rails.values(), clients):
        rail._http_client = client
    return clients


class TestRailsManagerStop:
    """Only API-backed rails own an HTTP client, and stop() releases every one of them."""

    def test_only_api_backed_rails_are_given_a_client(self, nemoguards_rails_manager):
        """A vendor API surface compiles with its own client; an LLM-backed surface gets none."""
        clients = {flow: rail._http_client for (_, flow), rail in nemoguards_rails_manager._rails.items()}

        assert clients["jailbreak detection model"] is not None
        assert clients["content safety check input $model=content_safety"] is None

    def test_the_injected_client_carries_no_retry_policy(self, nemoguards_rails_manager):
        """The injected client pools connections only; retry stays the vendor action's to apply."""
        # Vendor actions wrap whatever client they are handed -- clavata and f5 build a
        # RetryingHTTPClient around it -- so a retrying client here would nest inside theirs
        # and multiply attempts against an API that is already rate-limiting us.
        client = nemoguards_rails_manager._rails[(RailDirection.INPUT, "jailbreak detection model")]._http_client

        assert not isinstance(client, RetryingHTTPClient)

    def test_each_api_backed_rail_gets_its_own_client(self):
        """Two API-backed flows get separate clients, so one vendor's pool cannot starve another."""
        config_dict = {
            **NEMOGUARDS_CONFIG,
            "rails": {
                **NEMOGUARDS_CONFIG["rails"],
                # Two API-backed input rails. Not the two jailbreak flows: heuristics shares a
                # config section with the model rail and is refused at compile time for it.
                "input": {"flows": ["jailbreak detection model", "activefence moderation on input"]},
                "output": {"flows": []},
            },
        }
        manager = _make_rails_manager(RailsConfig.from_content(config=config_dict))

        clients = [rail._http_client for rail in manager._rails.values()]

        assert len(clients) == 2
        assert clients[0] is not clients[1]

    @pytest.mark.asyncio
    async def test_stop_closes_every_client(self, nemoguards_rails_manager):
        """Shutdown releases each rail's connection pool."""
        clients = _give_every_rail_a_client(nemoguards_rails_manager)

        await nemoguards_rails_manager.stop()

        assert [client.close_count for client in clients] == [1] * len(clients)

    @pytest.mark.asyncio
    async def test_a_failing_close_still_releases_the_remaining_clients(self, nemoguards_rails_manager):
        """One client raising does not leak the pools behind it in the dispatch order."""
        clients = _give_every_rail_a_client(nemoguards_rails_manager)
        clients[0].close_error = RuntimeError("Event loop is closed")

        with pytest.raises(RuntimeError, match="Failed to close rail HTTP clients"):
            await nemoguards_rails_manager.stop()

        assert [client.close_count for client in clients] == [1] * len(clients)

    @pytest.mark.asyncio
    async def test_stop_names_the_rail_whose_client_failed(self, nemoguards_rails_manager):
        """The combined error names the flow, so a leaked pool is attributable."""
        clients = _give_every_rail_a_client(nemoguards_rails_manager)
        clients[0].close_error = RuntimeError("socket stuck")

        with pytest.raises(RuntimeError, match="socket stuck") as excinfo:
            await nemoguards_rails_manager.stop()

        failed_flow = list(nemoguards_rails_manager._rails)[0][1]
        assert failed_flow in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_second_stop_is_safe(self, nemoguards_rails_manager):
        """stop() is idempotent, because a closed client's close() is a no-op."""
        await nemoguards_rails_manager.stop()
        await nemoguards_rails_manager.stop()


class TestPooledClientCoverage:
    """Every rail IORails enables that speaks HTTP is named in the pooling list."""

    # _HTTP_CLIENT_SURFACE_NAMES is hand-maintained while the set of rails IORails runs is
    # derived from the catalog, so nothing holds the two in agreement. Enabling a vendor rail
    # without adding it here regresses silently rather than loudly: http_call builds an owned
    # client per request when it receives None, so the rail still returns the right verdict
    # and only the connection pooling is lost. The pooling list may name surfaces IORails has
    # not enabled -- it is deliberately ahead of the tier -- so only this direction is checked.

    def test_every_enabled_http_surface_is_pooled(self):
        """A rail in the enabled tier whose action declares http_client is compiled with one."""
        declares_http_client = {
            name
            for (direction, name), surface in default_rail_catalog().surfaces().items()
            if direction is not SurfaceDirection.RETRIEVAL
            and unservable_reason(name, direction) is None
            and "http_client" in inspect.signature(resolve_import_ref(surface.action)).parameters
        }

        unpooled = sorted(declares_http_client - _HTTP_CLIENT_SURFACE_NAMES)

        assert declares_http_client, "no enabled surface declares http_client; this test is broken, not the tier"
        assert not unpooled, f"enabled rails absent from _HTTP_CLIENT_SURFACE_NAMES: {unpooled}"


# --- Sequential input/output tests ---


class TestIsInputSafe:
    """Test is_input_safe with sequential execution."""

    @pytest.mark.asyncio
    async def test_safe(self, content_safety_rails_manager):
        mock_rail_model(
            content_safety_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=SAFE_INPUT_JSON))
        )
        result = await content_safety_rails_manager.is_input_safe(MESSAGES)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_unsafe(self, content_safety_rails_manager):
        mock_rail_model(
            content_safety_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=UNSAFE_INPUT_JSON))
        )
        result = await content_safety_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe
        assert result.return_value["policy_violations"] == ["S1: Violence"]

    @pytest.mark.asyncio
    async def test_no_flows_returns_safe(self, content_safety_rails_manager):
        content_safety_rails_manager.input_flows = []
        result = await content_safety_rails_manager.is_input_safe(MESSAGES)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_model_error_returns_unsafe(self, content_safety_rails_manager):
        mock_rail_model(content_safety_rails_manager.engine_registry, AsyncMock(side_effect=RuntimeError("timeout")))
        result = await content_safety_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe
        assert "error" in result.reason.lower()


class TestIsOutputSafe:
    """Test is_output_safe with sequential execution."""

    @pytest.mark.asyncio
    async def test_safe(self, content_safety_rails_manager):
        mock_rail_model(
            content_safety_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=SAFE_OUTPUT_JSON))
        )
        result = await content_safety_rails_manager.is_output_safe(MESSAGES, "response")
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_unsafe(self, content_safety_rails_manager):
        mock_rail_model(
            content_safety_rails_manager.engine_registry,
            AsyncMock(return_value=LLMResponse(content=UNSAFE_OUTPUT_JSON)),
        )
        result = await content_safety_rails_manager.is_output_safe(MESSAGES, "bad response")
        assert not result.is_safe
        assert result.return_value["policy_violations"] == ["S17: Malware"]

    @pytest.mark.asyncio
    async def test_no_flows_returns_safe(self, content_safety_rails_manager):
        content_safety_rails_manager.output_flows = []
        result = await content_safety_rails_manager.is_output_safe(MESSAGES, "response")
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_model_error_returns_unsafe(self, content_safety_rails_manager):
        mock_rail_model(content_safety_rails_manager.engine_registry, AsyncMock(side_effect=RuntimeError("fail")))
        result = await content_safety_rails_manager.is_output_safe(MESSAGES, "response")
        assert not result.is_safe


class TestIsInputSafeToggle:
    """The per-request enabled toggle selects which input rails run (bool / list / normalized name)."""

    @pytest.mark.asyncio
    async def test_disabled_toggle_skips_input_rails(self, content_safety_rails_manager):
        """enabled=False runs no input rails, so an otherwise-unsafe input passes."""
        mock_rail_model(
            content_safety_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=UNSAFE_INPUT_JSON))
        )
        result = await content_safety_rails_manager.is_input_safe(MESSAGES, enabled=False)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_empty_list_toggle_skips_input_rails(self, content_safety_rails_manager):
        """enabled=[] selects no input rails, so an otherwise-unsafe input passes."""
        mock_rail_model(
            content_safety_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=UNSAFE_INPUT_JSON))
        )
        result = await content_safety_rails_manager.is_input_safe(MESSAGES, enabled=[])
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_normalized_name_list_runs_input_rail(self, content_safety_rails_manager):
        """A toggle listing the canonical rail name matches the configured $model=-suffixed flow and runs it."""
        mock_rail_model(
            content_safety_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=UNSAFE_INPUT_JSON))
        )
        result = await content_safety_rails_manager.is_input_safe(MESSAGES, enabled=["content safety check input"])
        assert not result.is_safe
        assert result.return_value["policy_violations"] == ["S1: Violence"]

    @pytest.mark.asyncio
    async def test_true_toggle_runs_input_rails(self, content_safety_rails_manager):
        """enabled=True runs every configured input rail, matching the default."""
        mock_rail_model(
            content_safety_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=UNSAFE_INPUT_JSON))
        )
        result = await content_safety_rails_manager.is_input_safe(MESSAGES, enabled=True)
        assert not result.is_safe


class TestIsOutputSafeToggle:
    """The per-request enabled toggle selects which output rails run (bool / list / normalized name)."""

    @pytest.mark.asyncio
    async def test_disabled_toggle_skips_output_rails(self, content_safety_rails_manager):
        """enabled=False runs no output rails, so an otherwise-unsafe response passes."""
        mock_rail_model(
            content_safety_rails_manager.engine_registry,
            AsyncMock(return_value=LLMResponse(content=UNSAFE_OUTPUT_JSON)),
        )
        result = await content_safety_rails_manager.is_output_safe(MESSAGES, "bad response", enabled=False)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_normalized_name_list_runs_output_rail(self, content_safety_rails_manager):
        """A toggle listing the canonical rail name matches the configured $model=-suffixed flow and runs it."""
        mock_rail_model(
            content_safety_rails_manager.engine_registry,
            AsyncMock(return_value=LLMResponse(content=UNSAFE_OUTPUT_JSON)),
        )
        result = await content_safety_rails_manager.is_output_safe(
            MESSAGES, "bad response", enabled=["content safety check output"]
        )
        assert not result.is_safe
        assert result.return_value["policy_violations"] == ["S17: Malware"]


# --- Multi-rail sequential tests (nemoguards config: content + topic + jailbreak) ---


class TestSequentialMultiRail:
    """Test sequential execution with multiple rails."""

    @pytest.mark.asyncio
    async def test_all_safe(self, nemoguards_rails_manager, httpx_mock):
        mock_rail_model(
            nemoguards_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=SAFE_INPUT_JSON))
        )
        mock_jailbreak_nim(httpx_mock, jailbreak=False, score=0.01)
        result = await nemoguards_rails_manager.is_input_safe(MESSAGES)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_first_rail_blocks(self, nemoguards_rails_manager, httpx_mock):
        """Content safety blocks -> topic safety and jailbreak never called."""
        mock_rail_model(
            nemoguards_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=UNSAFE_INPUT_JSON))
        )
        result = await nemoguards_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe
        # Jailbreak API should not have been called (short-circuit)
        assert not httpx_mock.get_requests()

    @pytest.mark.asyncio
    async def test_jailbreak_blocks(self, nemoguards_rails_manager, httpx_mock):
        """Content and topic pass, jailbreak blocks."""
        mock_rail_model(
            nemoguards_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=SAFE_INPUT_JSON))
        )
        mock_jailbreak_nim(httpx_mock, jailbreak=True, score=0.95)
        result = await nemoguards_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe
        assert result.return_value == {"allowed": False}


# --- Topic safety via is_input_safe ---


class TestTopicSafetyIsInputSafe:
    """Test topic safety via the public is_input_safe method."""

    @pytest.mark.asyncio
    async def test_on_topic(self, topic_safety_rails_manager):
        mock_rail_model(
            topic_safety_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content="on-topic"))
        )
        result = await topic_safety_rails_manager.is_input_safe(MESSAGES)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_off_topic(self, topic_safety_rails_manager):
        mock_rail_model(
            topic_safety_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content="off-topic"))
        )
        result = await topic_safety_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe
        assert result.return_value == {"allowed": False}

    @pytest.mark.asyncio
    async def test_model_error(self, topic_safety_rails_manager):
        mock_rail_model(topic_safety_rails_manager.engine_registry, AsyncMock(side_effect=RuntimeError("timeout")))
        result = await topic_safety_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe


# --- Jailbreak detection via is_input_safe ---


class TestJailbreakDetectionIsInputSafe:
    """Test jailbreak detection via the public is_input_safe method (nemoguards config)."""

    @pytest.mark.asyncio
    async def test_safe(self, nemoguards_rails_manager, httpx_mock):
        mock_rail_model(
            nemoguards_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=SAFE_INPUT_JSON))
        )
        mock_jailbreak_nim(httpx_mock, jailbreak=False, score=-0.99)
        result = await nemoguards_rails_manager.is_input_safe(MESSAGES)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_jailbreak_detected(self, nemoguards_rails_manager, httpx_mock):
        mock_rail_model(
            nemoguards_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=SAFE_INPUT_JSON))
        )
        mock_jailbreak_nim(httpx_mock, jailbreak=True, score=0.92)
        result = await nemoguards_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe

    @pytest.mark.asyncio
    async def test_endpoint_error_allows(self, nemoguards_rails_manager, httpx_mock):
        """An unreachable NIM allows the request; the library action swallows every failure."""
        mock_rail_model(
            nemoguards_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=SAFE_INPUT_JSON))
        )
        mock_jailbreak_nim_failure(httpx_mock)
        result = await nemoguards_rails_manager.is_input_safe(MESSAGES)
        assert result.is_safe


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
    async def test_all_safe(self, parallel_input_rails_manager, httpx_mock):
        mock_rail_model(
            parallel_input_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=SAFE_INPUT_JSON))
        )
        mock_jailbreak_nim(httpx_mock, jailbreak=False, score=0.01)
        result = await parallel_input_rails_manager.is_input_safe(MESSAGES)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_one_unsafe(self, parallel_input_rails_manager, httpx_mock):
        mock_rail_model(
            parallel_input_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=UNSAFE_INPUT_JSON))
        )
        mock_jailbreak_nim(httpx_mock, jailbreak=False, score=0.01)
        result = await parallel_input_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe

    @pytest.mark.asyncio
    async def test_empty_flows(self, parallel_input_rails_manager):
        parallel_input_rails_manager.input_flows = []
        result = await parallel_input_rails_manager.is_input_safe(MESSAGES)
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_model_error(self, parallel_input_rails_manager, httpx_mock):
        mock_rail_model(parallel_input_rails_manager.engine_registry, AsyncMock(side_effect=RuntimeError("fail")))
        mock_jailbreak_nim(httpx_mock, jailbreak=False, score=0.01)
        result = await parallel_input_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe


# --- Parallel output ---


class TestParallelIsOutputSafe:
    """Test parallel output rail execution."""

    @pytest.mark.asyncio
    async def test_all_safe(self, parallel_output_rails_manager):
        mock_rail_model(
            parallel_output_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=SAFE_OUTPUT_JSON))
        )
        result = await parallel_output_rails_manager.is_output_safe(MESSAGES, "response")
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_one_unsafe(self, parallel_output_rails_manager):
        mock_rail_model(
            parallel_output_rails_manager.engine_registry,
            AsyncMock(return_value=LLMResponse(content=UNSAFE_OUTPUT_JSON)),
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
    async def test_both_safe(self, parallel_rails_manager, httpx_mock):
        mock_rail_model(
            parallel_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=SAFE_INPUT_JSON))
        )
        mock_jailbreak_nim(httpx_mock, jailbreak=False, score=0.01)
        input_result = await parallel_rails_manager.is_input_safe(MESSAGES)
        assert input_result.is_safe

        mock_rail_model(
            parallel_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=SAFE_OUTPUT_JSON))
        )
        output_result = await parallel_rails_manager.is_output_safe(MESSAGES, "response")
        assert output_result.is_safe

    @pytest.mark.asyncio
    async def test_input_unsafe(self, parallel_rails_manager, httpx_mock):
        mock_rail_model(
            parallel_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=UNSAFE_INPUT_JSON))
        )
        mock_jailbreak_nim(httpx_mock, jailbreak=False, score=0.01)
        result = await parallel_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe

    @pytest.mark.asyncio
    async def test_output_unsafe(self, parallel_rails_manager):
        mock_rail_model(
            parallel_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=UNSAFE_OUTPUT_JSON))
        )
        result = await parallel_rails_manager.is_output_safe(MESSAGES, "response")
        assert not result.is_safe


def _tool_rails_manager(*, tool_call_flows=None, tool_result_flows=None) -> RailsManager:
    """Build a RailsManager with only tool rails wired (no LLM input/output flows)."""
    config = RailsConfig.from_content(config={"models": []})
    return RailsManager(
        engine_registry=EngineRegistry(config.models),
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
        engine_registry = EngineRegistry(config.models)
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
        assert_result_blocked(result, "rm_rf")

    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_safe(self):
        mgr = _tool_rails_manager_with_main(tool_call_flows=["tool call validation"])
        result = await mgr.are_tool_calls_safe([], {"tools": [WEATHER_TOOL]})
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_a_short_circuit_leaves_no_unawaited_coroutine(self):
        """Tool rails are built up front, so the ones a block skips are closed rather than abandoned."""
        mgr = _tool_rails_manager_with_main(tool_call_flows=["tool call validation"])
        rails = {
            "blocks": _coroutine_returning(RailResult.block(reason="unsafe")),
            "never reached": _coroutine_returning(RailResult.allow()),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = await mgr._run_tool_rails_sequential(rails, RailDirection.OUTPUT)
            gc.collect()

        assert result.is_safe is False

    @pytest.mark.asyncio
    async def test_fails_closed_on_duplicate_tool_definitions(self):
        # parse_tools raises ValueError on a duplicate tool name; the method must
        # convert that into a block, not propagate.
        mgr = _tool_rails_manager_with_main(tool_call_flows=["tool call validation"])
        result = await mgr.are_tool_calls_safe(
            [_call("get_weather", {"city": "Paris"})], {"tools": [WEATHER_TOOL, WEATHER_TOOL]}
        )
        assert_result_blocked(result, "tool parsing failed")

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
        assert_result_blocked(result, "rm_rf")


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
        assert_result_blocked(result, "call_999")

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
        assert_result_blocked(result, "tool exchange extraction failed")


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
        assert_result_blocked(result, "rm_rf")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("configured_flow", SUFFIXED_RESULT_FLOWS)
    async def test_result_toggle_matches_normalized_name(self, configured_flow):
        # Configured flow carries a suffix; the request toggle uses the canonical name.
        # The result rail must still run and block the unlinked result.
        mgr = _tool_rails_manager_with_main(tool_result_flows=[configured_flow])
        result = await mgr.are_tool_results_safe(
            make_tool_conversation(result_call_id="call_999"), enabled=["tool result validation"]
        )
        assert_result_blocked(result, "call_999")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("configured_flow", SUFFIXED_CALL_FLOWS)
    async def test_call_toggle_raw_flow_string_also_matches(self, configured_flow):
        # Passing the raw configured flow string (including suffix) must also select it,
        # so normalizing the comparison does not break exact-string callers.
        mgr = _tool_rails_manager_with_main(tool_call_flows=[configured_flow])
        result = await mgr.are_tool_calls_safe(
            [_call("rm_rf", {})], {"tools": [WEATHER_TOOL]}, enabled=[configured_flow]
        )
        assert_result_blocked(result, "rm_rf")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("configured_flow", SUFFIXED_RESULT_FLOWS)
    async def test_result_toggle_raw_flow_string_also_matches(self, configured_flow):
        mgr = _tool_rails_manager_with_main(tool_result_flows=[configured_flow])
        result = await mgr.are_tool_results_safe(
            make_tool_conversation(result_call_id="call_999"), enabled=[configured_flow]
        )
        assert_result_blocked(result, "call_999")


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
        engine_registry = EngineRegistry(config.models)
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
                return RailResult.allow()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def fast_unsafe():
            return RailResult.block(reason="blocked fast")

        rails = {"slow": slow_safe(), "fast": fast_unsafe()}
        result = await mgr._run_rails_parallel(rails, RailDirection.INPUT)

        assert_result_blocked(result, "blocked fast")
        assert cancelled.is_set(), "the still-pending rail should have been cancelled"

    @pytest.mark.asyncio
    async def test_rail_exception_cancels_all_and_propagates(self):
        mgr = _tool_rails_manager()
        cancelled = asyncio.Event()

        async def slow_safe():
            try:
                await asyncio.sleep(5)
                return RailResult.allow()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def raises():
            raise RuntimeError("rail boom")

        rails = {"slow": slow_safe(), "boom": raises()}
        with pytest.raises(RuntimeError, match="rail boom"):
            await mgr._run_rails_parallel(rails, RailDirection.INPUT)

        assert cancelled.is_set(), "remaining rails should be cancelled on a rail error"


class TestTriggeredRail:
    """A blocking rail records its base flow name in RailResult.triggered_rail."""

    @pytest.mark.asyncio
    async def test_input_block_sets_triggered_rail(self, content_safety_rails_manager):
        """An input-rail block records the flow's base name in triggered_rail."""
        mock_rail_model(
            content_safety_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=UNSAFE_INPUT_JSON))
        )
        result = await content_safety_rails_manager.is_input_safe(MESSAGES)
        assert not result.is_safe
        assert result.triggered_rail == "content safety check input"

    @pytest.mark.asyncio
    async def test_output_block_sets_triggered_rail(self, content_safety_rails_manager):
        """An output-rail block records the flow's base name in triggered_rail."""
        mock_rail_model(
            content_safety_rails_manager.engine_registry,
            AsyncMock(return_value=LLMResponse(content=UNSAFE_OUTPUT_JSON)),
        )
        result = await content_safety_rails_manager.is_output_safe(MESSAGES, "response")
        assert not result.is_safe
        assert result.triggered_rail == "content safety check output"

    @pytest.mark.asyncio
    async def test_safe_result_has_no_triggered_rail(self, content_safety_rails_manager):
        """A safe result leaves triggered_rail unset (None)."""
        mock_rail_model(
            content_safety_rails_manager.engine_registry, AsyncMock(return_value=LLMResponse(content=SAFE_INPUT_JSON))
        )
        result = await content_safety_rails_manager.is_input_safe(MESSAGES)
        assert result.is_safe
        assert result.triggered_rail is None


class TestOutcomeToResult:
    """`_rail_result` maps a rail's verdict onto IORails' result type."""

    @pytest.mark.parametrize(
        "outcome, is_safe",
        [(RailOutcome.allow(), True), (RailOutcome.block(), False)],
        ids=["allow", "block"],
    )
    def test_a_decision_becomes_a_verdict(self, outcome, is_safe):
        """Allow and block map onto is_safe, with the decision echoed in the verdict payload."""
        result = _rail_result(outcome)

        assert result.is_safe is is_safe
        assert result.return_value == {"allowed": is_safe}

    def test_a_transform_becomes_a_safe_verdict_carrying_its_rewrite(self):
        """A rewrite does not block, and the text it produced survives on the outcome."""
        outcome = RailOutcome.transform([(TransformTarget.USER_MESSAGE, "masked")])

        result = _rail_result(outcome)

        assert result.is_safe is True
        assert result.outcome.transform_text == {"user_message": "masked"}

    def test_a_tool_rail_returning_a_rewrite_raises(self):
        """Tool rails declare no variable to rewrite, so one arriving fails loudly instead of vanishing."""
        outcome = RailOutcome.transform([(TransformTarget.USER_MESSAGE, "masked")])

        with pytest.raises(NotImplementedError, match="tool call validation"):
            _tool_rail_result(outcome, "tool call validation")

    @pytest.mark.parametrize(
        "outcome, is_safe",
        [(RailOutcome.allow(), True), (RailOutcome.block(), False)],
        ids=["allow", "block"],
    )
    def test_a_tool_rail_decision_becomes_a_verdict(self, outcome, is_safe):
        """The decisions a tool rail can reach pass through unchanged."""
        assert _tool_rail_result(outcome, "tool call validation").is_safe is is_safe


CONTENT_SAFETY_INPUT_FLOW = "content safety check input $model=content_safety"
TOPIC_SAFETY_INPUT_FLOW = "topic safety check input $model=topic_control"
CONTENT_SAFETY_OUTPUT_FLOW = "content safety check output $model=content_safety"
# Names without their $model= suffix, which is the form the per-request toggle matches on.
INPUT_PAIR = ["content safety check input", "topic safety check input"]

SSN_MESSAGES = [{"role": "system", "content": "be helpful"}, {"role": "user", "content": "my ssn is 123-45-6789"}]
MASKED = "my ssn is <SSN>"


def _mask_user_message(text: str) -> RailOutcome:
    return RailOutcome.transform([(TransformTarget.USER_MESSAGE, text)])


def _mask_bot_message(text: str) -> RailOutcome:
    return RailOutcome.transform([(TransformTarget.BOT_MESSAGE, text)])


@pytest.mark.asyncio
class TestRailsThatRewrite:
    """A rail's rewrite reaches the rails behind it and is reported as the direction's verdict."""

    def _install(self, manager, direction, rails: dict) -> None:
        for flow, rail in rails.items():
            manager._rails[(direction, flow)] = rail

    async def test_a_rewrite_reaches_the_next_rail(self, nemoguards_rails_manager):
        """The rail behind a masking rail checks the masked text, which is the point of rewriting."""
        recorder = StubRail(RailOutcome.allow())
        self._install(
            nemoguards_rails_manager,
            RailDirection.INPUT,
            {CONTENT_SAFETY_INPUT_FLOW: StubRail(_mask_user_message(MASKED)), TOPIC_SAFETY_INPUT_FLOW: recorder},
        )

        await nemoguards_rails_manager.is_input_safe(SSN_MESSAGES, enabled=INPUT_PAIR)

        assert recorder.seen_messages == [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": MASKED},
        ]

    async def test_the_verdict_carries_the_text_the_last_rail_left(self, nemoguards_rails_manager):
        """Two rewrites compose, and the surviving text is what the caller is told to use."""
        self._install(
            nemoguards_rails_manager,
            RailDirection.INPUT,
            {
                CONTENT_SAFETY_INPUT_FLOW: StubRail(_mask_user_message(MASKED)),
                TOPIC_SAFETY_INPUT_FLOW: StubRail(_mask_user_message("my ssn is [redacted]")),
            },
        )

        result = await nemoguards_rails_manager.is_input_safe(SSN_MESSAGES, enabled=INPUT_PAIR)

        assert result.is_safe is True
        assert result.outcome.transform_text == {"user_message": "my ssn is [redacted]"}

    async def test_a_rewrite_back_to_the_original_reports_nothing_happened(self, nemoguards_rails_manager):
        """Only a net change counts, so ``check`` can tell PASSED from MODIFIED by comparing content."""
        self._install(
            nemoguards_rails_manager,
            RailDirection.INPUT,
            {
                CONTENT_SAFETY_INPUT_FLOW: StubRail(_mask_user_message(MASKED)),
                TOPIC_SAFETY_INPUT_FLOW: StubRail(_mask_user_message("my ssn is 123-45-6789")),
            },
        )

        result = await nemoguards_rails_manager.is_input_safe(SSN_MESSAGES, enabled=INPUT_PAIR)

        assert result.outcome.is_transform is False
        assert result.is_safe is True

    async def test_a_block_behind_a_rewrite_returns_the_block(self, nemoguards_rails_manager):
        """A later block wins: the request is refused, so the rewrite has nothing left to apply to."""
        self._install(
            nemoguards_rails_manager,
            RailDirection.INPUT,
            {
                CONTENT_SAFETY_INPUT_FLOW: StubRail(_mask_user_message(MASKED)),
                TOPIC_SAFETY_INPUT_FLOW: StubRail(RailOutcome.block(reason="off topic")),
            },
        )

        result = await nemoguards_rails_manager.is_input_safe(SSN_MESSAGES, enabled=INPUT_PAIR)

        assert result.is_safe is False
        assert result.outcome.is_transform is False
        assert result.triggered_rail == "topic safety check input"
        assert len(result.records) == 2

    async def test_a_rewrite_leaves_the_callers_messages_untouched(self, nemoguards_rails_manager):
        """The caller's list arrives by identity, so masking must not edit the conversation it owns."""
        messages = [{"role": "user", "content": "my ssn is 123-45-6789"}]
        original_turn = messages[0]
        self._install(
            nemoguards_rails_manager,
            RailDirection.INPUT,
            {CONTENT_SAFETY_INPUT_FLOW: StubRail(_mask_user_message(MASKED))},
        )

        await nemoguards_rails_manager.is_input_safe(messages, enabled=["content safety check input"])

        assert messages == [{"role": "user", "content": "my ssn is 123-45-6789"}]
        assert messages[0] is original_turn

    async def test_an_output_rewrite_reaches_the_next_rail_and_the_verdict(self, nemoguards_rails_manager):
        """The output direction rewrites the response under check rather than the messages."""
        recorder = StubRail(RailOutcome.allow())
        second_flow = "mask pii on output"
        self._install(
            nemoguards_rails_manager,
            RailDirection.OUTPUT,
            {CONTENT_SAFETY_OUTPUT_FLOW: StubRail(_mask_bot_message("call me on <PHONE>")), second_flow: recorder},
        )

        result = await nemoguards_rails_manager._run_rails_sequential(
            [CONTENT_SAFETY_OUTPUT_FLOW, second_flow], RailDirection.OUTPUT, SSN_MESSAGES, "call me on 555-0100"
        )

        assert recorder.seen_bot_response == "call me on <PHONE>"
        assert recorder.seen_messages == SSN_MESSAGES
        assert result.outcome.transform_text == {"bot_message": "call me on <PHONE>"}

    async def test_a_rail_rewriting_the_other_directions_variable_raises(self, nemoguards_rails_manager):
        """An action contradicting its surface's declared target fails loudly rather than being dropped."""
        self._install(
            nemoguards_rails_manager,
            RailDirection.INPUT,
            {CONTENT_SAFETY_INPUT_FLOW: StubRail(_mask_bot_message("masked"))},
        )

        with pytest.raises(NotImplementedError, match="cannot apply"):
            await nemoguards_rails_manager.is_input_safe(SSN_MESSAGES, enabled=["content safety check input"])

    async def test_a_rewrite_from_a_rail_running_in_parallel_raises(self, parallel_input_rails_manager):
        """Concurrent rails all read the arriving text, so a rewrite there cannot be applied to anything."""
        self._install(
            parallel_input_rails_manager,
            RailDirection.INPUT,
            {CONTENT_SAFETY_INPUT_FLOW: StubRail(_mask_user_message(MASKED))},
        )

        with pytest.raises(NotImplementedError, match="in parallel"):
            await parallel_input_rails_manager.is_input_safe(SSN_MESSAGES, enabled=["content safety check input"])

    async def test_a_rail_behind_a_block_is_never_dispatched(self, nemoguards_rails_manager):
        """Rails are built when their turn comes, so a short-circuited one does no work at all."""
        recorder = StubRail(RailOutcome.allow())
        self._install(
            nemoguards_rails_manager,
            RailDirection.INPUT,
            {
                CONTENT_SAFETY_INPUT_FLOW: StubRail(RailOutcome.block(reason="unsafe")),
                TOPIC_SAFETY_INPUT_FLOW: recorder,
            },
        )

        result = await nemoguards_rails_manager.is_input_safe(SSN_MESSAGES, enabled=INPUT_PAIR)

        assert result.is_safe is False
        assert recorder.seen_messages == []


class TestRewritingFlows:
    """`_rewriting_flows` reports which of a direction's configured flows may rewrite."""

    def test_only_rails_declaring_a_target_are_named(self):
        """What the surface declares decides scheduling, not what a given request produced."""
        rails = {
            (RailDirection.INPUT, CONTENT_SAFETY_INPUT_FLOW): StubRail(),
            (RailDirection.INPUT, TOPIC_SAFETY_INPUT_FLOW): declared_rewriter(SurfaceDirection.INPUT),
        }

        rewriting = _rewriting_flows(rails, RailDirection.INPUT, [CONTENT_SAFETY_INPUT_FLOW, TOPIC_SAFETY_INPUT_FLOW])

        assert rewriting == (TOPIC_SAFETY_INPUT_FLOW,)

    def test_a_direction_with_no_rewriting_rail_names_nothing(self):
        """The common config, which must keep running exactly as it did."""
        rails = {(RailDirection.INPUT, CONTENT_SAFETY_INPUT_FLOW): StubRail()}

        assert _rewriting_flows(rails, RailDirection.INPUT, [CONTENT_SAFETY_INPUT_FLOW]) == ()


class TestTransformsFirst:
    """`_transforms_first` puts rails that may rewrite ahead of rails that only judge."""

    def test_a_rewriting_rail_moves_ahead_of_the_rails_that_judge(self):
        """A rewrite is only of use to the rails behind it, so it has to run before them."""
        ordered = _transforms_first([TOPIC_SAFETY_INPUT_FLOW], [CONTENT_SAFETY_INPUT_FLOW, TOPIC_SAFETY_INPUT_FLOW])

        assert ordered == [TOPIC_SAFETY_INPUT_FLOW, CONTENT_SAFETY_INPUT_FLOW]

    def test_configured_order_survives_within_each_group(self):
        """Reordering is between the two groups only; inside each, the config decides."""
        flows = ["a", "b", "c", "d"]

        assert _transforms_first(["b", "d"], flows) == ["b", "d", "a", "c"]

    def test_nothing_moves_when_no_rail_rewrites(self):
        """A config with no rewriting rail runs exactly as it was written."""
        flows = ["a", "b", "c"]

        assert _transforms_first([], flows) == flows


class TestSchedulingRailsThatRewrite:
    """A configured rewriting rail decides run order and rules out concurrent execution."""

    def test_the_rewriting_rails_are_named_per_direction(self):
        """Each direction is asked separately, because the two are scheduled for different reasons."""
        with rails_compiled_as({TOPIC_SAFETY_INPUT_FLOW: declared_rewriter(SurfaceDirection.INPUT)}):
            manager = _make_rails_manager(RailsConfig.from_content(config=NEMOGUARDS_CONFIG))

        assert manager.transform_flows[RailDirection.INPUT] == (TOPIC_SAFETY_INPUT_FLOW,)
        assert manager.transform_flows[RailDirection.OUTPUT] == ()

    @pytest.mark.asyncio
    async def test_a_rewriting_rail_runs_before_a_rail_configured_ahead_of_it(self):
        """Ordering is observable: the judging rail reads text the rewriting rail produced."""
        judge = StubRail()
        with rails_compiled_as(
            {
                CONTENT_SAFETY_INPUT_FLOW: judge,
                TOPIC_SAFETY_INPUT_FLOW: rewriting_stub(MASKED, SurfaceDirection.INPUT),
            }
        ):
            manager = _make_rails_manager(RailsConfig.from_content(config=NEMOGUARDS_CONFIG))

        await manager.is_input_safe(SSN_MESSAGES, enabled=INPUT_PAIR)

        assert judge.seen_messages[-1]["content"] == MASKED

    def test_the_configured_order_is_kept_when_nothing_rewrites(self):
        """Nothing about scheduling changes for the configs that shipped before rewrites existed."""
        with rails_compiled_as({}):
            manager = _make_rails_manager(RailsConfig.from_content(config=NEMOGUARDS_CONFIG))

        assert manager._flows_to_run(RailDirection.INPUT, manager.input_flows, True) == manager.input_flows

    def test_emptying_the_configured_flows_leaves_nothing_to_run(self):
        """The configured list stays the single source of truth for what runs, ordering aside."""
        with rails_compiled_as({CONTENT_SAFETY_INPUT_FLOW: declared_rewriter(SurfaceDirection.INPUT)}):
            manager = _make_rails_manager(RailsConfig.from_content(config=NEMOGUARDS_CONFIG))
        manager.input_flows = []

        assert manager._flows_to_run(RailDirection.INPUT, manager.input_flows, True) == []

    def test_parallel_rails_are_turned_off_in_both_directions(self):
        """One rewriting rail settles how the whole config runs, rather than per section."""
        with rails_compiled_as({CONTENT_SAFETY_OUTPUT_FLOW: declared_rewriter(SurfaceDirection.OUTPUT)}):
            with pytest.warns(UserWarning, match="parallel"):
                manager = _make_rails_manager(RailsConfig.from_content(config=NEMOGUARDS_PARALLEL_CONFIG))

        assert manager.input_parallel is False
        assert manager.output_parallel is False

    def test_the_warning_names_the_rail_that_forced_it(self):
        """A downgrade a user did not ask for has to say which rail caused it."""
        with rails_compiled_as({CONTENT_SAFETY_INPUT_FLOW: declared_rewriter(SurfaceDirection.INPUT)}):
            with pytest.warns(UserWarning, match=CONTENT_SAFETY_INPUT_FLOW.replace("$", r"\$")):
                _make_rails_manager(RailsConfig.from_content(config=NEMOGUARDS_PARALLEL_INPUT_CONFIG))

    def test_parallel_rails_are_left_alone_when_nothing_rewrites(self, recwarn):
        """The downgrade is silent and absent for every config that does not need it."""
        with rails_compiled_as({}):
            manager = _make_rails_manager(RailsConfig.from_content(config=NEMOGUARDS_PARALLEL_CONFIG))

        assert manager.input_parallel is True
        assert manager.output_parallel is True
        assert [warning for warning in recwarn if "parallel" in str(warning.message)] == []

    @pytest.mark.asyncio
    async def test_the_output_direction_is_scheduled_the_same_way(self):
        """Output rails order by the same rule, against the response rather than the messages."""
        judge = StubRail()
        second_flow = "mask pii on output"
        config = RailsConfig.from_content(config=NEMOGUARDS_CONFIG)
        config.rails.output.flows.append(second_flow)
        with rails_compiled_as(
            {
                CONTENT_SAFETY_OUTPUT_FLOW: judge,
                second_flow: rewriting_stub("call me on <PHONE>", SurfaceDirection.OUTPUT),
            }
        ):
            manager = _make_rails_manager(config)

        await manager.is_output_safe(SSN_MESSAGES, "call me on 555-0100")

        assert manager.transform_flows[RailDirection.OUTPUT] == (second_flow,)
        assert judge.seen_bot_response == "call me on <PHONE>"


class TestCheckedText:
    """`_checked_text` names the text a direction's rails read, which is the text a rewrite replaces."""

    def test_input_rails_check_the_current_user_turn(self):
        """Input rails judge the turn being handled, not the whole conversation."""
        assert _checked_text(RailDirection.INPUT, SSN_MESSAGES, None) == "my ssn is 123-45-6789"

    def test_input_rails_ignore_a_response(self):
        """A response only exists for the output direction, so it cannot be what an input rail read."""
        assert _checked_text(RailDirection.INPUT, SSN_MESSAGES, "a reply") == "my ssn is 123-45-6789"

    def test_output_rails_check_the_response(self):
        """Output rails judge the generated response, which is not yet part of the messages."""
        assert _checked_text(RailDirection.OUTPUT, SSN_MESSAGES, "a reply") == "a reply"

    def test_output_rails_with_no_response_check_nothing(self):
        """A turn that produced no text compares equal to itself, so it reports no rewrite."""
        assert _checked_text(RailDirection.OUTPUT, SSN_MESSAGES, None) == ""


class TestRewrittenText:
    """`_rewritten_text` reads the rewrite a direction can apply, and refuses every other one."""

    def test_an_input_rail_rewrites_the_user_message(self):
        """The variable an input rail is allowed to rewrite."""
        assert _rewritten_text(_mask_user_message(MASKED), RailDirection.INPUT, "mask pii on input") == MASKED

    def test_an_output_rail_rewrites_the_bot_message(self):
        """The variable an output rail is allowed to rewrite."""
        assert _rewritten_text(_mask_bot_message("redacted"), RailDirection.OUTPUT, "mask pii on output") == "redacted"

    def test_the_other_directions_variable_is_refused(self):
        """An action contradicting its surface's declared target is a bug, not a verdict to apply."""
        with pytest.raises(NotImplementedError, match="may rewrite 'user_message'"):
            _rewritten_text(_mask_bot_message("redacted"), RailDirection.INPUT, "mask pii on input")

    def test_rewriting_more_than_one_variable_is_refused(self):
        """Applying half of a two-variable rewrite would report a verdict over text no rail produced."""
        outcome = RailOutcome.transform(
            [(TransformTarget.USER_MESSAGE, MASKED), (TransformTarget.BOT_MESSAGE, "redacted")]
        )

        with pytest.raises(NotImplementedError, match="cannot apply"):
            _rewritten_text(outcome, RailDirection.INPUT, "pangea ai guard input")

    def test_a_retrieval_rewrite_is_refused(self):
        """``relevant_chunks`` has no home in an input/output request, so there is nowhere to put it."""
        outcome = RailOutcome.transform([(TransformTarget.RELEVANT_CHUNKS, "chunk text")])

        with pytest.raises(NotImplementedError, match="cannot apply"):
            _rewritten_text(outcome, RailDirection.INPUT, "mask pii on retrieval")


class TestResultAfterRewrites:
    """`_result_after_rewrites` reports a direction's net rewrite, or that nothing changed."""

    RECORDS = (RailCallRecord(flow="mask pii on input", rail_type="input", is_safe=True),)

    def test_unchanged_text_is_a_plain_allow(self):
        """Rails that rewrote nothing, or rewrote and restored, leave the request as it arrived."""
        result = _result_after_rewrites(RailDirection.INPUT, "hello", "hello", self.RECORDS)

        assert result.outcome.is_transform is False
        assert result.is_safe is True
        assert result.records == self.RECORDS

    def test_an_input_rewrite_names_the_user_message(self):
        """The caller learns which variable to replace, not just that something changed."""
        result = _result_after_rewrites(RailDirection.INPUT, "my ssn is 123-45-6789", MASKED, self.RECORDS)

        assert result.outcome.transform_text == {"user_message": MASKED}
        assert result.is_safe is True
        assert result.records == self.RECORDS

    def test_an_output_rewrite_names_the_bot_message(self):
        """The output direction reports against the variable its rails checked."""
        result = _result_after_rewrites(RailDirection.OUTPUT, "call me on 555-0100", "call me on <PHONE>", ())

        assert result.outcome.transform_text == {"bot_message": "call me on <PHONE>"}

    def test_a_rewrite_to_empty_text_is_still_a_rewrite(self):
        """Redacting a response to nothing is a change, and must not read as leaving it alone."""
        result = _result_after_rewrites(RailDirection.OUTPUT, "my ssn is 123-45-6789", "", ())

        assert result.outcome.transform_text == {"bot_message": ""}

    def test_a_rewrite_keeps_every_rail_that_ran(self):
        """The log covers the whole direction, not only the rail that rewrote."""
        result = _result_after_rewrites(RailDirection.INPUT, "before", "after", self.RECORDS)

        assert result.records == self.RECORDS


class TestRefuseConcurrentRewrite:
    """`_refuse_concurrent_rewrite` keeps rewrites out of the path that cannot compose them."""

    @pytest.mark.parametrize(
        "outcome",
        [RailOutcome.allow(), RailOutcome.block(reason="unsafe")],
        ids=["allow", "block"],
    )
    def test_a_decision_passes_through(self, outcome):
        """Parallel mode is unaffected for the verdicts its rails can actually reach."""
        assert _refuse_concurrent_rewrite(RailResult(outcome), "content safety check input") is None

    def test_a_rewrite_raises(self):
        """Concurrent rails all read the arriving text, so no rewrite among them can be applied."""
        result = RailResult(_mask_user_message(MASKED))

        with pytest.raises(NotImplementedError, match="in parallel"):
            _refuse_concurrent_rewrite(result, "mask pii on input")


class TestRailCallRecordNaming:
    """`_rail_call_record` names task/action_name in LLMRails' underscore form.

    The GenerationLog's ``executed_actions[].action_name`` and ``llm_calls[].task``
    must match LLMRails, which uses the prompt-template key (underscores) rather than
    the space-separated Colang flow name. ``flow`` itself keeps the space form.
    """

    @pytest.mark.parametrize(
        "flow, action_name, task",
        [
            (
                "content safety check input $model=content_safety",
                "content_safety_check_input",
                "content_safety_check_input $model=content_safety",
            ),
            ("jailbreak detection model", "jailbreak_detection_model", "jailbreak_detection_model"),
        ],
        ids=["modelled", "modelless"],
    )
    def test_underscore_task_and_action_name(self, flow, action_name, task):
        """action_name/task use the underscore prompt-template key; ``flow`` keeps its space form."""
        record = _rail_call_record(flow=flow, rail_type="input", result=RailResult.allow())

        assert record.flow == flow
        assert record.action_name == action_name
        assert record.task == task


class TestRailCallRecordMultipleCalls:
    """A rail's sink is a list but `RailCallRecord` holds one call, so extras are reported, not dropped."""

    @staticmethod
    def _call(model: str, tokens: int) -> LLMCallInfo:
        return LLMCallInfo(llm_model_name=model, prompt_tokens=tokens, completion_tokens=0, total_tokens=tokens)

    def test_the_last_call_wins_and_the_rest_are_logged(self, caplog):
        """No in-scope rail makes two model calls; if one ever does, that shows up as a warning."""
        calls = [self._call("first-model", 10), self._call("second-model", 20)]

        with caplog.at_level(logging.WARNING, logger="nemoguardrails.guardrails.rails_manager"):
            record = _rail_call_record(
                flow="content safety check input $model=content_safety",
                rail_type="input",
                result=RailResult.allow(),
                calls=calls,
            )

        assert record.made_call is True
        assert record.llm_model_name == "second-model"
        assert record.usage.total_tokens == 20
        assert "made 2 model calls" in caplog.text

    def test_one_call_records_it_without_a_warning(self, caplog):
        """The single-call case is the expected one, so it passes through quietly."""
        with caplog.at_level(logging.WARNING, logger="nemoguardrails.guardrails.rails_manager"):
            record = _rail_call_record(
                flow="content safety check input $model=content_safety",
                rail_type="input",
                result=RailResult.allow(),
                calls=[self._call("only-model", 7)],
            )

        assert record.made_call is True
        assert record.llm_model_name == "only-model"
        assert caplog.text == ""


class TestSerializePrompt:
    """`serialize_prompt` renders a message list to a role-labeled string for the log."""

    def test_role_labeled_join(self):
        """Each message renders as '<role>: <content>', blank-line separated."""
        out = serialize_prompt(
            [
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "hi"},
            ]
        )
        assert out == "system: be nice\n\nuser: hi"

    def test_missing_content_renders_empty(self):
        """A message with no content and no other fields renders as just the role label."""
        out = serialize_prompt([{"role": "assistant", "content": None}])
        assert out == "assistant: "

    def test_tool_calls_preserved(self):
        """An assistant tool-call turn keeps its tool_calls rather than rendering blank."""
        out = serialize_prompt(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_1", "function": {"name": "get_weather"}}],
                }
            ]
        )
        assert "call_1" in out
        assert "get_weather" in out

    def test_tool_result_fields_preserved(self):
        """A tool-result turn keeps its tool_call_id and name alongside the content."""
        out = serialize_prompt([{"role": "tool", "content": "sunny", "tool_call_id": "call_1", "name": "get_weather"}])
        assert "sunny" in out
        assert "call_1" in out
        assert "get_weather" in out

    def test_reasoning_preserved(self):
        """A reasoning-only turn keeps its reasoning text instead of dropping it."""
        out = serialize_prompt([{"role": "assistant", "content": None, "reasoning": "thinking hard"}])
        assert "thinking hard" in out


class TestParallelBatchDrainsRecords:
    """`_run_rails_parallel` keeps records from every task that completed in a batch."""

    @pytest.mark.asyncio
    async def test_unsafe_first_does_not_drop_later_safe_records(self):
        """When an unsafe rail sorts before a safe one that finished in the same wait batch, the safe rail's records survive."""
        manager = _make_rails_manager(RailsConfig.from_content(config=CONTENT_SAFETY_CONFIG))

        unsafe_record = RailCallRecord(flow="jailbreak detection model", rail_type="input", is_safe=False)
        safe_record = RailCallRecord(flow="content safety check input", rail_type="input", is_safe=True)

        async def _unsafe():
            return RailResult.block(reason="blocked", records=(unsafe_record,))

        async def _safe():
            return RailResult.allow(records=(safe_record,))

        # Insertion order sets task_order, so the unsafe rail sorts first in the done batch.
        rails = {"unsafe": _unsafe(), "safe": _safe()}
        result = await manager._run_rails_parallel(rails, RailDirection.INPUT)

        assert result.is_safe is False
        assert {r.flow for r in result.records} == {"jailbreak detection model", "content safety check input"}


def _content_safety_config_without_max_tokens() -> dict:
    """CONTENT_SAFETY_CONFIG with the prompt max_tokens removed, so the action's default applies."""
    config = copy.deepcopy(CONTENT_SAFETY_CONFIG)
    for prompt in config["prompts"]:
        prompt.pop("max_tokens", None)
    return config


def _mocked_reply(reply: str):
    """Patch the transport both the old and new rail paths bottom out in."""
    return patch.object(ModelEngine, "chat_completion", AsyncMock(return_value=LLMResponse(content=reply)))


class TestLibraryActionContract:
    """What a rail's RailResult carries once the library action produces the verdict."""

    @pytest.mark.asyncio
    async def test_a_block_carries_no_reason(self, content_safety_rails_manager):
        """A blocking rail leaves reason unset; the action supplies evidence, not prose."""
        with _mocked_reply(UNSAFE_INPUT_JSON):
            result = await content_safety_rails_manager.is_input_safe(MESSAGES)

        assert result.is_safe is False
        assert result.reason is None

    @pytest.mark.asyncio
    async def test_a_block_verdict_carries_the_outcome_metadata(self, content_safety_rails_manager):
        """The verdict merges the decision with the action's evidence."""
        with _mocked_reply(UNSAFE_INPUT_JSON):
            result = await content_safety_rails_manager.is_input_safe(MESSAGES)

        assert result.return_value == {"allowed": False, "policy_violations": ["S1: Violence"]}

    @pytest.mark.asyncio
    async def test_topic_safety_reports_its_decision_under_allowed(self, topic_safety_rails_manager):
        """Topic safety uses the same verdict key as every other migrated rail."""
        with _mocked_reply("off-topic"):
            result = await topic_safety_rails_manager.is_input_safe(MESSAGES)

        assert result.is_safe is False
        assert result.return_value == {"allowed": False}

    @pytest.mark.asyncio
    async def test_an_unconfigured_max_tokens_uses_the_library_default(self):
        """With no max_tokens in the prompt config, the library's own default reaches the model."""
        manager = _make_rails_manager(RailsConfig.from_content(config=_content_safety_config_without_max_tokens()))
        chat = AsyncMock(return_value=LLMResponse(content=SAFE_INPUT_JSON))

        with patch.object(ModelEngine, "chat_completion", chat):
            await manager.is_input_safe(MESSAGES)

        assert chat.await_args.kwargs["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_a_request_with_no_user_message_does_not_block(self, content_safety_rails_manager):
        """An absent user turn reaches the model as empty text rather than failing closed."""
        with _mocked_reply(SAFE_INPUT_JSON):
            result = await content_safety_rails_manager.is_input_safe([{"role": "system", "content": "be nice"}])

        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_the_model_call_reaches_the_generation_record(self, content_safety_rails_manager):
        """A rail's model call is captured with the task label GenerationLog reports."""
        with _mocked_reply(SAFE_INPUT_JSON):
            result = await content_safety_rails_manager.is_input_safe(MESSAGES)

        record = result.records[0]
        assert record.made_call is True
        assert record.task == "content_safety_check_input $model=content_safety"


def _completion(content: str, **extra) -> dict:
    """A provider chat-completion payload carrying *content*."""
    message = {"role": "assistant", "content": content, **extra}
    return {"id": "chatcmpl-1", "model": "m", "choices": [{"message": message, "finish_reason": "stop"}]}


class TestRawResponseParsing:
    """A rail's verdict survives the whole chain, mocked below ``_parse_chat_completion``."""

    @pytest.mark.asyncio
    async def test_safe_input(self, content_safety_rails_manager):
        """A safe classification parses through to an allow."""
        call = mock_rail_http_response(content_safety_rails_manager.engine_registry, _completion(SAFE_INPUT_JSON))

        result = await content_safety_rails_manager.is_input_safe(MESSAGES)

        assert result.is_safe
        call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unsafe_input_carries_its_categories(self, content_safety_rails_manager):
        """An unsafe classification parses through to a block naming the violated policy."""
        mock_rail_http_response(content_safety_rails_manager.engine_registry, _completion(UNSAFE_INPUT_JSON))

        result = await content_safety_rails_manager.is_input_safe(MESSAGES)

        assert not result.is_safe
        assert result.return_value["policy_violations"] == ["S1: Violence"]

    @pytest.mark.asyncio
    async def test_unsafe_output_carries_its_categories(self, content_safety_rails_manager):
        """The output rail parses its own verdict field, not the input one."""
        mock_rail_http_response(content_safety_rails_manager.engine_registry, _completion(UNSAFE_OUTPUT_JSON))

        result = await content_safety_rails_manager.is_output_safe(MESSAGES, "bot response")

        assert not result.is_safe
        assert result.return_value["policy_violations"] == ["S17: Malware"]

    @pytest.mark.asyncio
    async def test_reasoning_content_does_not_affect_classification(self, content_safety_rails_manager):
        """Only ``content`` is parsed; a reasoning field alongside it is ignored."""
        payload = _completion(SAFE_INPUT_JSON, reasoning_content="the prompt looks fine to me")
        mock_rail_http_response(content_safety_rails_manager.engine_registry, payload)

        result = await content_safety_rails_manager.is_input_safe(MESSAGES)

        assert result.is_safe

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reply, expected_safe", [("on-topic", True), ("off-topic", False)])
    async def test_topic_safety_verdict(self, topic_safety_rails_manager, reply, expected_safe):
        """The topic-control reply parses through to the matching verdict."""
        mock_rail_http_response(topic_safety_rails_manager.engine_registry, _completion(reply))

        result = await topic_safety_rails_manager.is_input_safe(MESSAGES)

        assert result.is_safe is expected_safe

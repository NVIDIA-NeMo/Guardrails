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

"""Unit tests for engine_registry module."""

from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from nemoguardrails.guardrails import telemetry
from nemoguardrails.guardrails.api_engine import APIEngine
from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.tracing import constants as tracing_constants
from nemoguardrails.tracing.constants import SystemConstants
from nemoguardrails.types import LLMResponse, LLMResponseChunk, UsageInfo
from tests.guardrails.metric_helpers import collect_histogram_sum, collect_metric_points
from tests.guardrails.test_data import NEMOGUARDS_CONFIG


@pytest.fixture
def rails_config():
    """Create a RailsConfig from the nemoguards_v2 test data."""
    return RailsConfig.from_content(config=NEMOGUARDS_CONFIG)


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def manager(rails_config):
    """Create a EngineRegistry from test config."""
    return EngineRegistry(rails_config.models, rails_config.rails.config)


@pytest.fixture(autouse=True)
def reset_telemetry_singletons():
    """Reset telemetry's module-level singletons before and after every
    test in this file.  Includes ``_tracer`` so cached tracer state
    doesn't leak into other test files (notably the LLMRails OTEL
    adapter tests).
    """
    telemetry._meter = None
    tracing_constants._llm_instruments = None
    telemetry._request_instruments = None
    telemetry._tracer = None
    yield
    telemetry._meter = None
    tracing_constants._llm_instruments = None
    telemetry._request_instruments = None
    telemetry._tracer = None


@pytest.fixture
def metric_reader():
    """Install a test-local Meter, return its reader.  Cleanup is
    handled by the autouse ``reset_telemetry_singletons`` fixture."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    telemetry._meter = provider.get_meter(
        SystemConstants.SYSTEM_NAME,
        version="0.0.0-dev",
        schema_url="https://opentelemetry.io/schemas/1.26.0",
    )
    return reader


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def manager_with_metrics(rails_config):
    """Create an EngineRegistry with metrics emission enabled."""
    return EngineRegistry(rails_config.models, rails_config.rails.config, metrics_enabled=True)


def _mock_stream(*chunks: LLMResponseChunk, error: Optional[Exception] = None):
    """Build an async generator that yields ``chunks`` in order, then
    optionally raises ``error``.  Drop-in replacement for inline
    ``async def mock_stream(msgs, **kwargs): yield ...`` definitions,
    cutting two lines of boilerplate per call site.
    """

    async def _gen(msgs, **kwargs):  # noqa: ARG001 (signature dictated by ModelEngine)
        for chunk in chunks:
            yield chunk
        if error is not None:
            raise error

    return _gen


class TestEngineRegistryInit:
    """Test EngineRegistry creates engines from config."""

    def test_create_engines_for_each_model_type(self, manager):
        """Creates one engine per model type in config, plus API engines."""
        engine_names = set(manager._engines.keys())
        assert {"main", "content_safety", "topic_control", "jailbreak_detection"} == engine_names

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    def test_empty_config_creates_no_engines(self):
        """Empty models list results in no engines."""
        config = RailsConfig.from_content(config={"models": []})
        mgr = EngineRegistry(config.models, config.rails.config)
        assert len(mgr._engines) == 0

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    def test_model_type_collision_with_api_engine_raises(self):
        """Raises ValueError when a model type collides with an API engine name."""
        config = RailsConfig.from_content(
            config={
                "models": [
                    {"type": "main", "engine": "nim", "model": "meta/llama-3.3-70b-instruct"},
                    {"type": "jailbreak_detection", "engine": "nim", "model": "some/model"},
                ],
                "rails": {
                    "config": {
                        "jailbreak_detection": {
                            "nim_base_url": "https://ai.api.nvidia.com",
                            "nim_server_endpoint": "/v1/security/nvidia/nemoguard-jailbreak-detect",
                            "api_key_env_var": "NVIDIA_API_KEY",
                        }
                    }
                },
            }
        )
        with pytest.raises(ValueError, match="already registered"):
            EngineRegistry(config.models, config.rails.config)


class TestEngineRegistryGetModelEngine:
    """Test engine lookup by model type."""

    def test_get_existing_engine(self, manager):
        """Returns the main LLM engine with correct model name."""
        engine = manager._get_engine("main", ModelEngine)
        assert engine is not None
        assert engine.model_name == "meta/llama-3.3-70b-instruct"

    def test_get_content_safety_engine(self, manager):
        """Returns the content safety engine with correct model name."""
        engine = manager._get_engine("content_safety", ModelEngine)
        assert engine.model_name == "nvidia/llama-3.1-nemoguard-8b-content-safety"

    def test_get_missing_engine_raises_key_error(self, manager):
        """Raises KeyError for an unconfigured model type."""
        with pytest.raises(KeyError, match="No engine configured with name 'nonexistent'"):
            manager._get_engine("nonexistent", ModelEngine)

    def test_key_error_message_lists_available_types(self, manager):
        """KeyError message includes available model types for debugging."""
        with pytest.raises(KeyError) as exc_info:
            manager._get_engine("missing", ModelEngine)
        assert "main" in str(exc_info.value)

    def test_wrong_type_raises_type_error(self, manager):
        """Raises TypeError when engine exists but is the wrong type."""
        with pytest.raises(TypeError, match="Engine 'jailbreak_detection' is APIEngine, expected ModelEngine"):
            manager._get_engine("jailbreak_detection", ModelEngine)


class TestEngineRegistryLifecycle:
    """Test EngineRegistry start/stop delegation to engines."""

    @pytest.mark.asyncio
    async def test_start_calls_start_on_all_engines(self, manager):
        """start() delegates to each engine's start() and sets _running."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()

        assert not manager._running
        await manager.start()
        assert manager._running

        for engine in manager._engines.values():
            engine.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_calls_stop_on_all_engines(self, manager):
        """stop() delegates to each engine's stop() and clears _running."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()
            engine.stop = AsyncMock()

        await manager.start()
        assert manager._running
        await manager.stop()
        assert not manager._running

        for engine in manager._engines.values():
            engine.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, manager):
        """Calling start() twice only starts engines once."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()

        await manager.start()
        await manager.start()  # second call is a no-op

        for engine in manager._engines.values():
            engine.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, manager):
        """Calling stop() twice only stops engines once."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()
            engine.stop = AsyncMock()

        await manager.start()
        await manager.stop()
        await manager.stop()  # second call is a no-op

        for engine in manager._engines.values():
            engine.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self, manager):
        """stop() without a prior start() does not raise."""
        for engine in manager._engines.values():
            engine.stop = AsyncMock()

        await manager.stop()  # should not raise
        assert not manager._running

        for engine in manager._engines.values():
            engine.stop.assert_not_called()


class TestEngineRegistryGenerateAsync:
    """Test model_call routes to the correct engine."""

    @pytest.mark.asyncio
    async def test_generate_from_correct_engine(self, manager):
        """Calls the named engine's chat_completion() and returns its LLMResponse."""
        messages = [{"role": "user", "content": "Hi"}]
        engine = manager._get_engine("main", ModelEngine)
        expected = LLMResponse(content="Hello world")
        engine.chat_completion = AsyncMock(return_value=expected)

        result = await manager.model_call("main", messages)
        assert result is expected
        engine.chat_completion.assert_called_once_with(messages)

    @pytest.mark.asyncio
    async def test_passes_kwargs_to_engine(self, manager):
        """Extra kwargs (temperature, max_tokens) are forwarded to engine.chat_completion()."""
        messages = [{"role": "user", "content": "Hi"}]
        engine = manager._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(return_value=LLMResponse(content="ok"))

        await manager.model_call("main", messages, temperature=0.5, max_tokens=100)

        call_kwargs = engine.chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_raises_key_error_for_unknown_model_type(self, manager):
        """Raises KeyError when the model type doesn't exist."""
        with pytest.raises(KeyError):
            await manager.model_call("nonexistent", [{"role": "user", "content": "Hi"}])


class TestEngineRegistryModelCallMetrics:
    """``model_call`` emits the OTEL GenAI client metrics
    (``gen_ai.client.token.usage`` Histogram, ``gen_ai.client.operation.duration``
    Histogram) when constructed with ``metrics_enabled=True``."""

    @pytest.mark.asyncio
    async def test_emits_token_usage_and_duration_on_safe_call(self, manager_with_metrics, metric_reader):
        """``LLMResponse.usage`` populated → both metrics emit.  Token
        usage produces two observations (input + output) labelled by
        ``gen_ai.token.type``."""
        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(
            return_value=LLMResponse(content="hi", usage=UsageInfo(input_tokens=10, output_tokens=5)),
        )

        await manager_with_metrics.model_call("main", [{"role": "user", "content": "hi"}])

        points = collect_metric_points(metric_reader)
        assert len(points["gen_ai.client.token.usage"]) == 2
        assert {p.attributes["gen_ai.token.type"] for p in points["gen_ai.client.token.usage"]} == {
            "input",
            "output",
        }
        # Histogram value is the recording count.
        assert points["gen_ai.client.operation.duration"][0].value == 1
        # Successful call → no error.type label on duration.
        assert "error.type" not in points["gen_ai.client.operation.duration"][0].attributes

    @pytest.mark.asyncio
    async def test_skips_token_usage_when_response_usage_is_none(self, manager_with_metrics, metric_reader):
        """``LLMResponse.usage = None`` → token metric absent, duration
        still emits.  Models the case where the provider didn't return
        a ``usage`` field (some non-OpenAI-compatible NIMs)."""
        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(return_value=LLMResponse(content="hi", usage=None))

        await manager_with_metrics.model_call("main", [{"role": "user", "content": "hi"}])

        points = collect_metric_points(metric_reader)
        assert "gen_ai.client.token.usage" not in points
        assert points["gen_ai.client.operation.duration"][0].value == 1

    @pytest.mark.asyncio
    async def test_records_duration_with_error_type_on_exception(self, manager_with_metrics, metric_reader):
        """Engine raises → duration emits with ``error.type=ExceptionClass``,
        token usage absent (the call never produced usage data), exception
        propagates."""
        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(side_effect=RuntimeError("provider down"))

        with pytest.raises(RuntimeError, match="provider down"):
            await manager_with_metrics.model_call("main", [{"role": "user", "content": "hi"}])

        points = collect_metric_points(metric_reader)
        assert "gen_ai.client.token.usage" not in points
        assert points["gen_ai.client.operation.duration"][0].attributes["error.type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_label_set_includes_provider_model_operation(self, manager_with_metrics, metric_reader):
        """Standard OTEL labels on every emitted observation:
        ``gen_ai.operation.name``, ``gen_ai.provider.name``,
        ``gen_ai.request.model``."""
        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(
            return_value=LLMResponse(content="hi", usage=UsageInfo(input_tokens=1, output_tokens=1)),
        )

        await manager_with_metrics.model_call("main", [{"role": "user", "content": "hi"}])

        points = collect_metric_points(metric_reader)
        for point in points["gen_ai.client.token.usage"] + points["gen_ai.client.operation.duration"]:
            assert point.attributes["gen_ai.operation.name"] == "chat"
            # NEMOGUARDS_CONFIG's "main" model uses the nim engine.
            assert point.attributes["gen_ai.provider.name"] == "nim"
            assert point.attributes["gen_ai.request.model"] == "meta/llama-3.3-70b-instruct"

    @pytest.mark.asyncio
    async def test_no_metrics_emitted_when_metrics_disabled(self, manager, metric_reader):
        """``metrics_enabled=False`` (default) → no metrics fire even
        when a MeterProvider is installed.  Catches the gating slip
        where the helper would emit purely on meter availability."""
        engine = manager._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(
            return_value=LLMResponse(content="hi", usage=UsageInfo(input_tokens=1, output_tokens=1)),
        )

        await manager.model_call("main", [{"role": "user", "content": "hi"}])

        points = collect_metric_points(metric_reader)
        assert "gen_ai.client.token.usage" not in points
        assert "gen_ai.client.operation.duration" not in points


class TestEngineRegistryStartErrors:
    """Test EngineRegistry start() error handling and rollback."""

    @pytest.mark.asyncio
    async def test_start_rolls_back_on_engine_failure(self, manager):
        """When one engine fails to start, already-started engines are stopped."""
        failing_engine = "jailbreak_detection"

        for name, engine in manager._engines.items():
            if name == failing_engine:
                engine.start = AsyncMock(side_effect=RuntimeError("Error starting model"))
            else:
                engine.start = AsyncMock()
                engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Failed to start engine"):
            await manager.start()

        # Engines before the failing one should have been rolled back
        engine_names = list(manager._engines.keys())
        failed_idx = engine_names.index(failing_engine)
        for i, name in enumerate(engine_names):
            if i < failed_idx:
                manager._engines[name].stop.assert_called_once()

        assert not manager._running

    @pytest.mark.asyncio
    async def test_start_error_message_includes_engine_type(self, manager):
        """Error message includes which engine types failed."""
        engine = manager._get_engine("main", ModelEngine)
        engine.start = AsyncMock(side_effect=RuntimeError("connection refused"))

        # Mock other engines to succeed
        for engine_type, engine in manager._engines.items():
            if engine_type != "main":
                engine.start = AsyncMock()
                engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Engine main"):
            await manager.start()

    @pytest.mark.asyncio
    async def test_start_rollback_swallows_stop_errors(self, manager):
        """Rollback continues even if stopping a started engine raises."""
        failing_engine = "jailbreak_detection"
        stop_error_engine = "main"

        for name, engine in manager._engines.items():
            if name == failing_engine:
                engine.start = AsyncMock(side_effect=RuntimeError("start failed"))
            elif name == stop_error_engine:
                engine.start = AsyncMock()
                engine.stop = AsyncMock(side_effect=RuntimeError("stop failed"))
            else:
                engine.start = AsyncMock()
                engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Failed to start engine"):
            await manager.start()

        # All started engines should have had stop() called (even if one raises)
        engine_names = list(manager._engines.keys())
        failed_idx = engine_names.index(failing_engine)
        for i, name in enumerate(engine_names):
            if i < failed_idx:
                manager._engines[name].stop.assert_called_once()


class TestEngineRegistryStopErrors:
    """Test EngineRegistry stop() error handling."""

    @pytest.mark.asyncio
    async def test_stop_raises_on_engine_error(self, manager):
        """stop() raises RuntimeError when an engine fails to stop."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()

        await manager.start()

        # One engine fails to stop
        engine = manager._get_engine("main", ModelEngine)
        engine.stop = AsyncMock(side_effect=RuntimeError("close failed"))
        for engine_type, engine in manager._engines.items():
            if engine_type != "main":
                engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Failed to stop engines"):
            await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_error_includes_engine_type(self, manager):
        """Error message includes which engine type failed to stop."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()

        await manager.start()

        engine = manager._get_engine("content_safety", ModelEngine)
        engine.stop = AsyncMock(side_effect=RuntimeError("timeout"))
        for engine_type, engine in manager._engines.items():
            if engine_type != "content_safety":
                engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Engine content_safety"):
            await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_attempts_all_engines_even_on_errors(self, manager):
        """stop() tries to stop all engines, not just the first one that fails."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()

        await manager.start()

        for engine in manager._engines.values():
            engine.stop = AsyncMock(side_effect=RuntimeError("fail"))

        with pytest.raises(RuntimeError):
            await manager.stop()

        # All engines should have had stop() called
        for engine in manager._engines.values():
            engine.stop.assert_called_once()


class TestEngineRegistryContextManager:
    """Test async context manager calls start/stop correctly."""

    @pytest.mark.asyncio
    async def test_context_manager_calls_start_and_stop(self, manager):
        """async with calls start() on enter and stop() on exit."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()
            engine.stop = AsyncMock()

        async with manager as mgr:
            assert mgr is manager
            for engine in manager._engines.values():
                engine.start.assert_called_once()

        for engine in manager._engines.values():
            engine.stop.assert_called_once()


class TestEngineRegistryGetApiEngine:
    """Test API engine lookup by name."""

    def test_get_existing_api_engine(self, manager):
        """Returns the jailbreak detection API engine."""
        api_engine = manager._get_engine("jailbreak_detection", APIEngine)
        assert api_engine is not None
        assert "jailbreak" in api_engine.url

    def test_get_missing_api_engine_raises_key_error(self, manager):
        """Raises KeyError for an unconfigured API engine name."""
        with pytest.raises(KeyError, match="No engine configured with name 'nonexistent'"):
            manager._get_engine("nonexistent", APIEngine)

    def test_key_error_message_lists_available_engines(self, manager):
        """KeyError message includes available API engine names."""
        with pytest.raises(KeyError) as exc_info:
            manager._get_engine("missing", APIEngine)
        assert "jailbreak_detection" in str(exc_info.value)


class TestEngineRegistryApiCall:
    """Test api_call routes to the correct API engine."""

    @pytest.mark.asyncio
    async def test_calls_correct_api_engine(self, manager):
        """api_call delegates to the named API engine and returns its response."""
        api_engine = manager._get_engine("jailbreak_detection", APIEngine)
        mock_response = {"jailbreak": False, "score": -0.95}
        api_engine.call = AsyncMock(return_value=mock_response)

        result = await manager.api_call("jailbreak_detection", {"input": "hello"})

        assert result == mock_response
        api_engine.call.assert_called_once_with({"input": "hello"})

    @pytest.mark.asyncio
    async def test_passes_kwargs_to_api_engine(self, manager):
        """Extra kwargs are forwarded to the API engine's call()."""
        api_engine = manager._get_engine("jailbreak_detection", APIEngine)
        api_engine.call = AsyncMock(return_value={"jailbreak": False, "score": -0.80})

        await manager.api_call("jailbreak_detection", {"input": "test"}, extra_param="value")

        api_engine.call.assert_called_once_with({"input": "test"}, extra_param="value")

    @pytest.mark.asyncio
    async def test_raises_key_error_for_unknown_api_name(self, manager):
        """Raises KeyError when the API engine name doesn't exist."""
        with pytest.raises(KeyError):
            await manager.api_call("nonexistent", {"input": "test"})


class TestEngineRegistryApiEngineStartErrors:
    """Test start() error handling for API engines."""

    @pytest.mark.asyncio
    async def test_start_rolls_back_on_api_engine_failure(self, manager):
        """When an API engine fails to start, all started engines are rolled back."""
        # Mock all engines to succeed
        for engine in manager._engines.values():
            engine.start = AsyncMock()
            engine.stop = AsyncMock()

        # Override jailbreak API engine to fail
        api_engine = manager._get_engine("jailbreak_detection", APIEngine)
        api_engine.start = AsyncMock(side_effect=RuntimeError("API unreachable"))
        api_engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Failed to start engine"):
            await manager.start()

        # Engines that started successfully should have been rolled back
        for name, engine in manager._engines.items():
            if name != "jailbreak_detection":
                engine.stop.assert_called_once()

        assert not manager._running

    @pytest.mark.asyncio
    async def test_start_error_message_includes_api_engine_name(self, manager):
        """Error message includes which API engine failed to start."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()
            engine.stop = AsyncMock()

        api_engine = manager._get_engine("jailbreak_detection", APIEngine)
        api_engine.start = AsyncMock(side_effect=RuntimeError("timeout"))
        api_engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Engine jailbreak_detection"):
            await manager.start()


class TestEngineRegistryApiEngineStopErrors:
    """Test stop() error handling for API engines."""

    @pytest.mark.asyncio
    async def test_stop_raises_on_api_engine_error(self, manager):
        """stop() raises RuntimeError when an API engine fails to stop."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()
            engine.stop = AsyncMock()

        api_engine = manager._get_engine("jailbreak_detection", APIEngine)
        api_engine.start = AsyncMock()
        api_engine.stop = AsyncMock(side_effect=RuntimeError("close failed"))

        await manager.start()

        with pytest.raises(RuntimeError, match="Failed to stop engines"):
            await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_error_includes_api_engine_name(self, manager):
        """Error message includes which API engine failed to stop."""
        for engine in manager._engines.values():
            engine.start = AsyncMock()
            engine.stop = AsyncMock()

        api_engine = manager._get_engine("jailbreak_detection", APIEngine)
        api_engine.start = AsyncMock()
        api_engine.stop = AsyncMock(side_effect=RuntimeError("timeout"))

        await manager.start()

        with pytest.raises(RuntimeError, match="Engine jailbreak_detection"):
            await manager.stop()


class TestEngineRegistryStreamModelCall:
    """Test stream_model_call routes to the correct engine and yields LLMResponseChunk objects."""

    @pytest.mark.asyncio
    async def test_streams_chunks_from_correct_engine(self, manager):
        """Calls the named engine's stream_chat_completion and forwards LLMResponseChunk objects."""
        messages = [{"role": "user", "content": "Hi"}]

        async def mock_stream_chat_completion(msgs, **kwargs):
            for text in ["Hello", " world"]:
                yield LLMResponseChunk(delta_content=text)

        engine = manager._get_engine("main", ModelEngine)
        engine.stream_chat_completion = mock_stream_chat_completion

        chunks = []
        async for chunk in manager.stream_model_call("main", messages):
            chunks.append(chunk)

        assert all(isinstance(c, LLMResponseChunk) for c in chunks)
        assert [c.delta_content for c in chunks] == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_streams_reasoning_and_content_chunks(self, manager):
        """Reasoning deltas flow through alongside content deltas."""
        messages = [{"role": "user", "content": "Hi"}]

        async def mock_stream_chat_completion(msgs, **kwargs):
            yield LLMResponseChunk(delta_reasoning="thinking")
            yield LLMResponseChunk(delta_content="Hello")
            yield LLMResponseChunk(delta_reasoning=" more")

        engine = manager._get_engine("main", ModelEngine)
        engine.stream_chat_completion = mock_stream_chat_completion

        chunks = []
        async for chunk in manager.stream_model_call("main", messages):
            chunks.append(chunk)

        assert [(c.delta_content, c.delta_reasoning) for c in chunks] == [
            (None, "thinking"),
            ("Hello", None),
            (None, " more"),
        ]

    @pytest.mark.asyncio
    async def test_forwards_kwargs_to_engine(self, manager):
        """Extra kwargs are forwarded to engine.stream_chat_completion()."""
        messages = [{"role": "user", "content": "Hi"}]
        captured_kwargs = {}

        async def mock_stream_chat_completion(msgs, **kwargs):
            captured_kwargs.update(kwargs)
            yield LLMResponseChunk(delta_content="ok")

        engine = manager._get_engine("main", ModelEngine)
        engine.stream_chat_completion = mock_stream_chat_completion

        async for _ in manager.stream_model_call("main", messages, temperature=0.7):
            pass

        assert captured_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_raises_key_error_for_unknown_model_type(self, manager):
        """Raises KeyError when the model type doesn't exist."""
        with pytest.raises(KeyError):
            await anext(manager.stream_model_call("nonexistent", [{"role": "user", "content": "Hi"}]))


class TestEngineRegistryStreamModelCallMetrics:
    """``stream_model_call`` emits the OTEL GenAI client metrics over
    the full stream lifetime when ``metrics_enabled=True``.

    Token usage is captured from the terminal SSE chunk (which carries
    ``usage`` only when the upstream payload had
    ``stream_options.include_usage=true`` — on by default in the
    OpenAI-compatible client).  Duration is recorded around the whole
    iteration."""

    @pytest.mark.asyncio
    async def test_emits_token_usage_and_duration_on_safe_stream(self, manager_with_metrics, metric_reader):
        """Final chunk carries ``usage`` → both metrics emit.  Token
        usage produces input + output observations once the stream
        completes."""
        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.stream_chat_completion = _mock_stream(
            LLMResponseChunk(delta_content="Hello"),
            LLMResponseChunk(delta_content=" world"),
            # Terminal chunk: no content delta, just usage.
            LLMResponseChunk(usage=UsageInfo(input_tokens=12, output_tokens=2)),
        )

        chunks = [c async for c in manager_with_metrics.stream_model_call("main", [{"role": "user", "content": "hi"}])]
        assert len(chunks) == 3

        points = collect_metric_points(metric_reader)
        assert len(points["gen_ai.client.token.usage"]) == 2
        token_types = {p.attributes["gen_ai.token.type"] for p in points["gen_ai.client.token.usage"]}
        assert token_types == {"input", "output"}
        assert points["gen_ai.client.operation.duration"][0].value == 1
        assert "error.type" not in points["gen_ai.client.operation.duration"][0].attributes

    @pytest.mark.asyncio
    async def test_skips_token_usage_when_no_chunk_carries_usage(self, manager_with_metrics, metric_reader):
        """No chunk has ``usage`` populated (e.g. provider doesn't
        support ``stream_options.include_usage`` or it was suppressed
        with ``include_usage_in_stream=False``) → token metric absent,
        duration still emits."""
        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.stream_chat_completion = _mock_stream(
            LLMResponseChunk(delta_content="Hello"),
            LLMResponseChunk(delta_content=" world"),
        )

        async for _ in manager_with_metrics.stream_model_call("main", [{"role": "user", "content": "hi"}]):
            pass

        points = collect_metric_points(metric_reader)
        assert "gen_ai.client.token.usage" not in points
        assert points["gen_ai.client.operation.duration"][0].value == 1

    @pytest.mark.asyncio
    async def test_records_duration_with_error_type_on_provider_error(self, manager_with_metrics, metric_reader):
        """Provider raises mid-stream → duration emits with
        ``error.type=ExceptionClass``, token usage absent (no terminal
        chunk arrived), exception propagates to consumer."""
        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.stream_chat_completion = _mock_stream(
            LLMResponseChunk(delta_content="Hello"),
            error=RuntimeError("provider died"),
        )

        with pytest.raises(RuntimeError, match="provider died"):
            async for _ in manager_with_metrics.stream_model_call("main", [{"role": "user", "content": "hi"}]):
                pass

        points = collect_metric_points(metric_reader)
        assert "gen_ai.client.token.usage" not in points
        assert points["gen_ai.client.operation.duration"][0].attributes["error.type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_no_token_usage_on_consumer_early_break(self, manager_with_metrics, metric_reader):
        """Consumer breaks out of the iteration before the terminal
        chunk arrives → captured_usage is None at that point, and the
        ``record_token_usage`` line after the ``with`` block doesn't
        run (GeneratorExit unwinds the with-stack but skips trailing
        code).  Duration still records via the ``finally`` in
        ``llm_operation_duration``."""
        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.stream_chat_completion = _mock_stream(
            LLMResponseChunk(delta_content="Hello"),
            LLMResponseChunk(delta_content=" world"),
            LLMResponseChunk(usage=UsageInfo(input_tokens=10, output_tokens=2)),
        )

        # Consume only the first chunk and abandon the iterator.
        agen = manager_with_metrics.stream_model_call("main", [{"role": "user", "content": "hi"}])
        first = await anext(agen)
        assert first.delta_content == "Hello"
        await agen.aclose()

        points = collect_metric_points(metric_reader)
        assert "gen_ai.client.token.usage" not in points
        assert points["gen_ai.client.operation.duration"][0].value == 1

    @pytest.mark.asyncio
    async def test_no_metrics_emitted_when_metrics_disabled(self, manager, metric_reader):
        """Default config (metrics disabled) → no metrics fire even
        when usage info is present on the final chunk."""
        engine = manager._get_engine("main", ModelEngine)
        engine.stream_chat_completion = _mock_stream(
            LLMResponseChunk(delta_content="Hello"),
            LLMResponseChunk(usage=UsageInfo(input_tokens=10, output_tokens=2)),
        )

        async for _ in manager.stream_model_call("main", [{"role": "user", "content": "hi"}]):
            pass

        points = collect_metric_points(metric_reader)
        assert "gen_ai.client.token.usage" not in points
        assert "gen_ai.client.operation.duration" not in points

    @pytest.mark.asyncio
    async def test_label_set_includes_provider_model_operation(self, manager_with_metrics, metric_reader):
        """Standard OTEL labels on every observation."""
        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.stream_chat_completion = _mock_stream(
            LLMResponseChunk(delta_content="Hi"),
            LLMResponseChunk(usage=UsageInfo(input_tokens=1, output_tokens=1)),
        )

        async for _ in manager_with_metrics.stream_model_call("main", [{"role": "user", "content": "hi"}]):
            pass

        points = collect_metric_points(metric_reader)
        for point in points["gen_ai.client.token.usage"] + points["gen_ai.client.operation.duration"]:
            assert point.attributes["gen_ai.operation.name"] == "chat"
            assert point.attributes["gen_ai.provider.name"] == "nim"
            assert point.attributes["gen_ai.request.model"] == "meta/llama-3.3-70b-instruct"


class TestEngineRegistryStreamModelCallChunkTiming:
    """``stream_model_call`` emits ``gen_ai.client.operation.time_to_first_chunk``
    and ``gen_ai.client.operation.time_per_output_chunk`` for each
    content-bearing chunk yielded.  Cosmetic SSE frames (terminal usage
    chunk, role-only frames already filtered by the parser) do NOT
    contribute timing observations."""

    @pytest.mark.asyncio
    async def test_records_ttfc_and_per_chunk_for_content_stream(self, manager_with_metrics, metric_reader):
        """N content chunks → 1 TTFC observation + (N-1) per-chunk
        observations.  Terminal usage chunk does NOT add a per-chunk
        observation (its ``delta_content``/``delta_reasoning`` are
        both None)."""
        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.stream_chat_completion = _mock_stream(
            LLMResponseChunk(delta_content="Hello"),
            LLMResponseChunk(delta_content=" "),
            LLMResponseChunk(delta_content="world"),
            LLMResponseChunk(usage=UsageInfo(input_tokens=10, output_tokens=3)),
        )

        async for _ in manager_with_metrics.stream_model_call("main", [{"role": "user", "content": "hi"}]):
            pass

        points = collect_metric_points(metric_reader)
        # Exactly one TTFC observation per stream.
        assert points["gen_ai.client.operation.time_to_first_chunk"][0].value == 1
        # Three content chunks → 2 per-chunk intervals (1→2, 2→3).
        assert points["gen_ai.client.operation.time_per_output_chunk"][0].value == 2

    @pytest.mark.asyncio
    async def test_reasoning_chunks_count_as_content_for_chunk_timing(self, manager_with_metrics, metric_reader):
        """``delta_reasoning`` chunks are content-bearing for OTEL's
        purposes — they're real output that the consumer will display.
        TTFC fires on the first reasoning OR content chunk; per-chunk
        intervals are recorded between any combination."""
        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.stream_chat_completion = _mock_stream(
            LLMResponseChunk(delta_reasoning="thinking"),
            LLMResponseChunk(delta_content="Hello"),
            LLMResponseChunk(delta_reasoning=" more"),
        )

        async for _ in manager_with_metrics.stream_model_call("main", [{"role": "user", "content": "hi"}]):
            pass

        points = collect_metric_points(metric_reader)
        assert points["gen_ai.client.operation.time_to_first_chunk"][0].value == 1
        assert points["gen_ai.client.operation.time_per_output_chunk"][0].value == 2

    @pytest.mark.asyncio
    async def test_single_content_chunk_records_ttfc_only(self, manager_with_metrics, metric_reader):
        """One content chunk → 1 TTFC, 0 per-chunk intervals (no
        "between" gaps with only one chunk)."""
        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.stream_chat_completion = _mock_stream(LLMResponseChunk(delta_content="just one"))

        async for _ in manager_with_metrics.stream_model_call("main", [{"role": "user", "content": "hi"}]):
            pass

        points = collect_metric_points(metric_reader)
        assert points["gen_ai.client.operation.time_to_first_chunk"][0].value == 1
        assert "gen_ai.client.operation.time_per_output_chunk" not in points

    @pytest.mark.asyncio
    async def test_no_chunk_timing_when_no_content_chunks(self, manager_with_metrics, metric_reader):
        """Stream that yields only the terminal usage chunk (no
        content/reasoning) → neither chunk-timing metric fires.
        Operation duration still records.  Models a degenerate
        provider response."""
        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.stream_chat_completion = _mock_stream(
            LLMResponseChunk(usage=UsageInfo(input_tokens=10, output_tokens=0))
        )

        async for _ in manager_with_metrics.stream_model_call("main", [{"role": "user", "content": "hi"}]):
            pass

        points = collect_metric_points(metric_reader)
        assert "gen_ai.client.operation.time_to_first_chunk" not in points
        assert "gen_ai.client.operation.time_per_output_chunk" not in points
        # Sanity: duration still emits.
        assert points["gen_ai.client.operation.duration"][0].value == 1

    @pytest.mark.asyncio
    async def test_no_chunk_timing_when_metrics_disabled(self, manager, metric_reader):
        """``metrics_enabled=False`` → chunk-timing metrics do not fire
        even on a content-bearing stream."""
        engine = manager._get_engine("main", ModelEngine)
        engine.stream_chat_completion = _mock_stream(
            LLMResponseChunk(delta_content="Hello"),
            LLMResponseChunk(delta_content=" world"),
        )

        async for _ in manager.stream_model_call("main", [{"role": "user", "content": "hi"}]):
            pass

        points = collect_metric_points(metric_reader)
        assert "gen_ai.client.operation.time_to_first_chunk" not in points
        assert "gen_ai.client.operation.time_per_output_chunk" not in points

    @pytest.mark.asyncio
    async def test_chunk_timing_intervals_match_mocked_clock(self, manager_with_metrics, metric_reader):
        """Mock ``time.monotonic`` with a known sequence and verify the
        recorded TTFC and per-chunk values match the expected intervals
        exactly.

        Mirrors a real OpenAI-shape stream after the parser:
          - role-only first SSE chunk is dropped at the parser layer (so
            ``engine.stream_chat_completion`` doesn't yield it here)
          - three content-bearing chunks
          - terminal usage chunk (no content delta) — must NOT contribute
            a per-chunk interval

        ``time.monotonic`` is consulted six times in this code path:
          1. ``llm_operation_duration`` __enter__
          2. ``stream_model_call`` t0 (just inside ``with duration_ctx``)
          3-5. once per content-bearing chunk in the loop
          6. ``llm_operation_duration`` __exit__ finally
        """

        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.stream_chat_completion = _mock_stream(
            LLMResponseChunk(delta_content="Hello"),
            LLMResponseChunk(delta_content=" "),
            LLMResponseChunk(delta_content="world"),
            LLMResponseChunk(usage=UsageInfo(input_tokens=10, output_tokens=3)),
        )

        clock = [
            100.000,  # llm_operation_duration t0
            100.001,  # stream_model_call t0 (essentially same instant)
            100.050,  # content chunk 1 → TTFC = 100.050 - 100.001 = 0.049
            100.080,  # content chunk 2 → per-chunk = 100.080 - 100.050 = 0.030
            100.120,  # content chunk 3 → per-chunk = 100.120 - 100.080 = 0.040
            100.130,  # llm_operation_duration end → duration = 100.130 - 100.000 = 0.130
        ]

        with patch("time.monotonic", side_effect=clock):
            async for _ in manager_with_metrics.stream_model_call("main", [{"role": "user", "content": "hi"}]):
                pass

        # TTFC: from stream_model_call's t0 to first content chunk arrival.
        ttfc_sum = collect_histogram_sum(metric_reader, "gen_ai.client.operation.time_to_first_chunk")
        assert ttfc_sum == pytest.approx(0.049, abs=1e-9)

        # Per-chunk: two intervals between three content chunks.  Terminal
        # usage chunk does NOT contribute since the gating predicate
        # (``chunk.delta_content or chunk.delta_reasoning``) is false for it.
        per_chunk_sum = collect_histogram_sum(metric_reader, "gen_ai.client.operation.time_per_output_chunk")
        assert per_chunk_sum == pytest.approx(0.030 + 0.040, abs=1e-9)

        # Sanity check: duration spans the whole operation including the
        # terminal usage chunk's parser pass.
        duration_sum = collect_histogram_sum(metric_reader, "gen_ai.client.operation.duration")
        assert duration_sum == pytest.approx(0.130, abs=1e-9)

        # Per-chunk count = number of intervals = (content chunks - 1) = 2.
        per_chunk_points = collect_metric_points(metric_reader)["gen_ai.client.operation.time_per_output_chunk"]
        assert per_chunk_points[0].value == 2

    @pytest.mark.asyncio
    async def test_provider_error_after_first_chunk_records_partial_timing(self, manager_with_metrics, metric_reader):
        """Provider errors after yielding one content chunk → TTFC
        recorded (the first chunk arrived), no per-chunk interval
        (would have needed a second), duration emits with
        ``error.type``, exception propagates."""
        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.stream_chat_completion = _mock_stream(
            LLMResponseChunk(delta_content="Hello"),
            error=RuntimeError("provider died"),
        )

        with pytest.raises(RuntimeError, match="provider died"):
            async for _ in manager_with_metrics.stream_model_call("main", [{"role": "user", "content": "hi"}]):
                pass

        points = collect_metric_points(metric_reader)
        assert points["gen_ai.client.operation.time_to_first_chunk"][0].value == 1
        assert "gen_ai.client.operation.time_per_output_chunk" not in points
        assert points["gen_ai.client.operation.duration"][0].attributes["error.type"] == "RuntimeError"

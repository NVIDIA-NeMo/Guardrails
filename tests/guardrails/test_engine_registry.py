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

import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nemoguardrails.guardrails import telemetry
from nemoguardrails.guardrails.base_engine import BaseEngine
from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.guardrails.tool_schema import Toolset
from nemoguardrails.llm.call import llm_call
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.tracing import constants as tracing_constants
from nemoguardrails.tracing.constants import SystemConstants
from nemoguardrails.types import LLMModel, LLMResponse, LLMResponseChunk, UsageInfo
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
    return EngineRegistry(rails_config.models)


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
    return EngineRegistry(rails_config.models, metrics_enabled=True)


@pytest.fixture
def span_exporter():
    """Install a test-local TracerProvider + in-memory exporter and return
    ``(tracer, exporter)``.  The tracer is passed explicitly to the registry
    (no global TracerProvider is set), so there is no global state to clean
    up beyond the autouse ``reset_telemetry_singletons`` fixture."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    return tracer, exporter


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def manager_with_tracer(rails_config, span_exporter):
    """Create an EngineRegistry wired to the test tracer (metrics + content
    capture off) so LLM calls produce real spans we can read back."""
    tracer, _ = span_exporter
    return EngineRegistry(rails_config.models, tracer=tracer)


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


def _mock_sse_response(raw_chunks: list[dict]):
    """Build a mock aiohttp streaming response that emits ``raw_chunks`` as
    SSE ``data:`` frames followed by ``[DONE]``.

    Drives ModelEngine.stream_call's real ``_parse_chat_completion_chunk``
    path (rather than ``_mock_stream``'s pre-parsed chunks), so a finish-only
    SSE frame is parsed end-to-end. readline() returns one ``\\n``-terminated
    line at a time, matching aiohttp's StreamReader.
    """
    lines = [f"data: {json.dumps(chunk)}\n".encode() for chunk in raw_chunks]
    lines.append(b"data: [DONE]\n")
    line_iter = iter(lines)

    async def _readline():
        return next(line_iter, b"")

    mock_content = MagicMock()
    mock_content.readline = _readline

    mock_response = AsyncMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.status = 200
    mock_response.content = mock_content
    return mock_response


@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def _registry_with_main_params(parameters: dict, tracer):
    """Build an EngineRegistry whose ``main`` model carries ``parameters``,
    wired to ``tracer`` so model_call / stream_model_call produce real spans we
    can read back. Used by the parameter-defaults merge tests, which need a
    model with non-empty ``parameters`` (the shared NEMOGUARDS_CONFIG models
    have none)."""
    config = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "nim",
                    "model": "meta/llama-3.3-70b-instruct",
                    "parameters": parameters,
                }
            ]
        }
    )
    return EngineRegistry(config.models, tracer=tracer)


class TestEngineRegistryInit:
    """Test EngineRegistry creates engines from config."""

    def test_create_engines_for_each_model_type(self, manager):
        """Creates one engine per model type in config."""
        engine_names = set(manager._engines.keys())
        assert {"main", "content_safety", "topic_control"} == engine_names

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    def test_empty_config_creates_no_engines(self):
        """Empty models list results in no engines."""
        config = RailsConfig.from_content(config={"models": []})
        mgr = EngineRegistry(config.models)
        assert len(mgr._engines) == 0

    def test_jailbreak_detection_config_registers_no_engine(self, manager):
        """The jailbreak rail reaches its NIM over HTTP, so the registry holds no engine for it."""
        assert "jailbreak_detection" not in manager._engines


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
        """Raises TypeError when the named engine is not the expected type."""
        # Every engine the registry builds is a ModelEngine now that APIEngine is gone, so the
        # type check is only reachable by planting one -- and it still guards model_call.
        manager._engines["not_a_model"] = BaseEngine()
        with pytest.raises(TypeError, match="Engine 'not_a_model' is BaseEngine, expected ModelEngine"):
            manager._get_engine("not_a_model", ModelEngine)


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


class TestEngineRegistryModelCallSpanAttributes:
    """``model_call`` sets gen_ai.request.* and gen_ai.response.* / usage.*
    attributes on the LLM CLIENT span, independent of metrics and content
    capture."""

    @pytest.mark.asyncio
    async def test_sets_request_and_response_attributes(self, manager_with_tracer, span_exporter):
        """Populated LLMResponse + request kwargs → the finished span carries
        usage, response, and request-param attrs; gen_ai.request.stream is
        absent on the non-streaming path and total_tokens is never emitted."""
        _, exporter = span_exporter
        engine = manager_with_tracer._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(
            return_value=LLMResponse(
                content="hi there",
                model="meta/llama-3.3-70b-instruct",
                finish_reason="stop",
                request_id="chatcmpl-xyz",
                usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15, reasoning_tokens=3),
            ),
        )

        await manager_with_tracer.model_call(
            "main",
            [{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=100,
            stop=["END"],
        )

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs["gen_ai.request.temperature"] == 0.5
        assert attrs["gen_ai.request.max_tokens"] == 100
        assert list(attrs["gen_ai.request.stop_sequences"]) == ["END"]
        assert "gen_ai.request.stream" not in attrs
        assert attrs["gen_ai.response.model"] == "meta/llama-3.3-70b-instruct"
        assert attrs["gen_ai.response.id"] == "chatcmpl-xyz"
        assert list(attrs["gen_ai.response.finish_reasons"]) == ["stop"]
        assert attrs["gen_ai.usage.input_tokens"] == 10
        assert attrs["gen_ai.usage.output_tokens"] == 5
        assert attrs["gen_ai.usage.reasoning.output_tokens"] == 3
        assert "gen_ai.usage.total_tokens" not in attrs

    @pytest.mark.asyncio
    async def test_attributes_set_without_metrics_or_content_capture(self, manager_with_tracer, span_exporter):
        """The new attrs are independent of metrics and content capture: with
        both off (the manager_with_tracer default), usage/response attrs are
        still present while message-content attrs are not."""
        _, exporter = span_exporter
        engine = manager_with_tracer._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(
            return_value=LLMResponse(content="hi", usage=UsageInfo(input_tokens=2, output_tokens=1)),
        )

        await manager_with_tracer.model_call("main", [{"role": "user", "content": "hi"}])

        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["gen_ai.usage.input_tokens"] == 2
        assert attrs["gen_ai.usage.output_tokens"] == 1
        assert "gen_ai.input.messages" not in attrs
        assert "guardrails.request.input" not in attrs

    @pytest.mark.asyncio
    async def test_request_attributes_present_on_error(self, manager_with_tracer, span_exporter):
        """Request params are set before the call, so they survive on the span
        when the call raises; response/usage attrs are absent and the span is
        marked ERROR via error.type."""
        _, exporter = span_exporter
        engine = manager_with_tracer._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(side_effect=RuntimeError("provider down"))

        with pytest.raises(RuntimeError, match="provider down"):
            await manager_with_tracer.model_call("main", [{"role": "user", "content": "hi"}], temperature=0.2)

        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["gen_ai.request.temperature"] == 0.2
        assert "gen_ai.usage.input_tokens" not in attrs
        assert "gen_ai.response.model" not in attrs
        assert attrs["error.type"] == "RuntimeError"


class TestEngineRegistryParameterDefaults:
    """``model_call`` / ``stream_model_call`` merge ModelEngine.body_param_defaults
    (the model's ``parameters`` config minus transport/secret/streaming keys)
    under the per-call kwargs. Both the request body and the gen_ai.request.*
    span attrs reflect the defaults, with per-call llm_params overriding."""

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_model_call_applies_config_defaults(self, span_exporter):
        """No per-call kwargs → the model's parameter defaults populate both the
        request body and the request span attrs."""
        tracer, exporter = span_exporter
        registry = _registry_with_main_params({"temperature": 0.7, "max_tokens": 256}, tracer)
        engine = registry._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(return_value=LLMResponse(content="ok"))

        await registry.model_call("main", [{"role": "user", "content": "hi"}])

        body = engine.chat_completion.call_args[1]
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 256
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["gen_ai.request.temperature"] == 0.7
        assert attrs["gen_ai.request.max_tokens"] == 256

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_model_call_per_call_kwargs_override_defaults(self, span_exporter):
        """A per-call kwarg overrides the config default for that key; the other
        defaults are retained, in both body and span."""
        tracer, exporter = span_exporter
        registry = _registry_with_main_params({"temperature": 0.7, "max_tokens": 256}, tracer)
        engine = registry._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(return_value=LLMResponse(content="ok"))

        await registry.model_call("main", [{"role": "user", "content": "hi"}], temperature=0.1)

        body = engine.chat_completion.call_args[1]
        assert body["temperature"] == 0.1  # per-call override wins
        assert body["max_tokens"] == 256  # config default retained
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["gen_ai.request.temperature"] == 0.1
        assert attrs["gen_ai.request.max_tokens"] == 256

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_llm_params_take_precedence_over_config_parameters(self, span_exporter):
        """Precedence guard: when the same key is set in BOTH the static
        Model.parameters config AND the per-call llm_params, the per-call value
        must win in the request body actually sent to the engine. Reversing the
        merge order ({**kwargs, **defaults}, config winning) flips the sent
        temperature back to the config value and fails this test."""
        config_temperature = 0.7
        per_call_temperature = 0.1
        tracer, exporter = span_exporter
        registry = _registry_with_main_params({"temperature": config_temperature}, tracer)
        engine = registry._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(return_value=LLMResponse(content="ok"))

        await registry.model_call("main", [{"role": "user", "content": "hi"}], temperature=per_call_temperature)

        sent_body = engine.chat_completion.call_args[1]
        assert sent_body["temperature"] == per_call_temperature
        assert sent_body["temperature"] != config_temperature  # the static config default must not win
        # The span reflects the same value that was sent.
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["gen_ai.request.temperature"] == per_call_temperature

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_model_call_excludes_non_body_keys(self, span_exporter):
        """Transport (base_url/timeout) and streaming-control (stream) keys in
        parameters never reach the request body; only the sampling param does."""
        tracer, _ = span_exporter
        registry = _registry_with_main_params(
            {"base_url": "https://custom.example.com", "timeout": 5, "stream": True, "temperature": 0.5},
            tracer,
        )
        engine = registry._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(return_value=LLMResponse(content="ok"))

        await registry.model_call("main", [{"role": "user", "content": "hi"}])

        body = engine.chat_completion.call_args[1]
        assert body == {"temperature": 0.5}

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_model_call_applies_defaults_and_override(self, span_exporter):
        """Streaming path merges the same way: defaults populate the body
        (captured off the engine call) and the span attrs (with stream=True),
        and a per-call kwarg overrides its default."""
        tracer, exporter = span_exporter
        registry = _registry_with_main_params({"temperature": 0.7, "max_tokens": 256}, tracer)
        engine = registry._get_engine("main", ModelEngine)

        captured: dict = {}

        async def _capturing_stream(messages, **kwargs):  # noqa: ARG001 (signature dictated by ModelEngine)
            captured.update(kwargs)
            yield LLMResponseChunk(delta_content="hi", finish_reason="stop")

        engine.stream_chat_completion = _capturing_stream

        async for _ in registry.stream_model_call("main", [{"role": "user", "content": "hi"}], temperature=0.1):
            pass

        assert captured["temperature"] == 0.1  # per-call override wins
        assert captured["max_tokens"] == 256  # config default retained
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["gen_ai.request.stream"] is True
        assert attrs["gen_ai.request.temperature"] == 0.1
        assert attrs["gen_ai.request.max_tokens"] == 256

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_llm_params_take_precedence_over_config_parameters(self, span_exporter):
        """Streaming precedence guard: when the same key is set in BOTH the
        static Model.parameters config AND the per-call llm_params, the per-call
        value must win in what stream_model_call forwards to the engine.
        Reversing the merge order ({**kwargs, **defaults}, config winning) flips
        the streamed temperature back to the config value and fails this test."""
        config_temperature = 0.7
        per_call_temperature = 0.1
        tracer, exporter = span_exporter
        registry = _registry_with_main_params({"temperature": config_temperature}, tracer)
        engine = registry._get_engine("main", ModelEngine)

        captured: dict = {}

        async def _capturing_stream(messages, **kwargs):  # noqa: ARG001 (signature dictated by ModelEngine)
            captured.update(kwargs)
            yield LLMResponseChunk(delta_content="hi", finish_reason="stop")

        engine.stream_chat_completion = _capturing_stream

        async for _ in registry.stream_model_call(
            "main", [{"role": "user", "content": "hi"}], temperature=per_call_temperature
        ):
            pass

        assert captured["temperature"] == per_call_temperature
        assert captured["temperature"] != config_temperature  # the static config default must not win
        # The span reflects the same value that was forwarded.
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["gen_ai.request.temperature"] == per_call_temperature

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_stream_model_call_with_stream_param_does_not_raise_type_error(self, span_exporter):
        """A model whose parameters include ``stream`` drives the real
        stream_call/_prepare_request path without a duplicate-keyword TypeError
        (stream is excluded from body_param_defaults). Only the sampling param
        reaches the sent body; transport/streaming keys are dropped."""
        tracer, exporter = span_exporter
        registry = _registry_with_main_params(
            {"stream": True, "base_url": "https://custom.example.com", "temperature": 0.5}, tracer
        )
        engine = registry._get_engine("main", ModelEngine)
        engine._client = AsyncMock()
        engine._client.post = MagicMock(
            return_value=_mock_sse_response(
                [
                    {
                        "id": "chatcmpl-stream",
                        "model": "meta/llama-3.3-70b-instruct",
                        "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}],
                    },
                ]
            )
        )
        engine._running = True

        # Must not raise TypeError("got multiple values for keyword argument 'stream'").
        async for _ in registry.stream_model_call("main", [{"role": "user", "content": "hi"}]):
            pass

        sent_body = engine._client.post.call_args.kwargs["json"]
        assert sent_body["temperature"] == 0.5
        assert sent_body["stream"] is True  # set explicitly by stream_call, not from parameters
        assert "base_url" not in sent_body
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["gen_ai.request.temperature"] == 0.5
        assert attrs["gen_ai.request.stream"] is True


class TestEngineRegistryStartErrors:
    """Test EngineRegistry start() error handling and rollback."""

    @pytest.mark.asyncio
    async def test_start_rolls_back_on_engine_failure(self, manager):
        """When one engine fails to start, already-started engines are stopped."""
        failing_engine = "topic_control"

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
        failing_engine = "topic_control"
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
        ``llm_operation_duration``.

        This also pins ``stream_model_call``'s ``finally: await stream.aclose()``.
        The metric context lives one generator down, in
        ``ModelEngine.stream_from_messages``; closing this generator only unwinds
        this one, so without that explicit close the delegate stays suspended at
        its yield and the duration is not recorded until garbage collection."""
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


class TestEngineRegistryStreamModelCallSpanAttributes:
    """``stream_model_call`` sets gen_ai.request.* (including stream=True) and
    the accumulated gen_ai.response.* / usage.* attributes on the LLM CLIENT
    span, independent of metrics and content capture (both off here)."""

    @pytest.mark.asyncio
    async def test_accumulates_response_attributes_across_chunks(self, manager_with_tracer, span_exporter):
        """Response fields arrive on different chunks (model + id early,
        finish_reason + usage on the terminal chunk); the span carries the
        accumulated values plus the request params and stream=True."""
        _, exporter = span_exporter
        engine = manager_with_tracer._get_engine("main", ModelEngine)
        engine.stream_chat_completion = _mock_stream(
            LLMResponseChunk(
                delta_content="Hello",
                model="meta/llama-3.3-70b-instruct",
                request_id="chatcmpl-stream",
            ),
            LLMResponseChunk(delta_content=" world"),
            LLMResponseChunk(
                finish_reason="stop",
                usage=UsageInfo(input_tokens=8, output_tokens=4, total_tokens=12, reasoning_tokens=2),
            ),
        )

        async for _ in manager_with_tracer.stream_model_call(
            "main", [{"role": "user", "content": "hi"}], temperature=0.3, stop=["X"]
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs["gen_ai.request.stream"] is True
        assert attrs["gen_ai.request.temperature"] == 0.3
        assert list(attrs["gen_ai.request.stop_sequences"]) == ["X"]
        assert attrs["gen_ai.response.model"] == "meta/llama-3.3-70b-instruct"
        assert attrs["gen_ai.response.id"] == "chatcmpl-stream"
        assert list(attrs["gen_ai.response.finish_reasons"]) == ["stop"]
        assert attrs["gen_ai.usage.input_tokens"] == 8
        assert attrs["gen_ai.usage.output_tokens"] == 4
        assert attrs["gen_ai.usage.reasoning.output_tokens"] == 2
        assert "gen_ai.usage.total_tokens" not in attrs

    @pytest.mark.asyncio
    async def test_finish_only_sse_frame_lands_finish_reasons_on_span(self, manager_with_tracer, span_exporter):
        """End-to-end regression for the dropped finish-only frame. A real
        OpenAI-style stream delivers ``finish_reason`` in a frame with an empty
        delta and no usage, then usage in a separate empty-``choices`` frame.
        Driving the actual ``_parse_chat_completion_chunk`` (not a pre-parsed
        ``_mock_stream``), the span must still carry
        ``gen_ai.response.finish_reasons`` — restoring the ``is None``-only
        parser guard would drop the finish frame and fail this assertion."""
        _, exporter = span_exporter
        engine = manager_with_tracer._get_engine("main", ModelEngine)
        engine._client = AsyncMock()
        engine._client.post = MagicMock(
            return_value=_mock_sse_response(
                [
                    {
                        "id": "chatcmpl-stream",
                        "model": "meta/llama-3.3-70b-instruct",
                        "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
                    },
                    {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
                    # Finish-only frame: empty delta, no usage — previously dropped.
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                    # Usage arrives on a separate empty-choices frame.
                    {"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}},
                ]
            )
        )
        engine._running = True

        async for _ in manager_with_tracer.stream_model_call("main", [{"role": "user", "content": "hi"}]):
            pass

        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert list(attrs["gen_ai.response.finish_reasons"]) == ["stop"]
        assert attrs["gen_ai.response.model"] == "meta/llama-3.3-70b-instruct"
        assert attrs["gen_ai.response.id"] == "chatcmpl-stream"
        assert attrs["gen_ai.usage.input_tokens"] == 8
        assert attrs["gen_ai.usage.output_tokens"] == 4
        assert attrs["gen_ai.request.stream"] is True

    @pytest.mark.asyncio
    async def test_stream_attribute_set_even_without_usage(self, manager_with_tracer, span_exporter):
        """gen_ai.request.stream=True is set before the first chunk, so it is
        present even when no chunk carries usage; usage attrs are then absent."""
        _, exporter = span_exporter
        engine = manager_with_tracer._get_engine("main", ModelEngine)
        engine.stream_chat_completion = _mock_stream(
            LLMResponseChunk(delta_content="Hello"),
            LLMResponseChunk(delta_content=" world"),
        )

        async for _ in manager_with_tracer.stream_model_call("main", [{"role": "user", "content": "hi"}]):
            pass

        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["gen_ai.request.stream"] is True
        assert "gen_ai.usage.input_tokens" not in attrs
        assert "gen_ai.usage.output_tokens" not in attrs

    @pytest.mark.asyncio
    async def test_request_attributes_present_on_provider_error(self, manager_with_tracer, span_exporter):
        """Provider errors mid-stream → request attrs (incl. stream) survive on
        the span; the post-loop response/usage attrs are never set (even though
        a chunk carried ``model``), and the span is marked ERROR."""
        _, exporter = span_exporter
        engine = manager_with_tracer._get_engine("main", ModelEngine)
        engine.stream_chat_completion = _mock_stream(
            LLMResponseChunk(delta_content="Hello", model="meta/llama-3.3-70b-instruct"),
            error=RuntimeError("provider died"),
        )

        with pytest.raises(RuntimeError, match="provider died"):
            async for _ in manager_with_tracer.stream_model_call(
                "main", [{"role": "user", "content": "hi"}], temperature=0.9
            ):
                pass

        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["gen_ai.request.stream"] is True
        assert attrs["gen_ai.request.temperature"] == 0.9
        # Captured during iteration but not written — response attrs are set
        # only after natural exhaustion, which the error skips.
        assert "gen_ai.response.model" not in attrs
        assert "gen_ai.usage.input_tokens" not in attrs
        assert attrs["error.type"] == "RuntimeError"


class TestEngineRegistryToolDelegation:
    _TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]

    def test_parse_tools_delegates_to_model_engine(self, manager):
        toolset = manager.parse_tools("main", {"tools": self._TOOLS})
        assert isinstance(toolset, Toolset)
        assert [t.key for t in toolset.tools] == ["get_weather"]

    def test_parse_tools_no_tools_returns_empty(self, manager):
        assert manager.parse_tools("main", None).tools == ()

    def test_extract_tool_results_delegates_to_model_engine(self, manager):
        messages = [{"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "18C"}]
        results = manager.extract_tool_results("main", messages)
        assert [(r.call_id, r.name, r.content) for r in results] == [("c1", "get_weather", "18C")]

    def test_extract_tool_exchanges_delegates_to_model_engine(self, manager):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "X"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "18C"},
        ]
        exchanges = manager.extract_tool_exchanges("main", messages)
        assert len(exchanges) == 1
        calls, results = exchanges[0]
        assert [(c.id, c.function.name, c.function.arguments) for c in calls] == [("c1", "get_weather", {"city": "X"})]
        assert [r.call_id for r in results] == ["c1"]

    def test_parse_tools_unknown_engine_raises_keyerror(self, manager):
        with pytest.raises(KeyError):
            manager.parse_tools("nonexistent", {"tools": self._TOOLS})

    def test_extract_tool_results_unknown_engine_raises_keyerror(self, manager):
        with pytest.raises(KeyError):
            manager.extract_tool_results("nonexistent", [])

    def test_extract_tool_exchanges_unknown_engine_raises_keyerror(self, manager):
        with pytest.raises(KeyError):
            manager.extract_tool_exchanges("nonexistent", [])

    def test_parse_tools_non_model_engine_raises_typeerror(self, manager):
        """Tool parsing needs a ModelEngine, and says so rather than failing on a missing method."""
        manager._engines["not_a_model"] = BaseEngine()
        with pytest.raises(TypeError):
            manager.parse_tools("not_a_model", {"tools": self._TOOLS})

    def test_parse_tools_includes_tools_from_model_parameters(self):
        """Tools declared in model parameters (body_param_defaults) are included even with no per-call llm_params."""
        config = RailsConfig.from_content(
            config={
                "models": [
                    {
                        "type": "main",
                        "engine": "nim",
                        "model": "meta/llama-3.3-70b-instruct",
                        "parameters": {"tools": self._TOOLS},
                    }
                ]
            }
        )
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            mgr = EngineRegistry(config.models)
        toolset = mgr.parse_tools("main", None)
        assert [t.key for t in toolset.tools] == ["get_weather"]


class TestEngineRegistryLLMs:
    """``llms`` exposes the model engines as the ``Dict[str, LLMModel]`` that library
    rail actions index by model type."""

    def test_maps_every_model_type(self, manager):
        """One entry per configured model, keyed by Model.type."""
        assert set(manager.llms) == {"main", "content_safety", "topic_control"}

    def test_values_are_the_registered_engines(self, manager):
        """Entries are the registry's own engines, not copies, so they share lifecycle."""
        assert manager.llms["main"] is manager._get_engine("main", ModelEngine)

    def test_values_satisfy_the_llm_model_protocol(self, manager):
        """Every entry is usable wherever a library action declares ``LLMModel``."""
        assert all(isinstance(model, LLMModel) for model in manager.llms.values())


class TestEngineRegistryMessagesEntryPoint:
    """The registry always holds wire messages — every IORails entry point normalizes
    through ``IORails._convert_to_messages`` — so it calls the messages-typed core
    directly instead of routing back through the ``LLMModel`` prompt adapter."""

    _MESSAGES = [{"role": "user", "content": "Hi"}]

    @pytest.mark.asyncio
    async def test_model_call_skips_the_prompt_adapter(self, manager):
        """model_call reaches generate_from_messages with its messages and kwargs intact."""
        engine = manager._get_engine("main", ModelEngine)
        engine.generate_from_messages = AsyncMock(return_value=LLMResponse(content="ok"))

        await manager.model_call("main", self._MESSAGES, temperature=0.5)

        engine.generate_from_messages.assert_called_once_with(self._MESSAGES, temperature=0.5)

    @pytest.mark.asyncio
    async def test_stream_model_call_skips_the_prompt_adapter(self, manager):
        """stream_model_call reaches stream_from_messages and yields its chunks."""
        engine = manager._get_engine("main", ModelEngine)
        captured = {}

        async def _fake_stream(messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            yield LLMResponseChunk(delta_content="ok")

        engine.stream_from_messages = _fake_stream

        chunks = [chunk async for chunk in manager.stream_model_call("main", self._MESSAGES, temperature=0.5)]

        assert captured == {"messages": self._MESSAGES, "kwargs": {"temperature": 0.5}}
        assert [chunk.delta_content for chunk in chunks] == ["ok"]


class TestEngineRegistryProviderName:
    """``provider_name`` reads the engine's own property rather than its model config."""

    def test_returns_the_engine_provider_name(self, manager):
        """The registry and the engine agree on the provider name."""
        engine = manager._get_engine("main", ModelEngine)
        assert manager.provider_name("main") == engine.provider_name == "nim"

    def test_unknown_model_type_raises_key_error(self, manager):
        """An unconfigured model type raises rather than reporting 'unknown'."""
        with pytest.raises(KeyError):
            manager.provider_name("nonexistent")


class TestRailCallTelemetryParity:
    """A rail-shaped ``llm_call(engine, ...)`` produces the same telemetry as
    ``EngineRegistry.model_call``.

    Library rail actions reach the model through ``llm_call`` -> ``generate_async``,
    not through ``model_call``. If the LLM span and the GenAI metrics stayed in the
    registry wrapper, every migrated rail would silently stop emitting them — no
    error, no failing test, just missing telemetry in production. These tests pin the
    instrumentation to the engine so both entry points stay equivalent.
    """

    _MESSAGES = [{"role": "user", "content": "Is this safe?"}]

    @staticmethod
    def _response() -> LLMResponse:
        return LLMResponse(
            content="safe",
            model="meta/llama-3.3-70b-instruct",
            finish_reason="stop",
            request_id="chatcmpl-1",
            usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15),
        )

    @pytest.mark.asyncio
    async def test_span_matches_model_call(self, manager_with_tracer, span_exporter, reset_llm_call_context):
        """Both entry points emit one LLM CLIENT span with identical name, kind, and attributes."""
        _, exporter = span_exporter
        engine = manager_with_tracer._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(return_value=self._response())

        await manager_with_tracer.model_call("main", self._MESSAGES, temperature=0.2, stop=["END"])
        registry_span = exporter.get_finished_spans()[-1]
        exporter.clear()

        await llm_call(engine, self._MESSAGES, stop=["END"], llm_params={"temperature": 0.2})
        rail_span = exporter.get_finished_spans()[-1]

        assert rail_span.name == registry_span.name
        assert rail_span.kind == registry_span.kind
        assert dict(rail_span.attributes) == dict(registry_span.attributes)

    @pytest.mark.asyncio
    async def test_error_span_matches_model_call(self, manager_with_tracer, span_exporter, reset_llm_call_context):
        """A provider failure marks the span the same way on both paths."""
        _, exporter = span_exporter
        engine = manager_with_tracer._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(side_effect=RuntimeError("provider down"))

        with pytest.raises(RuntimeError, match="provider down"):
            await manager_with_tracer.model_call("main", self._MESSAGES)
        registry_span = exporter.get_finished_spans()[-1]
        exporter.clear()

        with pytest.raises(Exception, match="provider down"):
            await llm_call(engine, self._MESSAGES)
        rail_span = exporter.get_finished_spans()[-1]

        assert rail_span.status.status_code == registry_span.status.status_code
        assert dict(rail_span.attributes)["error.type"] == dict(registry_span.attributes)["error.type"]

    @pytest.mark.asyncio
    async def test_content_capture_matches_model_call(self, rails_config, span_exporter, reset_llm_call_context):
        """With capture on, the rail path records the same message content."""
        tracer, exporter = span_exporter
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            registry = EngineRegistry(
                rails_config.models,
                tracer=tracer,
                content_capture_enabled=True,
            )
        engine = registry._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(return_value=self._response())

        await registry.model_call("main", self._MESSAGES)
        registry_attrs = dict(exporter.get_finished_spans()[-1].attributes)
        exporter.clear()

        await llm_call(engine, self._MESSAGES)
        rail_attrs = dict(exporter.get_finished_spans()[-1].attributes)

        assert rail_attrs == registry_attrs

    @pytest.mark.asyncio
    async def test_rail_call_emits_token_usage_and_duration(
        self, manager_with_metrics, metric_reader, reset_llm_call_context
    ):
        """The GenAI metrics fire on the rail path, labelled by model, provider, and operation."""
        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(return_value=self._response())

        await llm_call(engine, self._MESSAGES)

        points = collect_metric_points(metric_reader)
        assert {p.attributes["gen_ai.token.type"] for p in points["gen_ai.client.token.usage"]} == {"input", "output"}
        assert points["gen_ai.client.operation.duration"][0].value == 1
        for point in points["gen_ai.client.token.usage"] + points["gen_ai.client.operation.duration"]:
            assert point.attributes["gen_ai.operation.name"] == "chat"
            assert point.attributes["gen_ai.provider.name"] == "nim"
            assert point.attributes["gen_ai.request.model"] == "meta/llama-3.3-70b-instruct"

    @pytest.mark.asyncio
    async def test_rail_call_records_error_type_on_failure(
        self, manager_with_metrics, metric_reader, reset_llm_call_context
    ):
        """A failed rail call still emits duration, labelled with the failure type."""
        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(side_effect=RuntimeError("provider down"))

        with pytest.raises(Exception, match="provider down"):
            await llm_call(engine, self._MESSAGES)

        points = collect_metric_points(metric_reader)
        assert "gen_ai.client.token.usage" not in points
        assert points["gen_ai.client.operation.duration"][0].attributes["error.type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_no_metrics_when_metrics_disabled(self, manager, metric_reader, reset_llm_call_context):
        """Metrics stay gated on the config flag, not on meter availability."""
        engine = manager._get_engine("main", ModelEngine)
        engine.chat_completion = AsyncMock(return_value=self._response())

        await llm_call(engine, self._MESSAGES)

        points = collect_metric_points(metric_reader)
        assert "gen_ai.client.token.usage" not in points
        assert "gen_ai.client.operation.duration" not in points

    @pytest.mark.asyncio
    async def test_stream_duration_recorded_on_consumer_early_break(self, manager_with_metrics, metric_reader):
        """Abandoning ``stream_async`` mid-stream must still record the duration.

        The adapter and the instrumented core are separate generators, so closing
        the adapter only unwinds the adapter — it has to close the core too, or
        the metric's ``finally`` waits on garbage collection.
        """
        engine = manager_with_metrics._get_engine("main", ModelEngine)
        engine.stream_chat_completion = _mock_stream(
            LLMResponseChunk(delta_content="sa"),
            LLMResponseChunk(delta_content="fe"),
            LLMResponseChunk(usage=UsageInfo(input_tokens=10, output_tokens=5)),
        )

        stream = engine.stream_async(self._MESSAGES)
        first = await anext(stream)
        assert first.delta_content == "sa"
        await stream.aclose()

        points = collect_metric_points(metric_reader)
        assert "gen_ai.client.token.usage" not in points
        assert points["gen_ai.client.operation.duration"][0].value == 1

    @pytest.mark.asyncio
    async def test_stream_span_matches_stream_model_call(self, manager_with_tracer, span_exporter):
        """The streaming rail path emits the same span as ``stream_model_call``."""
        _, exporter = span_exporter
        engine = manager_with_tracer._get_engine("main", ModelEngine)
        chunks = (
            LLMResponseChunk(delta_content="sa", model="meta/llama-3.3-70b-instruct", request_id="chatcmpl-1"),
            LLMResponseChunk(
                delta_content="fe",
                finish_reason="stop",
                usage=UsageInfo(input_tokens=10, output_tokens=5),
            ),
        )

        engine.stream_chat_completion = _mock_stream(*chunks)
        async for _ in manager_with_tracer.stream_model_call("main", self._MESSAGES, temperature=0.2):
            pass
        registry_span = exporter.get_finished_spans()[-1]
        exporter.clear()

        engine.stream_chat_completion = _mock_stream(*chunks)
        async for _ in engine.stream_async(self._MESSAGES, temperature=0.2):
            pass
        rail_span = exporter.get_finished_spans()[-1]

        assert rail_span.name == registry_span.name
        assert dict(rail_span.attributes) == dict(registry_span.attributes)

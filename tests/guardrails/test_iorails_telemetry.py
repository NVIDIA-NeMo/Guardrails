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

"""Integration tests: IORails generate_async with inline OTEL instrumentation."""

import asyncio
import copy
import json
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode, format_trace_id

from nemoguardrails.guardrails import telemetry
from nemoguardrails.guardrails.guardrails_types import REQUEST_ID_HEX_CHARS, RailResult, get_request_id
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.tracing import constants as tracing_constants
from nemoguardrails.tracing.constants import (
    GenAIAttributes,
    GuardrailsAttributes,
    OtelContentCapture,
    SystemConstants,
)
from nemoguardrails.types import LLMResponse, LLMResponseChunk, UsageInfo
from tests.guardrails.async_helpers import saturate_stream_semaphore, wait_for_queue_state
from tests.guardrails.metric_helpers import collect_histogram_sum, collect_metric_points
from tests.guardrails.test_data import NEMOGUARDS_CONFIG
from tests.guardrails.test_telemetry import _is_valid_hex_string


def _make_tracing_config():
    """NEMOGUARDS_CONFIG copy with **both** tracing and metrics enabled.

    Historical name — in the current architecture tracing and metrics are
    independent OTEL signals, and this helper sets both on so existing span
    AND metric tests share the same fixture.  For single-signal scenarios
    use :func:`_make_metrics_only_config` or :func:`_make_tracing_only_config`.
    """
    cfg = copy.deepcopy(NEMOGUARDS_CONFIG)
    cfg["tracing"] = {"enabled": True}
    cfg["metrics"] = {"enabled": True}
    return cfg


def _make_metrics_only_config():
    """Metrics enabled, tracing disabled (the independent-signals case)."""
    cfg = copy.deepcopy(NEMOGUARDS_CONFIG)
    cfg["metrics"] = {"enabled": True}
    return cfg


def _make_tracing_only_config():
    """Tracing enabled, metrics disabled."""
    cfg = copy.deepcopy(NEMOGUARDS_CONFIG)
    cfg["tracing"] = {"enabled": True}
    return cfg


def _stub_safe_pipeline(iorails, llm_response="Hello"):
    """Mock input/output rails as safe and the LLM to return *llm_response*."""
    iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
    iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content=llm_response))
    iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def tracer_from_provider(exporter):
    """Create a TracerProvider with InMemorySpanExporter and return its tracer."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test")


def _gated_generate(gate: asyncio.Event):
    """Build a stubbed ``IORails._do_generate`` that blocks on
    ``gate.wait()`` and returns a fixed response.  Used by tests that
    observe queue / worker state while a request is mid-pipeline."""

    async def _gen(messages, req_id, request_span=None, **kwargs):
        await gate.wait()
        return {"role": "assistant", "content": "done"}

    return _gen


@pytest_asyncio.fixture
async def iorails_tracing(tracer_from_provider):
    """IORails instance with tracing enabled, using a test tracer.

    Patches the module-level ``_tracer`` before constructing IORails so that
    ``IORails.__init__`` picks up the test tracer via ``get_tracer()`` and
    threads it through EngineRegistry/RailsManager/RailAction constructors.
    The ``async with`` block starts and stops the IORails-owned worker
    queue so no asyncio tasks leak past the test's event loop.
    """
    with patch.object(telemetry, "_tracer", tracer_from_provider):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            config = RailsConfig.from_content(config=_make_tracing_config())
            iorails = IORails(config)
        async with iorails:
            yield iorails


@pytest_asyncio.fixture
async def iorails_no_tracing():
    """IORails instance with default config (tracing disabled)."""
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
        iorails = IORails(RailsConfig.from_content(config=NEMOGUARDS_CONFIG))
    async with iorails:
        yield iorails


class TestGenerateAsyncWithTracing:
    @pytest.mark.asyncio
    async def test_creates_span(self, iorails_tracing, exporter):
        _stub_safe_pipeline(iorails_tracing)

        result = await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        assert result == {"role": "assistant", "content": "Hello"}
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "guardrails.request"
        assert spans[0].kind == SpanKind.SERVER

    @pytest.mark.asyncio
    async def test_span_has_required_attributes(self, iorails_tracing, exporter):
        _stub_safe_pipeline(iorails_tracing)

        await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["gen_ai.operation.name"] == "guardrails"
        assert "request.id" in attrs
        assert len(attrs["request.id"]) == REQUEST_ID_HEX_CHARS

    @pytest.mark.asyncio
    async def test_request_id_derived_from_trace_id(self, iorails_tracing, exporter):
        """The request ID visible to downstream code matches the span's trace ID suffix."""
        captured_req_id = None

        async def capture_req_id(messages):
            nonlocal captured_req_id
            captured_req_id = get_request_id()
            return RailResult(is_safe=True)

        _stub_safe_pipeline(iorails_tracing)
        iorails_tracing.rails_manager.is_input_safe = capture_req_id

        await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        spans = exporter.get_finished_spans()
        span_req_id = spans[0].attributes["request.id"]
        assert captured_req_id == span_req_id

    @pytest.mark.asyncio
    async def test_span_records_exception(self, iorails_tracing, exporter):
        _stub_safe_pipeline(iorails_tracing)
        iorails_tracing.engine_registry.model_call = AsyncMock(side_effect=RuntimeError("LLM failed"))

        with pytest.raises(RuntimeError, match="LLM failed"):
            await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.ERROR
        exception_events = [e for e in spans[0].events if e.name == "exception"]
        assert len(exception_events) == 1

    @pytest.mark.asyncio
    async def test_span_created_on_input_block(self, iorails_tracing, exporter):
        """A span is still created and completed even when input rails block."""
        iorails_tracing.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=False, reason="unsafe"))

        result = await iorails_tracing.generate_async([{"role": "user", "content": "bad"}])

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "guardrails.request"


class TestGenerateAsyncWithoutTracing:
    @pytest.mark.asyncio
    async def test_no_spans_exported(self, iorails_no_tracing, exporter):
        _stub_safe_pipeline(iorails_no_tracing)

        result = await iorails_no_tracing.generate_async([{"role": "user", "content": "hi"}])

        assert result == {"role": "assistant", "content": "Hello"}
        assert len(exporter.get_finished_spans()) == 0

    @pytest.mark.asyncio
    async def test_request_id_length_consistent_without_tracing(self, iorails_no_tracing):
        """Without tracing, request IDs are random but the same length as trace-derived IDs."""
        captured_req_id = None

        async def capture_req_id(messages):
            nonlocal captured_req_id
            captured_req_id = get_request_id()
            return RailResult(is_safe=True)

        _stub_safe_pipeline(iorails_no_tracing)
        iorails_no_tracing.rails_manager.is_input_safe = capture_req_id

        await iorails_no_tracing.generate_async([{"role": "user", "content": "hi"}])

        assert captured_req_id is not None
        assert len(captured_req_id) == REQUEST_ID_HEX_CHARS


class TestEndToEndTracing:
    """End-to-end tests that verify the full tracing lifecycle."""

    @pytest.mark.asyncio
    async def test_full_trace_lifecycle(self, iorails_tracing, exporter):
        """Verify span timing, attributes, status, and request-ID propagation
        through the entire generate_async pipeline."""
        captured = {}

        async def capturing_input_check(messages):
            captured["input_req_id"] = get_request_id()
            captured["input_messages"] = messages
            return RailResult(is_safe=True)

        async def capturing_model_call(model_name, messages, **kwargs):
            captured["llm_req_id"] = get_request_id()
            captured["llm_model"] = model_name
            return LLMResponse(content="Generated response")

        async def capturing_output_check(messages, response):
            captured["output_req_id"] = get_request_id()
            captured["output_response"] = response
            return RailResult(is_safe=True)

        iorails_tracing.rails_manager.is_input_safe = capturing_input_check
        iorails_tracing.engine_registry.model_call = capturing_model_call
        iorails_tracing.rails_manager.is_output_safe = capturing_output_check

        messages = [{"role": "user", "content": "hello"}]
        result = await iorails_tracing.generate_async(messages)

        # Response correctness
        assert result == {"role": "assistant", "content": "Generated response"}

        # Span structure
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "guardrails.request"
        assert span.kind == SpanKind.SERVER
        assert span.status.status_code == StatusCode.UNSET

        # Timing sanity
        assert span.start_time > 0
        assert span.end_time >= span.start_time

        # Attributes
        attrs = dict(span.attributes)
        assert attrs["gen_ai.operation.name"] == "guardrails"
        req_id = attrs["request.id"]
        assert _is_valid_hex_string(req_id, REQUEST_ID_HEX_CHARS)

        # Request ID derived from trace ID
        full_trace_id = format_trace_id(span.context.trace_id)
        assert full_trace_id.endswith(req_id)

        # Same request ID visible at every pipeline stage
        assert captured["input_req_id"] == req_id
        assert captured["llm_req_id"] == req_id
        assert captured["output_req_id"] == req_id

        # Pipeline received correct data
        assert captured["input_messages"] == messages
        assert captured["llm_model"] == "main"
        assert captured["output_response"] == "Generated response"

    @pytest.mark.asyncio
    async def test_concurrent_requests_get_distinct_traces(self, iorails_tracing, exporter):
        """Two concurrent generate_async calls produce separate spans
        with distinct trace IDs and request IDs."""
        req_ids_seen = []

        async def record_req_id(messages):
            req_ids_seen.append(get_request_id())
            return RailResult(is_safe=True)

        _stub_safe_pipeline(iorails_tracing)
        iorails_tracing.rails_manager.is_input_safe = record_req_id

        messages = [{"role": "user", "content": "hi"}]
        await asyncio.gather(
            iorails_tracing.generate_async(messages),
            iorails_tracing.generate_async(messages),
        )

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        trace_ids = {s.context.trace_id for s in spans}
        assert len(trace_ids) == 2

        span_req_ids = {s.attributes["request.id"] for s in spans}
        assert len(span_req_ids) == 2

        assert set(req_ids_seen) == span_req_ids

    @pytest.mark.asyncio
    async def test_error_span_preserves_full_context(self, iorails_tracing, exporter):
        """On LLM failure: span records the exception type, message,
        and traceback, and the request ID is still valid."""
        captured_req_id = None

        async def capture_then_pass(messages):
            nonlocal captured_req_id
            captured_req_id = get_request_id()
            return RailResult(is_safe=True)

        iorails_tracing.rails_manager.is_input_safe = capture_then_pass
        iorails_tracing.engine_registry.model_call = AsyncMock(side_effect=RuntimeError("connection refused"))

        with pytest.raises(RuntimeError, match="connection refused"):
            await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        # Status
        assert span.status.status_code == StatusCode.ERROR
        assert "connection refused" in span.status.description

        # Exception event
        exc_events = [e for e in span.events if e.name == "exception"]
        assert len(exc_events) == 1
        assert exc_events[0].attributes["exception.type"] == "RuntimeError"
        assert "connection refused" in exc_events[0].attributes["exception.message"]
        assert "exception.stacktrace" in exc_events[0].attributes

        # Request ID was still set correctly before the error
        assert captured_req_id == span.attributes["request.id"]

        # Span still has valid timing
        assert span.end_time >= span.start_time


SAFE_INPUT_JSON = json.dumps({"User Safety": "safe"})
SAFE_OUTPUT_JSON = json.dumps({"User Safety": "safe", "Response Safety": "safe"})
UNSAFE_INPUT_JSON = json.dumps({"User Safety": "unsafe", "Safety Categories": "S1: Violence"})


def _stub_deep_pipeline(iorails, main_llm_response="Hello", input_safe=True):
    """Mock at the engine level so the full RailsManager → RailAction → EngineRegistry
    chain executes (including span creation), but actual HTTP calls are skipped.

    Mocks ModelEngine.chat_completion and APIEngine.call on each registered engine.
    The content_safety engine returns different JSON for input vs output checks —
    we use SAFE_INPUT_JSON as default since the output rail's parser also accepts it
    when Response Safety is absent (it just checks User Safety).
    """
    from nemoguardrails.guardrails.api_engine import APIEngine
    from nemoguardrails.guardrails.model_engine import ModelEngine

    input_json = SAFE_INPUT_JSON if input_safe else UNSAFE_INPUT_JSON
    for name, engine in iorails.engine_registry._engines.items():
        if isinstance(engine, ModelEngine):
            if name == "main":
                engine.chat_completion = AsyncMock(return_value=LLMResponse(content=main_llm_response))
            elif name == "content_safety":
                # Content safety output parser needs Response Safety field
                engine.chat_completion = AsyncMock(
                    return_value=LLMResponse(content=SAFE_OUTPUT_JSON if input_safe else input_json)
                )
            else:
                engine.chat_completion = AsyncMock(return_value=LLMResponse(content=input_json))
        elif isinstance(engine, APIEngine):
            engine.call = AsyncMock(return_value={"jailbreak": False, "score": 0.01})


class TestSpanHierarchy:
    """Tests that verify parent-child span relationships across the full pipeline."""

    @pytest.mark.asyncio
    async def test_span_hierarchy_on_safe_request(self, iorails_tracing, exporter):
        """Full safe request produces: request → rail → action → LLM/API spans."""
        _stub_deep_pipeline(iorails_tracing)

        result = await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])
        assert result["content"] == "Hello"

        spans = exporter.get_finished_spans()

        # Find the root request span
        request_spans = [s for s in spans if s.name == "guardrails.request"]
        assert len(request_spans) == 1
        root = request_spans[0]
        root_ctx = root.context

        # Rail spans should exist for input and output rails
        rail_spans = [s for s in spans if s.name == "guardrails.rail"]
        # 3 input rails + 1 output rail = 4
        assert len(rail_spans) == 4

        # Action spans should be nested under rail spans
        action_spans = [s for s in spans if s.name == "guardrails.action"]
        assert len(action_spans) == 4

        # LLM call spans (content_safety input, topic_safety input, content_safety output, main LLM)
        llm_spans = [s for s in spans if s.kind == SpanKind.CLIENT]
        assert len(llm_spans) >= 4  # at least 3 rail LLMs + 1 API + 1 main

        # All rail spans are children of the request span
        for rail_span in rail_spans:
            assert rail_span.parent.trace_id == root_ctx.trace_id

    @pytest.mark.asyncio
    async def test_rail_span_attributes(self, iorails_tracing, exporter):
        """Rail spans have correct rail.type and rail.name attributes."""
        _stub_deep_pipeline(iorails_tracing)

        await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        spans = exporter.get_finished_spans()
        rail_spans = [s for s in spans if s.name == "guardrails.rail"]

        input_rails = [s for s in rail_spans if s.attributes["rail.type"] == "Input"]
        output_rails = [s for s in rail_spans if s.attributes["rail.type"] == "Output"]
        assert len(input_rails) == 3
        assert len(output_rails) == 1

        rail_names = {s.attributes["rail.name"] for s in rail_spans}
        assert "content safety check input $model=content_safety" in rail_names
        assert "topic safety check input $model=topic_control" in rail_names
        assert "jailbreak detection model" in rail_names
        assert "content safety check output $model=content_safety" in rail_names

    @pytest.mark.asyncio
    async def test_rail_stop_attribute_on_block(self, iorails_tracing, exporter):
        """When a rail blocks, its span has rail.stop=True."""
        _stub_deep_pipeline(iorails_tracing, input_safe=False)

        result = await iorails_tracing.generate_async([{"role": "user", "content": "bad"}])
        assert result["content"] == REFUSAL_MESSAGE

        spans = exporter.get_finished_spans()
        rail_spans = [s for s in spans if s.name == "guardrails.rail"]
        # First rail blocked → should have rail.stop
        blocked = [s for s in rail_spans if s.attributes.get("rail.stop") is True]
        assert len(blocked) >= 1

    @pytest.mark.asyncio
    async def test_span_hierarchy_on_unsafe_request(self, iorails_tracing, exporter):
        """Unsafe request: main LLM and output rails never run, request span still completes cleanly."""
        _stub_deep_pipeline(iorails_tracing, input_safe=False)

        result = await iorails_tracing.generate_async([{"role": "user", "content": "bad"}])
        assert result["content"] == REFUSAL_MESSAGE

        spans = exporter.get_finished_spans()

        # Request span still completes without error (blocking is not an exception)
        request_spans = [s for s in spans if s.name == "guardrails.request"]
        assert len(request_spans) == 1
        assert request_spans[0].status.status_code == StatusCode.UNSET

        # Output rail spans must be absent (pipeline short-circuits before Step 3)
        rail_spans = [s for s in spans if s.name == "guardrails.rail"]
        output_rails = [s for s in rail_spans if s.attributes["rail.type"] == "Output"]
        assert output_rails == []

        # Main LLM span must be absent — the meta/llama main model is never called
        main_llm_spans = [
            s
            for s in spans
            if s.kind == SpanKind.CLIENT and s.attributes.get("gen_ai.request.model", "").startswith("meta/llama")
        ]
        assert main_llm_spans == []

        # The blocking rail's span is marked with rail.stop=True
        blocked = [s for s in rail_spans if s.attributes.get("rail.stop") is True]
        assert len(blocked) >= 1

    @pytest.mark.asyncio
    async def test_action_span_records_engine_error(self, iorails_tracing, exporter):
        """When the engine raises, the action span must record it (not swallow it)."""
        from nemoguardrails.guardrails.model_engine import ModelEngine

        # Make the content_safety engine fail — RailAction.run will catch and
        # convert to RailResult(is_safe=False), but the action span must still
        # reflect the error.
        for name, engine in iorails_tracing.engine_registry._engines.items():
            if isinstance(engine, ModelEngine) and name == "content_safety":
                engine.chat_completion = AsyncMock(side_effect=RuntimeError("LLM down"))
            elif isinstance(engine, ModelEngine):
                engine.chat_completion = AsyncMock(return_value=LLMResponse(content=SAFE_INPUT_JSON))

        result = await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])
        assert result["content"] == REFUSAL_MESSAGE

        spans = exporter.get_finished_spans()
        action_spans = [s for s in spans if s.name == "guardrails.action"]
        content_safety_action = next(
            s for s in action_spans if s.attributes["action.name"] == "content safety check input"
        )

        # Span has ERROR status and an exception event recording the RuntimeError
        assert content_safety_action.status.status_code == StatusCode.ERROR
        exc_events = [e for e in content_safety_action.events if e.name == "exception"]
        assert len(exc_events) == 1
        assert exc_events[0].attributes["exception.type"] == "RuntimeError"
        assert "LLM down" in exc_events[0].attributes["exception.message"]

    @pytest.mark.asyncio
    async def test_span_tree_parent_child_links(self, iorails_tracing, exporter):
        """Verify strict parent-child links across the full safe-path span tree.

        Uses sequential rail execution (NEMOGUARDS_CONFIG default). Every span
        must (a) share the same trace_id, (b) have a unique span_id, and
        (c) point to a valid ancestor per the expected shape:

            guardrails.request (SERVER, parent=None)
              ├─ guardrails.rail [Input]   → parent=request
              │   └─ guardrails.action     → parent=its rail
              │       └─ CLIENT span       → parent=its action
              ├─ chat <main model>  (CLIENT, parent=request)
              └─ guardrails.rail [Output]  → parent=request
                  └─ guardrails.action     → parent=its rail
                      └─ CLIENT span       → parent=its action
        """
        _stub_deep_pipeline(iorails_tracing)

        await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        spans = exporter.get_finished_spans()

        # All spans share a single trace_id
        trace_ids = {s.context.trace_id for s in spans}
        assert len(trace_ids) == 1

        # All span_ids are unique
        span_ids = [s.context.span_id for s in spans]
        assert len(span_ids) == len(set(span_ids))

        # Index spans by ID for parent lookup
        by_id = {s.context.span_id: s for s in spans}

        # Exactly one root — the request span — with parent=None
        roots = [s for s in spans if s.parent is None]
        assert len(roots) == 1
        request_span = roots[0]
        assert request_span.name == "guardrails.request"
        assert request_span.kind == SpanKind.SERVER

        # Every non-root span's parent resolves to another span in the same trace
        for span in spans:
            if span.parent is None:
                continue
            assert span.parent.trace_id == request_span.context.trace_id
            assert span.parent.span_id in by_id, f"{span.name}'s parent not in trace"

        # Each guardrails.rail is a direct child of the request span
        rail_spans = [s for s in spans if s.name == "guardrails.rail"]
        assert len(rail_spans) == 4  # 3 input + 1 output
        for rail in rail_spans:
            assert rail.parent.span_id == request_span.context.span_id

        # Each guardrails.action is a direct child of exactly one rail
        action_spans = [s for s in spans if s.name == "guardrails.action"]
        assert len(action_spans) == 4
        rail_ids = {r.context.span_id for r in rail_spans}
        for action in action_spans:
            assert action.parent.span_id in rail_ids, (
                f"action '{action.attributes['action.name']}' parent is not a rail span"
            )

        # Each CLIENT span is either the main LLM (parent=request)
        # or a rail-LLM/API call (parent=one of the action spans)
        client_spans = [s for s in spans if s.kind == SpanKind.CLIENT]
        action_ids = {a.context.span_id for a in action_spans}
        main_llm_spans = []
        rail_call_spans = []
        for client in client_spans:
            if client.parent.span_id == request_span.context.span_id:
                main_llm_spans.append(client)
            elif client.parent.span_id in action_ids:
                rail_call_spans.append(client)
            else:
                raise AssertionError(f"CLIENT span '{client.name}' has unexpected parent")

        # Exactly one main LLM call, and one CLIENT span per action
        assert len(main_llm_spans) == 1
        assert main_llm_spans[0].attributes["gen_ai.request.model"] == "meta/llama-3.3-70b-instruct"
        assert len(rail_call_spans) == len(action_spans)

    @pytest.mark.asyncio
    async def test_action_span_attributes(self, iorails_tracing, exporter):
        """Action spans have correct action.name attributes."""
        _stub_deep_pipeline(iorails_tracing)

        await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        spans = exporter.get_finished_spans()
        action_spans = [s for s in spans if s.name == "guardrails.action"]
        action_names = {s.attributes["action.name"] for s in action_spans}
        assert "content safety check input" in action_names
        assert "topic safety check input" in action_names
        assert "jailbreak detection model" in action_names
        assert "content safety check output" in action_names

    @pytest.mark.asyncio
    async def test_llm_span_attributes(self, iorails_tracing, exporter):
        """LLM spans have GenAI semantic convention attributes."""
        _stub_deep_pipeline(iorails_tracing)

        await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        spans = exporter.get_finished_spans()
        llm_spans = [s for s in spans if "gen_ai.request.model" in (s.attributes or {})]
        models_seen = {s.attributes["gen_ai.request.model"] for s in llm_spans}

        # Every LLM span from the NEMOGUARDS_CONFIG pipeline must have its model recorded
        assert "nvidia/llama-3.1-nemoguard-8b-content-safety" in models_seen
        assert "nvidia/llama-3.1-nemoguard-8b-topic-control" in models_seen
        assert "meta/llama-3.3-70b-instruct" in models_seen

    @pytest.mark.asyncio
    async def test_no_child_spans_when_tracing_disabled(self, iorails_no_tracing, exporter):
        """With tracing disabled, no spans at all are created.

        Uses ``_stub_deep_pipeline`` so the full RailsManager → RailAction →
        EngineRegistry chain executes.  This exercises the code paths that
        would otherwise create orphaned child spans, not just the top-level
        IORails entry point.
        """
        _stub_deep_pipeline(iorails_no_tracing)

        await iorails_no_tracing.generate_async([{"role": "user", "content": "hi"}])

        assert len(exporter.get_finished_spans()) == 0

    @pytest.mark.asyncio
    async def test_no_orphaned_child_spans_with_global_provider(self, tracer_from_provider, exporter):
        """Regression: with a working TracerProvider wired to an exporter but
        ``config.tracing.enabled=False``, no child spans must leak through.

        Previously ``EngineRegistry``/``RailsManager``/``RailAction`` called
        ``get_tracer()`` directly, which would return a real tracer whenever
        the host app had a global provider — producing orphaned rail/action/LLM
        spans with no parent ``guardrails.request`` span.  Threading the tracer
        through constructors means every helper uses ``self._tracer``, which
        is ``None`` when tracing is disabled.
        """
        # Install the exporter-backed tracer as the module singleton, so any
        # accidental get_tracer() call in the pipeline would emit to it.
        with patch.object(telemetry, "_tracer", tracer_from_provider):
            with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
                # tracing NOT enabled in this config
                iorails = IORails(RailsConfig.from_content(config=NEMOGUARDS_CONFIG))

            async with iorails:
                _stub_deep_pipeline(iorails)
                await iorails.generate_async([{"role": "user", "content": "hi"}])

        # Zero spans of any kind
        assert exporter.get_finished_spans() == ()


_INPUT_ONLY_STREAMING_TRACING_CONFIG = {
    **NEMOGUARDS_CONFIG,
    "rails": {
        **NEMOGUARDS_CONFIG["rails"],
        "output": {"flows": []},
    },
    # Both signals enabled: streaming-span tests and streaming-metric tests
    # share this config.  See docstring on ``_make_tracing_config`` for the
    # tracing-vs-metrics independence rationale.
    "tracing": {"enabled": True},
    "metrics": {"enabled": True},
}

_INPUT_ONLY_STREAMING_NO_TRACING_CONFIG = {
    **NEMOGUARDS_CONFIG,
    "rails": {
        **NEMOGUARDS_CONFIG["rails"],
        "output": {"flows": []},
    },
}


def _make_output_streaming_tracing_config(*, stream_first=True):
    """Config with output-rail streaming + both telemetry signals enabled."""
    base = copy.deepcopy(NEMOGUARDS_CONFIG)
    base["rails"]["output"]["streaming"] = {
        "enabled": True,
        "chunk_size": 5,
        "context_size": 2,
        "stream_first": stream_first,
    }
    base["tracing"] = {"enabled": True}
    base["metrics"] = {"enabled": True}
    return base


async def _mock_chunks_stream(model_type, messages, **kwargs):
    """stream_model_call-level mock yielding three LLMResponseChunk objects."""
    for text in ["Hello", " ", "world"]:
        yield LLMResponseChunk(delta_content=text)


async def _engine_default_stream(messages, **kwargs):
    """ModelEngine.stream_chat_completion-level mock for the main LLM."""
    for text in ["Hello", " from", " the", " stream"]:
        yield LLMResponseChunk(delta_content=text)


async def _engine_failing_stream(messages, **kwargs):
    """ModelEngine.stream_chat_completion-level mock that raises mid-stream."""
    yield LLMResponseChunk(delta_content="Hello")
    raise RuntimeError("stream broke")


def _stub_deep_streaming_pipeline(iorails, main_stream=None, input_safe=True):
    """Engine-level mocks for the streaming path.

    Non-main engines still use ``chat_completion`` (rails are non-streaming);
    the main engine uses ``stream_chat_completion`` so the LLM span in
    ``stream_model_call`` sees the real wrapper code.
    """
    from nemoguardrails.guardrails.api_engine import APIEngine
    from nemoguardrails.guardrails.model_engine import ModelEngine

    if main_stream is None:
        main_stream = _engine_default_stream

    input_json = SAFE_INPUT_JSON if input_safe else UNSAFE_INPUT_JSON
    for name, engine in iorails.engine_registry._engines.items():
        if isinstance(engine, ModelEngine):
            if name == "main":
                engine.stream_chat_completion = main_stream
            elif name == "content_safety":
                engine.chat_completion = AsyncMock(
                    return_value=LLMResponse(content=SAFE_OUTPUT_JSON if input_safe else input_json)
                )
            else:
                engine.chat_completion = AsyncMock(return_value=LLMResponse(content=input_json))
        elif isinstance(engine, APIEngine):
            engine.call = AsyncMock(return_value={"jailbreak": False, "score": 0.01})


@pytest_asyncio.fixture
async def iorails_streaming_input_only_tracing(tracer_from_provider):
    """Input-rails only + streaming + tracing enabled."""
    with patch.object(telemetry, "_tracer", tracer_from_provider):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            config = RailsConfig.from_content(config=_INPUT_ONLY_STREAMING_TRACING_CONFIG)
            iorails = IORails(config)
        async with iorails:
            yield iorails


@pytest_asyncio.fixture
async def iorails_streaming_output_tracing(tracer_from_provider):
    """Full input+output streaming + tracing enabled."""
    with patch.object(telemetry, "_tracer", tracer_from_provider):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            config = RailsConfig.from_content(config=_make_output_streaming_tracing_config())
            iorails = IORails(config)
        async with iorails:
            yield iorails


@pytest_asyncio.fixture
async def iorails_streaming_no_tracing():
    """Input-rails only + streaming + tracing disabled."""
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
        iorails = IORails(RailsConfig.from_content(config=_INPUT_ONLY_STREAMING_NO_TRACING_CONFIG))
    async with iorails:
        yield iorails


class TestStreamAsyncSpanHierarchy:
    """Inline OTEL instrumentation for stream_async."""

    @pytest.mark.asyncio
    async def test_creates_request_span(self, iorails_streaming_input_only_tracing, exporter):
        iorails = iorails_streaming_input_only_tracing
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.engine_registry.stream_model_call = _mock_chunks_stream

        chunks = [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]
        assert chunks == ["Hello", " ", "world"]

        request_spans = [s for s in exporter.get_finished_spans() if s.name == "guardrails.request"]
        assert len(request_spans) == 1
        assert request_spans[0].kind == SpanKind.SERVER
        assert request_spans[0].status.status_code == StatusCode.UNSET

    @pytest.mark.asyncio
    async def test_context_propagation_across_create_task(self, iorails_streaming_input_only_tracing, exporter):
        """Rails run inside _generation_task attach as children of the request span.

        This is the core PR3 correctness property: ``asyncio.create_task`` must
        snapshot OTEL context from ``_wrapped_iterator`` (where the request span
        is current) so downstream rail/action/LLM spans parent correctly.
        """
        iorails = iorails_streaming_input_only_tracing
        _stub_deep_streaming_pipeline(iorails)

        [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]

        spans = exporter.get_finished_spans()
        root = next(s for s in spans if s.name == "guardrails.request")

        rail_spans = [s for s in spans if s.name == "guardrails.rail"]
        assert len(rail_spans) == 3  # 3 input rails, no output rails

        for r in rail_spans:
            assert r.parent.span_id == root.context.span_id, (
                f"rail span {r.attributes.get('rail.name')} not parented under request span"
            )

    @pytest.mark.asyncio
    async def test_streaming_llm_span_has_genai_attributes(self, iorails_streaming_input_only_tracing, exporter):
        """stream_model_call produces a CLIENT span with GenAI attributes."""
        iorails = iorails_streaming_input_only_tracing
        _stub_deep_streaming_pipeline(iorails)

        [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]

        spans = exporter.get_finished_spans()
        main_llm_spans = [
            s
            for s in spans
            if s.kind == SpanKind.CLIENT and s.attributes.get("gen_ai.request.model") == "meta/llama-3.3-70b-instruct"
        ]
        assert len(main_llm_spans) == 1
        attrs = dict(main_llm_spans[0].attributes)
        assert attrs["gen_ai.operation.name"] == "chat"
        assert attrs["gen_ai.provider.name"] == "nim"
        assert main_llm_spans[0].name == "chat meta/llama-3.3-70b-instruct"

    @pytest.mark.asyncio
    async def test_llm_stream_error_recorded_on_both_spans(self, iorails_streaming_input_only_tracing, exporter):
        """Stream failure mid-generation → LLM span and request span both ERROR."""
        iorails = iorails_streaming_input_only_tracing
        _stub_deep_streaming_pipeline(iorails, main_stream=_engine_failing_stream)

        chunks = [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]
        # The generation task swallowed the exception and pushed an error payload
        assert any(c.startswith('{"error"') for c in chunks)

        spans = exporter.get_finished_spans()

        request_span = next(s for s in spans if s.name == "guardrails.request")
        assert request_span.status.status_code == StatusCode.ERROR
        assert request_span.attributes["error.type"] == "RuntimeError"

        main_llm_span = next(
            s
            for s in spans
            if s.kind == SpanKind.CLIENT and s.attributes.get("gen_ai.request.model") == "meta/llama-3.3-70b-instruct"
        )
        assert main_llm_span.status.status_code == StatusCode.ERROR
        assert main_llm_span.attributes["error.type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_input_block_leaves_llm_span_absent(self, iorails_streaming_input_only_tracing, exporter):
        """When input rails block, no main LLM span is created."""
        iorails = iorails_streaming_input_only_tracing
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=False, reason="blocked"))

        chunks = [c async for c in iorails.stream_async([{"role": "user", "content": "bad"}])]
        assert chunks == [REFUSAL_MESSAGE]

        spans = exporter.get_finished_spans()
        request_spans = [s for s in spans if s.name == "guardrails.request"]
        assert len(request_spans) == 1
        assert request_spans[0].status.status_code == StatusCode.UNSET

        main_llm_spans = [
            s
            for s in spans
            if s.kind == SpanKind.CLIENT and s.attributes.get("gen_ai.request.model") == "meta/llama-3.3-70b-instruct"
        ]
        assert main_llm_spans == []

    @pytest.mark.asyncio
    async def test_output_rails_produce_per_batch_spans(self, iorails_streaming_output_tracing, exporter):
        """Each output-rail batch produces a rail-span subtree under the request span."""
        iorails = iorails_streaming_output_tracing
        _stub_deep_streaming_pipeline(iorails)

        [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]

        spans = exporter.get_finished_spans()
        root = next(s for s in spans if s.name == "guardrails.request")

        rail_spans = [s for s in spans if s.name == "guardrails.rail"]
        output_rails = [s for s in rail_spans if s.attributes["rail.type"] == "Output"]
        # At least one output-rail batch was checked (likely more given 4-token stream
        # and chunk_size=5).
        assert len(output_rails) >= 1
        for r in output_rails:
            assert r.parent.span_id == root.context.span_id

    @pytest.mark.asyncio
    async def test_no_spans_when_tracing_disabled(self, iorails_streaming_no_tracing, exporter):
        """With tracing off, streaming produces zero spans but still works."""
        iorails = iorails_streaming_no_tracing
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.engine_registry.stream_model_call = _mock_chunks_stream

        chunks = [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]
        assert chunks == ["Hello", " ", "world"]
        assert exporter.get_finished_spans() == ()

    @pytest.mark.asyncio
    async def test_queue_full_creates_no_spans(self, iorails_streaming_input_only_tracing, exporter):
        """Load-shed rejections happen before tracing starts — no spans leak."""
        iorails = iorails_streaming_input_only_tracing
        # Force all slots unavailable.
        iorails._stream_semaphore = asyncio.Semaphore(0)

        with pytest.raises(asyncio.QueueFull):
            [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]

        assert exporter.get_finished_spans() == ()

    @pytest.mark.asyncio
    async def test_stream_failure_does_not_pollute_ambient_span_when_tracing_disabled(
        self, tracer_from_provider, exporter
    ):
        """Regression: when IORails tracing is OFF but the host app has an
        active OTEL span (e.g. a FastAPI/gRPC service span), a streaming
        failure must NOT mark that ambient span ERROR.

        Without the ``self._tracing_enabled`` guard in ``_generation_task``,
        ``trace.get_current_span()`` returns the caller's ambient span
        (captured by ``asyncio.create_task``'s context snapshot), and the
        error from the swallowed stream exception would silently corrupt
        unrelated traces in production.
        """
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            # Tracing NOT enabled in the IORails config.
            iorails = IORails(RailsConfig.from_content(config=_INPUT_ONLY_STREAMING_NO_TRACING_CONFIG))

        async with iorails:
            _stub_deep_streaming_pipeline(iorails, main_stream=_engine_failing_stream)

            # Open an ambient span as the host application would.
            with tracer_from_provider.start_as_current_span("host.ambient") as ambient_span:
                chunks = [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]

            # Stream failure was still converted to an error payload for the consumer.
            assert any(c.startswith('{"error"') for c in chunks)

            spans = exporter.get_finished_spans()

            # Exactly one span exported — the ambient host span. No guardrails spans
            # (tracing off) and no orphaned child spans.
            assert len(spans) == 1
            assert spans[0].name == "host.ambient"
            assert spans[0].context.span_id == ambient_span.context.span_id

            # Ambient span was NOT polluted by the streaming failure.
            assert spans[0].status.status_code == StatusCode.UNSET
            assert "error.type" not in dict(spans[0].attributes)
            assert [e for e in spans[0].events if e.name == "exception"] == []


class TestOtelNotInstalled:
    @pytest.mark.asyncio
    async def test_falls_back_gracefully(self, exporter):
        """When OTEL is not available, IORails works without tracing."""
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._tracer = None
            with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
                config = RailsConfig.from_content(config=_make_tracing_config())
                iorails = IORails(config)

            async with iorails:
                _stub_safe_pipeline(iorails)

                result = await iorails.generate_async([{"role": "user", "content": "hi"}])

                assert result == {"role": "assistant", "content": "Hello"}
                assert iorails._tracing_enabled is False
                assert len(exporter.get_finished_spans()) == 0


@pytest.fixture(autouse=True)
def reset_telemetry_singletons():
    """Reset telemetry's module-level singletons before and after every
    test in this file.

    Without this, tests that exercise the OTEL emission paths (e.g.
    anything that constructs IORails from a metrics-enabled config and
    triggers an LLM call) leave ``_meter`` / ``_request_instruments`` /
    ``_llm_instruments`` / ``_tracer`` populated.  That state would
    otherwise survive into later test files (e.g. the LLMRails
    OpenTelemetry adapter tests) and produce ghost data points,
    stale-meter bindings, or spurious warnings — a class of bug
    that's painful to chase because it only manifests when test
    ordering changes.

    Cheap (four ``None`` assignments per test) and matches the
    pattern used in ``test_telemetry_metrics.py`` and
    ``test_engine_registry.py``.
    """
    telemetry._meter = None
    telemetry._request_instruments = None
    tracing_constants._llm_instruments = None
    telemetry._tracer = None
    yield
    telemetry._meter = None
    telemetry._request_instruments = None
    tracing_constants._llm_instruments = None
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


class TestGenerateAsyncRequestMetrics:
    """Non-streaming path emits ``guardrails.requests`` /
    ``guardrails.request.duration`` / ``guardrails.requests.errors`` and
    nets ``guardrails.requests.active`` to zero on completion."""

    @pytest.mark.asyncio
    async def test_emits_counter_and_duration_on_safe_request(self, iorails_tracing, metric_reader):
        """Happy-path generate_async → counter +1, duration recorded once,
        no errors, requests.active back to 0."""
        _stub_safe_pipeline(iorails_tracing)

        await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        points = collect_metric_points(metric_reader)
        assert points["guardrails.requests"][0].value == 1
        # Histogram value is the recording count, not the duration itself.
        assert points["guardrails.request.duration"][0].value == 1
        assert "guardrails.requests.errors" not in points
        # Aggregate saturation counter nets to 0 after a completed request.
        assert points["guardrails.requests.active"][0].value == 0

    @pytest.mark.asyncio
    async def test_emits_errors_counter_on_exception(self, iorails_tracing, metric_reader):
        _stub_safe_pipeline(iorails_tracing)
        iorails_tracing.engine_registry.model_call = AsyncMock(side_effect=RuntimeError("LLM failed"))

        with pytest.raises(RuntimeError, match="LLM failed"):
            await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        points = collect_metric_points(metric_reader)
        assert points["guardrails.requests"][0].value == 1
        assert points["guardrails.requests.errors"][0].value == 1
        assert points["guardrails.requests.errors"][0].attributes["error.type"] == "RuntimeError"
        # Duration still recorded for an errored request.
        assert points["guardrails.request.duration"][0].value == 1

    @pytest.mark.asyncio
    async def test_no_metrics_emitted_when_metrics_disabled(self, iorails_no_tracing, metric_reader):
        _stub_safe_pipeline(iorails_no_tracing)

        await iorails_no_tracing.generate_async([{"role": "user", "content": "hi"}])

        points = collect_metric_points(metric_reader)
        assert points == {}

    @pytest.mark.asyncio
    async def test_emits_blocked_counter_on_input_block(self, iorails_tracing, metric_reader):
        _stub_safe_pipeline(iorails_tracing)
        iorails_tracing.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=False, reason="unsafe"))

        result = await iorails_tracing.generate_async([{"role": "user", "content": "bad"}])

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        points = collect_metric_points(metric_reader)
        assert points["guardrails.requests.blocked"][0].value == 1
        assert points["guardrails.requests.blocked"][0].attributes["rail.type"] == "Input"
        # No LLM call, so no errors; duration still recorded.
        assert "guardrails.requests.errors" not in points
        assert points["guardrails.request.duration"][0].value == 1

    @pytest.mark.asyncio
    async def test_emits_blocked_counter_on_output_block(self, iorails_tracing, metric_reader):
        _stub_safe_pipeline(iorails_tracing)
        iorails_tracing.rails_manager.is_output_safe = AsyncMock(
            return_value=RailResult(is_safe=False, reason="unsafe response")
        )

        result = await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        points = collect_metric_points(metric_reader)
        assert points["guardrails.requests.blocked"][0].value == 1
        assert points["guardrails.requests.blocked"][0].attributes["rail.type"] == "Output"

    @pytest.mark.asyncio
    async def test_no_blocked_counter_emitted_when_tracing_disabled(self, iorails_no_tracing, metric_reader):
        """With metrics disabled, a blocked-by-input-rail request emits no
        ``requests.blocked`` data point."""
        _stub_safe_pipeline(iorails_no_tracing)
        iorails_no_tracing.rails_manager.is_input_safe = AsyncMock(
            return_value=RailResult(is_safe=False, reason="unsafe")
        )

        result = await iorails_no_tracing.generate_async([{"role": "user", "content": "bad"}])

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        points = collect_metric_points(metric_reader)
        assert points == {}

    @pytest.mark.asyncio
    async def test_nonstream_rejections_counter_on_queue_full(self, iorails_tracing, metric_reader):
        """When the admission queue raises ``asyncio.QueueFull``,
        ``generate_async`` catches the exception, increments
        ``guardrails.nonstream.rejections``, and re-raises.

        Note: the rejection also propagates through the outer
        ``request_metrics()`` wrapper, so ``requests.errors`` and
        ``request.duration`` fire too — covered by the dedicated
        dual-signal test below.

        The queue's overflow semantics are covered in
        ``test_async_work_queue.py``; here we stub ``submit`` to raise
        directly so the test stays fast and focused on the counter wiring.
        """
        _stub_safe_pipeline(iorails_tracing)
        iorails_tracing._generate_async_queue.submit = AsyncMock(side_effect=asyncio.QueueFull("admission queue full"))

        with pytest.raises(asyncio.QueueFull, match="admission queue full"):
            await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        points = collect_metric_points(metric_reader)
        assert points["guardrails.nonstream.rejections"][0].value == 1

    @pytest.mark.asyncio
    async def test_queuefull_bumps_both_errors_and_nonstream_rejections(self, iorails_tracing, metric_reader):
        """Dual-signal semantics: a ``QueueFull`` rejection is BOTH a
        saturation signal (``nonstream.rejections``) AND a request error
        (``requests.errors{error.type=QueueFull}``).  Dashboards can
        count either one.  Also bumps the ``requests`` counter and
        records into the duration histogram — the request ran through
        the full lifecycle, even if only briefly.
        """
        _stub_safe_pipeline(iorails_tracing)
        iorails_tracing._generate_async_queue.submit = AsyncMock(side_effect=asyncio.QueueFull("admission queue full"))

        with pytest.raises(asyncio.QueueFull):
            await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        points = collect_metric_points(metric_reader)
        assert points["guardrails.nonstream.rejections"][0].value == 1
        assert points["guardrails.requests.errors"][0].value == 1
        assert points["guardrails.requests.errors"][0].attributes["error.type"] == "QueueFull"
        assert points["guardrails.requests"][0].value == 1
        assert points["guardrails.request.duration"][0].value == 1

    @pytest.mark.asyncio
    async def test_request_duration_includes_queue_wait(self, metric_reader):
        """``request.duration`` measures the full ``generate_async``
        lifecycle (OTEL HTTP semconv), so the time a request spends
        waiting in the admission queue is included.

        With a single worker and two submitted requests, the second
        request sits in the queue for the duration of the first.  The
        histogram's aggregate sum therefore captures at least one
        queue-wait period — if duration were worker-scope the waiting
        request would contribute ~0 to the sum.
        """
        gate = asyncio.Event()

        block_seconds = 0.05

        with (
            patch("nemoguardrails.guardrails.iorails.NONSTREAM_MAX_CONCURRENCY", 1),
            patch("nemoguardrails.guardrails.iorails.NONSTREAM_QUEUE_DEPTH", 4),
        ):
            with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
                iorails = IORails(RailsConfig.from_content(config=_make_metrics_only_config()))
            async with iorails:
                iorails._do_generate = _gated_generate(gate)

                tasks = [
                    asyncio.create_task(iorails.generate_async([{"role": "user", "content": f"m{i}"}]))
                    for i in range(2)
                ]
                # Wait until exactly one is executing and one is queued.
                await wait_for_queue_state(iorails._generate_async_queue, busy=1, pending=1)
                # Hold this state for a measurable duration so the queued
                # request accumulates queue-wait time.
                await asyncio.sleep(block_seconds)
                gate.set()
                await asyncio.gather(*tasks)

        # The sum of both recorded durations must exceed one block period.
        # A worker-scope duration would sum to ~block_seconds (the first
        # request held a worker, the second ran trivially after pickup);
        # full-lifecycle duration also covers the queued request's wait,
        # lifting the sum near 2×block.
        duration_sum = collect_histogram_sum(metric_reader, "guardrails.request.duration")
        assert duration_sum >= block_seconds * 1.5

    @pytest.mark.asyncio
    async def test_no_nonstream_rejections_counter_when_metrics_disabled(self, iorails_no_tracing, metric_reader):
        """Metrics disabled → even a QueueFull raise doesn't emit the counter."""
        _stub_safe_pipeline(iorails_no_tracing)
        iorails_no_tracing._generate_async_queue.submit = AsyncMock(
            side_effect=asyncio.QueueFull("admission queue full")
        )

        with pytest.raises(asyncio.QueueFull):
            await iorails_no_tracing.generate_async([{"role": "user", "content": "hi"}])

        points = collect_metric_points(metric_reader)
        assert "guardrails.nonstream.rejections" not in points


class TestStreamAsyncRequestMetrics:
    """Streaming path also emits request-level metrics via the shared
    ``traced_request`` helper — no separate plumbing."""

    @pytest.mark.asyncio
    async def test_emits_counter_and_duration_on_safe_stream(self, iorails_streaming_input_only_tracing, metric_reader):
        """Happy-path stream_async → counter +1, duration recorded once,
        no errors, stream.active back to 0, no rejections."""
        iorails = iorails_streaming_input_only_tracing
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.engine_registry.stream_model_call = _mock_chunks_stream

        chunks = [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]
        assert chunks == ["Hello", " ", "world"]

        points = collect_metric_points(metric_reader)
        assert points["guardrails.requests"][0].value == 1
        assert points["guardrails.request.duration"][0].value == 1
        assert "guardrails.requests.errors" not in points
        # Saturation: stream.active nets to 0 after the stream completes;
        # stream.rejections never fires on the happy path.
        assert points["guardrails.stream.active"][0].value == 0
        assert "guardrails.stream.rejections" not in points

    @pytest.mark.asyncio
    async def test_stream_rejections_counter_on_semaphore_full(
        self, iorails_streaming_input_only_tracing, metric_reader
    ):
        """A stream that arrives while the semaphore is fully occupied is
        rejected with ``asyncio.QueueFull`` and the ``stream.rejections``
        counter increments.
        """
        iorails = iorails_streaming_input_only_tracing
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.engine_registry.stream_model_call = _mock_chunks_stream

        saturate_stream_semaphore(iorails)
        with pytest.raises(asyncio.QueueFull, match="Streaming concurrency limit reached"):
            [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]

        points = collect_metric_points(metric_reader)
        assert points["guardrails.stream.rejections"][0].value == 1
        # Active counter untouched — the semaphore was never acquired by the
        # rejected stream, so the UpDownCounter never saw a +1 and emits no
        # data point (UpDownCounters only export points after first ``.add()``).
        assert "guardrails.stream.active" not in points

    @pytest.mark.asyncio
    async def test_stream_queuefull_bumps_both_errors_and_stream_rejections(
        self, iorails_streaming_input_only_tracing, metric_reader
    ):
        """Streaming equivalent of the non-streaming dual-signal test: a
        ``QueueFull`` on the semaphore check is BOTH a saturation signal
        (``stream.rejections``) AND a request error
        (``requests.errors{error.type=QueueFull}``)
        """
        iorails = iorails_streaming_input_only_tracing
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.engine_registry.stream_model_call = _mock_chunks_stream

        saturate_stream_semaphore(iorails)
        with pytest.raises(asyncio.QueueFull, match="Streaming concurrency limit reached"):
            [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]

        points = collect_metric_points(metric_reader)
        assert points["guardrails.stream.rejections"][0].value == 1
        assert points["guardrails.requests.errors"][0].value == 1
        assert points["guardrails.requests.errors"][0].attributes["error.type"] == "QueueFull"
        assert points["guardrails.requests"][0].value == 1
        assert points["guardrails.request.duration"][0].value == 1

    @pytest.mark.asyncio
    async def test_stream_active_is_one_mid_flight(self, iorails_streaming_input_only_tracing, metric_reader):
        """Observe the UpDownCounter mid-flight: after the first chunk is
        pulled (semaphore acquired, stream body running), ``stream.active``
        reads 1; after the iterator is consumed, it reads 0.
        """
        iorails = iorails_streaming_input_only_tracing
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.engine_registry.stream_model_call = _mock_chunks_stream

        iterator = iorails.stream_async([{"role": "user", "content": "hi"}]).__aiter__()

        # Before any chunk is pulled, nothing has touched the UpDownCounter
        # yet, so no data point is emitted (semantically equivalent to 0).
        before = collect_metric_points(metric_reader)
        assert "guardrails.stream.active" not in before

        # During: pull the first chunk → semaphore acquired, counter at +1.
        first = await iterator.__anext__()
        assert first == "Hello"
        mid = collect_metric_points(metric_reader)
        assert mid["guardrails.stream.active"][0].value == 1

        # Drain the rest.
        rest = [c async for c in iterator]
        assert rest == [" ", "world"]

        # After: counter back to 0 (net of +1 / -1).
        final = collect_metric_points(metric_reader)
        assert final["guardrails.stream.active"][0].value == 0

    @pytest.mark.asyncio
    async def test_stream_active_nets_to_zero_on_stream_failure(
        self, iorails_streaming_input_only_tracing, metric_reader
    ):
        """Even when the LLM raises mid-stream, ``stream.active`` decrements
        on exit — the ``stream_active_metric`` context manager's ``finally``
        guarantees the -1.
        """
        iorails = iorails_streaming_input_only_tracing
        _stub_deep_streaming_pipeline(iorails, main_stream=_engine_failing_stream)

        chunks = [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]
        # Stream failure was converted to an error-payload chunk.
        assert any(c.startswith('{"error"') for c in chunks)

        points = collect_metric_points(metric_reader)
        assert points["guardrails.stream.active"][0].value == 0

    @pytest.mark.asyncio
    async def test_no_stream_saturation_metrics_when_metrics_disabled(
        self, iorails_streaming_no_tracing, metric_reader
    ):
        """Tracing + metrics disabled → ``stream.active`` and
        ``stream.rejections`` don't emit, even on rejection path.
        """
        iorails = iorails_streaming_no_tracing
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.engine_registry.stream_model_call = _mock_chunks_stream

        saturate_stream_semaphore(iorails)
        with pytest.raises(asyncio.QueueFull):
            [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]

        points = collect_metric_points(metric_reader)
        assert "guardrails.stream.active" not in points
        assert "guardrails.stream.rejections" not in points

    @pytest.mark.asyncio
    async def test_emits_errors_counter_on_stream_failure(self, iorails_streaming_input_only_tracing, metric_reader):
        """Streaming failures don't propagate — the generation task swallows
        the exception and pushes an error chunk.  The errors counter is
        bumped explicitly via ``record_request_error`` so dashboards still
        see the failure.
        """
        iorails = iorails_streaming_input_only_tracing
        _stub_deep_streaming_pipeline(iorails, main_stream=_engine_failing_stream)

        chunks = [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]
        # The generation task converts the exception into an error-payload chunk.
        assert any(c.startswith('{"error"') for c in chunks)

        points = collect_metric_points(metric_reader)
        assert points["guardrails.requests"][0].value == 1
        assert points["guardrails.requests.errors"][0].value == 1
        assert points["guardrails.requests.errors"][0].attributes["error.type"] == "RuntimeError"
        assert points["guardrails.request.duration"][0].value == 1

    @pytest.mark.asyncio
    async def test_emits_blocked_counter_on_stream_input_block(
        self, iorails_streaming_input_only_tracing, metric_reader
    ):
        iorails = iorails_streaming_input_only_tracing
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=False, reason="unsafe"))

        chunks = [c async for c in iorails.stream_async([{"role": "user", "content": "bad"}])]
        assert chunks == [REFUSAL_MESSAGE]

        points = collect_metric_points(metric_reader)
        assert points["guardrails.requests.blocked"][0].value == 1
        assert points["guardrails.requests.blocked"][0].attributes["rail.type"] == "Input"
        # No LLM call / no errors.
        assert "guardrails.requests.errors" not in points
        assert points["guardrails.request.duration"][0].value == 1

    @pytest.mark.asyncio
    async def test_emits_blocked_counter_on_stream_output_block(self, iorails_streaming_output_tracing, metric_reader):
        """Streaming + output-rail block exercises
        ``_run_output_rails_in_streaming`` — a separate code path from the
        non-streaming ``_do_generate`` output-block site.
        """
        iorails = iorails_streaming_output_tracing
        _stub_deep_streaming_pipeline(iorails)
        iorails.rails_manager.is_output_safe = AsyncMock(
            return_value=RailResult(is_safe=False, reason="unsafe response")
        )

        chunks = [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]
        # Output block terminates the stream with a JSON error payload chunk.
        assert any(c.startswith('{"error"') for c in chunks)

        points = collect_metric_points(metric_reader)
        assert points["guardrails.requests.blocked"][0].value == 1
        assert points["guardrails.requests.blocked"][0].attributes["rail.type"] == "Output"

    @pytest.mark.asyncio
    async def test_no_metrics_on_stream_output_block_when_tracing_disabled(self, metric_reader):
        """Regression guard for the ``self._tracing_enabled`` gate on the
        output-rail streaming block path: when tracing is disabled, no
        metrics emit even though the block still happens.  Catches any
        future removal of the gate on this emit site — the exact class of
        bug the P1 review flagged on the other streaming path.
        """
        cfg = copy.deepcopy(NEMOGUARDS_CONFIG)
        cfg["rails"]["output"]["streaming"] = {
            "enabled": True,
            "chunk_size": 5,
            "context_size": 2,
            "stream_first": True,
        }
        # No tracing.enabled=True → guardrails tracing/metrics disabled.
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            iorails = IORails(RailsConfig.from_content(config=cfg))
        async with iorails:
            _stub_deep_streaming_pipeline(iorails)
            iorails.rails_manager.is_output_safe = AsyncMock(
                return_value=RailResult(is_safe=False, reason="unsafe response")
            )

            chunks = [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]
            # Block still works — customer-visible behavior unchanged.
            assert any(c.startswith('{"error"') for c in chunks)

            points = collect_metric_points(metric_reader)
            assert points == {}

    @pytest.mark.asyncio
    async def test_emits_no_metrics_on_stream_failure_when_tracing_disabled(
        self, iorails_streaming_no_tracing, metric_reader
    ):
        """Regression: with tracing disabled, a streaming failure must emit
        no metrics — including the errors counter.  ``record_request_error``
        inside ``_generation_task`` is gated on ``request_span`` precisely
        to keep the errors/requests pair consistent; this locks that in.
        """
        iorails = iorails_streaming_no_tracing
        _stub_deep_streaming_pipeline(iorails, main_stream=_engine_failing_stream)

        chunks = [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]
        assert any(c.startswith('{"error"') for c in chunks)

        points = collect_metric_points(metric_reader)
        assert points == {}


class TestIndependentTracingAndMetrics:
    """Tracing and metrics are independent OTEL signals.

    Exercises the three non-trivial config combinations that the single-flag
    design did not support:

    * metrics-only (tracing off, metrics on) — the setup Pouyanpi called
      out on the PR as cost-optimized SLO dashboards
    * tracing-only (tracing on, metrics off)
    * both off — already covered elsewhere, included here for completeness
      of the four-quadrant matrix

    The "both on" quadrant is covered extensively by
    :class:`TestGenerateAsyncWithTracing` and
    :class:`TestGenerateAsyncRequestMetrics`.
    """

    @pytest.mark.asyncio
    async def test_metrics_only_emits_metrics_but_no_spans(self, metric_reader, exporter):
        """Metrics enabled, tracing disabled → guardrails counters emit,
        zero spans exported.  Core regression test for the decoupling.
        """
        with patch.object(telemetry, "_tracer", None):
            with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
                iorails = IORails(RailsConfig.from_content(config=_make_metrics_only_config()))
            async with iorails:
                _stub_safe_pipeline(iorails)
                await iorails.generate_async([{"role": "user", "content": "hi"}])

        points = collect_metric_points(metric_reader)
        assert points["guardrails.requests"][0].value == 1
        assert points["guardrails.request.duration"][0].value == 1
        assert exporter.get_finished_spans() == ()

    @pytest.mark.asyncio
    async def test_tracing_only_emits_spans_but_no_metrics(self, tracer_from_provider, metric_reader, exporter):
        """Tracing enabled, metrics disabled → guardrails.request span
        emits, zero metric data points.
        """
        with patch.object(telemetry, "_tracer", tracer_from_provider):
            with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
                iorails = IORails(RailsConfig.from_content(config=_make_tracing_only_config()))
            async with iorails:
                _stub_safe_pipeline(iorails)
                await iorails.generate_async([{"role": "user", "content": "hi"}])

        spans = exporter.get_finished_spans()
        assert any(s.name == "guardrails.request" for s in spans)
        points = collect_metric_points(metric_reader)
        assert points == {}


class TestNonstreamStateGauges:
    """IORails-level integration tests for the ``nonstream.queued`` +
    ``nonstream.active`` ObservableGauges.

    Each test constructs IORails *after* ``metric_reader`` has installed its
    test-local Meter — the gauges are registered in ``IORails.start()``,
    so the Meter in effect at startup time is what the reader sees.  The
    ``iorails_tracing`` / ``iorails_no_tracing`` fixtures set up the Meter
    too late for gauges, which is why these tests build IORails inline.
    """

    @pytest.mark.asyncio
    async def test_gauges_registered_and_read_zero_at_rest(self, metric_reader):
        """A freshly-started IORails with no pending work → both gauges read 0."""
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            iorails = IORails(RailsConfig.from_content(config=_make_metrics_only_config()))
        async with iorails:
            points = collect_metric_points(metric_reader)
            assert points["guardrails.nonstream.queued"][0].value == 0
            assert points["guardrails.nonstream.active"][0].value == 0

    @pytest.mark.asyncio
    async def test_nonstream_active_reads_one_while_worker_is_busy(self, metric_reader):
        """A pipeline blocked on an Event → ``nonstream.active == 1`` during
        the block, 0 after the Event is set and the worker finishes.
        """
        gate = asyncio.Event()

        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            iorails = IORails(RailsConfig.from_content(config=_make_metrics_only_config()))
        async with iorails:
            iorails._do_generate = _gated_generate(gate)

            task = asyncio.create_task(iorails.generate_async([{"role": "user", "content": "hi"}]))
            # Wait for the worker to pick up the item and enter
            # the gated generate (busy_count=1, pending=0).
            await wait_for_queue_state(iorails._generate_async_queue, busy=1, pending=0)

            mid = collect_metric_points(metric_reader)
            assert mid["guardrails.nonstream.active"][0].value == 1
            assert mid["guardrails.nonstream.queued"][0].value == 0

            gate.set()
            await task

            final = collect_metric_points(metric_reader)
            assert final["guardrails.nonstream.active"][0].value == 0
            assert final["guardrails.nonstream.queued"][0].value == 0

    @pytest.mark.asyncio
    async def test_nonstream_queued_reflects_backlog_past_worker_capacity(self, metric_reader):
        """With a single worker occupied and extras pending, ``nonstream.queued``
        reports the backlog size while ``nonstream.active == 1``.
        """
        gate = asyncio.Event()

        # Patch module-level budgets so the backlog test doesn't need to
        # spin up 256 workers.
        with (
            patch("nemoguardrails.guardrails.iorails.NONSTREAM_MAX_CONCURRENCY", 1),
            patch("nemoguardrails.guardrails.iorails.NONSTREAM_QUEUE_DEPTH", 8),
        ):
            with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
                iorails = IORails(RailsConfig.from_content(config=_make_metrics_only_config()))
            async with iorails:
                iorails._do_generate = _gated_generate(gate)

                tasks = [
                    asyncio.create_task(iorails.generate_async([{"role": "user", "content": f"m{i}"}]))
                    for i in range(3)
                ]
                # Wait for one worker to pick up an item and the other two
                # to sit in the queue (busy=1, pending=2).
                await wait_for_queue_state(iorails._generate_async_queue, busy=1, pending=2)

                mid = collect_metric_points(metric_reader)
                assert mid["guardrails.nonstream.active"][0].value == 1
                assert mid["guardrails.nonstream.queued"][0].value == 2

                gate.set()
                await asyncio.gather(*tasks)

                final = collect_metric_points(metric_reader)
                assert final["guardrails.nonstream.active"][0].value == 0
                assert final["guardrails.nonstream.queued"][0].value == 0

    @pytest.mark.asyncio
    async def test_gauges_not_registered_when_metrics_disabled(self, metric_reader):
        """Metrics disabled in config → ``start()`` does not register gauges
        and they never appear in collection output.
        """
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            iorails = IORails(RailsConfig.from_content(config=NEMOGUARDS_CONFIG))
        async with iorails:
            points = collect_metric_points(metric_reader)
            assert "guardrails.nonstream.queued" not in points
            assert "guardrails.nonstream.active" not in points

    @pytest.mark.asyncio
    async def test_gauges_soft_disabled_after_stop(self, metric_reader):
        """``stop()`` flips ``self._running`` to False; the gauge callbacks
        return ``[]`` on subsequent collection (soft-disable).  Needed
        because OTEL Python has no public unregister API — an always-alive
        callback would leak dead-IORails state across tests and long-running
        processes.
        """
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            iorails = IORails(RailsConfig.from_content(config=_make_metrics_only_config()))
        await iorails.start()
        try:
            alive = collect_metric_points(metric_reader)
            assert alive["guardrails.nonstream.queued"][0].value == 0
            assert alive["guardrails.nonstream.active"][0].value == 0
        finally:
            await iorails.stop()

        stopped = collect_metric_points(metric_reader)
        # Either the metric is absent (SDK-dependent when callbacks return [])
        # or present with no data points — both mean "no observation".
        assert stopped.get("guardrails.nonstream.queued", []) == []
        assert stopped.get("guardrails.nonstream.active", []) == []

    @pytest.mark.asyncio
    async def test_stop_then_start_resumes_gauge_observations(self, metric_reader):
        """``stop()`` → ``start()`` cycle must re-enable observations without
        re-registering (the ``_gauges_registered`` flag remains True across
        the cycle; ``self._running`` flipping back to True is what
        re-enables the callbacks).
        """
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            iorails = IORails(RailsConfig.from_content(config=_make_metrics_only_config()))

        await iorails.start()
        assert iorails._gauges_registered is True
        await iorails.stop()

        # Restart — must NOT re-register (flag still True) but must resume
        # reporting because ``self._running`` flips back to True.
        await iorails.start()
        try:
            assert iorails._gauges_registered is True
            resumed = collect_metric_points(metric_reader)
            assert resumed["guardrails.nonstream.queued"][0].value == 0
            assert resumed["guardrails.nonstream.active"][0].value == 0
        finally:
            await iorails.stop()


class TestRequestsActiveAggregate:
    """End-to-end coverage for ``guardrails.requests.active`` — the path-
    agnostic aggregate that counts both non-streaming and streaming
    in-flight requests.

    The invariant this metric set satisfies at any collection instant:

        requests.active  ≈  nonstream.queued + nonstream.active + stream.active

    IORails is built inline so ``metric_reader`` installs its test Meter
    before ``start()`` registers the saturation ObservableGauges (same
    rationale as :class:`TestNonstreamStateGauges`).
    """

    @pytest.mark.asyncio
    async def test_captures_queued_nonstream_request_mid_flight(self, metric_reader):
        """A request stuck in the admission queue contributes to
        ``requests.active`` — the full-lifecycle scope means queue-wait
        counts, not just worker time.  With one worker busy and one
        queued, ``requests.active`` reads 2.
        """
        gate = asyncio.Event()

        with (
            patch("nemoguardrails.guardrails.iorails.NONSTREAM_MAX_CONCURRENCY", 1),
            patch("nemoguardrails.guardrails.iorails.NONSTREAM_QUEUE_DEPTH", 4),
        ):
            with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
                iorails = IORails(RailsConfig.from_content(config=_make_metrics_only_config()))
            async with iorails:
                iorails._do_generate = _gated_generate(gate)
                tasks = [
                    asyncio.create_task(iorails.generate_async([{"role": "user", "content": f"m{i}"}]))
                    for i in range(2)
                ]
                await wait_for_queue_state(iorails._generate_async_queue, busy=1, pending=1)

                mid = collect_metric_points(metric_reader)
                assert mid["guardrails.requests.active"][0].value == 2

                gate.set()
                await asyncio.gather(*tasks)

                final = collect_metric_points(metric_reader)
                assert final["guardrails.requests.active"][0].value == 0

    @pytest.mark.asyncio
    async def test_captures_streaming_request_mid_flight(self, iorails_streaming_input_only_tracing, metric_reader):
        """A streaming request holding a semaphore permit contributes to
        ``requests.active``.  Mid-stream, the counter reads 1; after the
        iterator drains, it nets to 0.
        """
        iorails = iorails_streaming_input_only_tracing
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.engine_registry.stream_model_call = _mock_chunks_stream

        iterator = iorails.stream_async([{"role": "user", "content": "hi"}]).__aiter__()

        # Pull the first chunk → semaphore acquired, stream body running.
        first = await iterator.__anext__()
        assert first == "Hello"
        mid = collect_metric_points(metric_reader)
        assert mid["guardrails.requests.active"][0].value == 1

        # Drain and check net-to-zero.
        rest = [c async for c in iterator]
        assert rest == [" ", "world"]
        final = collect_metric_points(metric_reader)
        assert final["guardrails.requests.active"][0].value == 0

    @pytest.mark.asyncio
    async def test_invariant_aggregate_equals_component_sum(self, metric_reader):
        """Core payoff check: with M non-streaming executing, N queued,
        and S streaming simultaneously, the aggregate counter matches
        the component-wise sum.

        Uses a single IORails instance: M=1 executing + N=1 queued +
        S=1 streaming → ``requests.active == 3``, each of the per-path
        metrics reads its own value, and the sum of the three equals
        the aggregate.
        """
        nonstream_gate = asyncio.Event()

        # Drop output rails so ``stream_async`` doesn't trip the
        # StreamingNotSupportedError path (matches the ``_INPUT_ONLY_*``
        # configs used by the streaming-path fixtures above).
        invariant_config = copy.deepcopy(NEMOGUARDS_CONFIG)
        invariant_config["rails"]["output"] = {"flows": []}
        invariant_config["metrics"] = {"enabled": True}

        with patch("nemoguardrails.guardrails.iorails.NONSTREAM_MAX_CONCURRENCY", 1):
            with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
                iorails = IORails(RailsConfig.from_content(config=invariant_config))
            async with iorails:
                iorails._do_generate = _gated_generate(nonstream_gate)
                iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
                iorails.engine_registry.stream_model_call = _mock_chunks_stream

                # Launch one executing + one queued non-streaming request.
                nonstream_tasks = [
                    asyncio.create_task(iorails.generate_async([{"role": "user", "content": f"n{i}"}]))
                    for i in range(2)
                ]
                await wait_for_queue_state(iorails._generate_async_queue, busy=1, pending=1)

                # Launch one streaming request and pull the first chunk
                # to force semaphore acquisition + stream-body execution.
                stream_iter = iorails.stream_async([{"role": "user", "content": "s0"}]).__aiter__()
                first = await stream_iter.__anext__()
                assert first == "Hello"

                mid = collect_metric_points(metric_reader)
                aggregate = mid["guardrails.requests.active"][0].value
                nonstream_active = mid["guardrails.nonstream.active"][0].value
                nonstream_queued = mid["guardrails.nonstream.queued"][0].value
                stream_active = mid["guardrails.stream.active"][0].value

                assert nonstream_active == 1
                assert nonstream_queued == 1
                assert stream_active == 1
                assert aggregate == 3
                # The invariant itself.
                assert aggregate == nonstream_active + nonstream_queued + stream_active

                # Drain everything so the fixture teardown is clean.
                nonstream_gate.set()
                [_ async for _ in stream_iter]
                await asyncio.gather(*nonstream_tasks)

                final = collect_metric_points(metric_reader)
                assert final["guardrails.requests.active"][0].value == 0


def _stub_deep_pipeline_with_usage(iorails):
    """Like ``_stub_deep_pipeline`` but attaches ``UsageInfo`` to every
    ``LLMResponse`` so the LLM-call metrics emitted by ``EngineRegistry.model_call``
    have something to record.

    The token counts are arbitrary and chosen to be distinct per model so
    tests asserting "metric fired for this model" can also check the
    recorded sum is the expected value (catching a label-shuffle bug).
    """
    from nemoguardrails.guardrails.api_engine import APIEngine
    from nemoguardrails.guardrails.model_engine import ModelEngine

    # (model_name → (input_tokens, output_tokens, response_content))
    per_model = {
        "main": (100, 50, "Hello"),
        "content_safety": (40, 5, SAFE_OUTPUT_JSON),
        "topic_control": (30, 5, SAFE_INPUT_JSON),
    }

    for name, engine in iorails.engine_registry._engines.items():
        if isinstance(engine, ModelEngine):
            input_tokens, output_tokens, content = per_model.get(name, (10, 10, SAFE_INPUT_JSON))
            engine.chat_completion = AsyncMock(
                return_value=LLMResponse(
                    content=content,
                    usage=UsageInfo(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    ),
                )
            )
        elif isinstance(engine, APIEngine):
            engine.call = AsyncMock(return_value={"jailbreak": False, "score": 0.01})


class TestGenerateAsyncLLMMetrics:
    """End-to-end coverage for the OTEL GenAI client metrics emitted from
    ``EngineRegistry.model_call`` during a real ``IORails.generate_async``
    invocation.

    Mocks at the ModelEngine layer (not the EngineRegistry layer) so the
    full RailsManager → RailAction → EngineRegistry → metric-emission
    chain executes.  A safe end-to-end call drives multiple LLM calls
    (input rails: content_safety + topic_control; main generation;
    output rails: content_safety) — token + duration metrics fire for
    each, distinguished by ``gen_ai.request.model``.
    """

    @pytest.mark.asyncio
    async def test_emits_token_and_duration_metrics_per_model(self, iorails_tracing, metric_reader):
        """Safe end-to-end → token usage and duration metrics emit for
        every distinct model the rails invoke.  Verifies the
        ``metrics_enabled=True`` flag plumbs from IORails through
        EngineRegistry to the per-call emission helpers.
        """
        _stub_deep_pipeline_with_usage(iorails_tracing)

        result = await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])
        assert result == {"role": "assistant", "content": "Hello"}

        points = collect_metric_points(metric_reader)

        # Token usage: each LLM call emits two observations (input,
        # output).  Models in NEMOGUARDS_CONFIG: main, content_safety,
        # topic_control.  content_safety is invoked twice (input + output
        # rails) but those merge into the same {model, token.type}
        # data points → 6 distinct points total.
        token_models = {p.attributes["gen_ai.request.model"] for p in points["gen_ai.client.token.usage"]}
        assert token_models == {
            "meta/llama-3.3-70b-instruct",  # main
            "nvidia/llama-3.1-nemoguard-8b-content-safety",
            "nvidia/llama-3.1-nemoguard-8b-topic-control",
        }
        # Each model produces both input and output observations.
        for model in token_models:
            types_for_model = {
                p.attributes["gen_ai.token.type"]
                for p in points["gen_ai.client.token.usage"]
                if p.attributes["gen_ai.request.model"] == model
            }
            assert types_for_model == {"input", "output"}

        # Duration: one data point per (model, no-error) — three models.
        duration_models = {p.attributes["gen_ai.request.model"] for p in points["gen_ai.client.operation.duration"]}
        assert duration_models == token_models

        # All observations carry the standard provider + operation labels.
        for point in points["gen_ai.client.token.usage"] + points["gen_ai.client.operation.duration"]:
            assert point.attributes["gen_ai.operation.name"] == "chat"
            assert point.attributes["gen_ai.provider.name"] == "nim"
            assert "error.type" not in point.attributes

    @pytest.mark.asyncio
    async def test_no_llm_metrics_when_metrics_disabled(self, iorails_no_tracing, metric_reader):
        """Default config (metrics off) → no LLM-call metrics emit even
        when a MeterProvider is installed and the rails actually call
        the engines.  Catches the gating slip where LLM-call metrics
        would fire purely on meter availability.
        """
        _stub_deep_pipeline_with_usage(iorails_no_tracing)

        await iorails_no_tracing.generate_async([{"role": "user", "content": "hi"}])

        points = collect_metric_points(metric_reader)
        assert "gen_ai.client.token.usage" not in points
        assert "gen_ai.client.operation.duration" not in points


@pytest.fixture(autouse=True)
def _clear_otel_content_envvars(monkeypatch):
    """Strip OTEL content-capture env vars from each test's environment.

    Without this, an inherited ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``
    or ``OTEL_SEMCONV_STABILITY_OPT_IN`` from CI / dev shell would flip
    capture on or change format mid-suite and produce flaky assertions.
    """
    monkeypatch.delenv(OtelContentCapture.CAPTURE_CONTENT_ENV, raising=False)
    monkeypatch.delenv(OtelContentCapture.STABILITY_OPT_IN_ENV, raising=False)


def _make_content_capture_config():
    """``_make_tracing_config()`` with ``enable_content_capture=True`` added."""
    cfg = _make_tracing_config()
    cfg["tracing"]["enable_content_capture"] = True
    return cfg


def _make_content_capture_streaming_config():
    """Input-only streaming config with tracing + content capture enabled."""
    cfg = copy.deepcopy(_INPUT_ONLY_STREAMING_TRACING_CONFIG)
    cfg["tracing"]["enable_content_capture"] = True
    return cfg


@pytest_asyncio.fixture
async def iorails_content_capture(tracer_from_provider):
    """IORails with tracing + content capture enabled."""
    with patch.object(telemetry, "_tracer", tracer_from_provider):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            config = RailsConfig.from_content(config=_make_content_capture_config())
            iorails = IORails(config)
        async with iorails:
            yield iorails


@pytest_asyncio.fixture
async def iorails_streaming_content_capture(tracer_from_provider):
    """Input-only streaming + tracing + content capture enabled."""
    with patch.object(telemetry, "_tracer", tracer_from_provider):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            config = RailsConfig.from_content(config=_make_content_capture_streaming_config())
            iorails = IORails(config)
        async with iorails:
            yield iorails


def _request_span(spans):
    """Return the single guardrails.request SERVER span from a list of spans."""
    request_spans = [s for s in spans if s.name == "guardrails.request"]
    assert len(request_spans) == 1
    return request_spans[0]


def _main_llm_span(spans):
    """Return the CLIENT span for the main LLM call (model name "meta/llama-3.3-70b-instruct")."""
    candidates = [
        s
        for s in spans
        if s.kind == SpanKind.CLIENT and s.attributes.get("gen_ai.request.model") == "meta/llama-3.3-70b-instruct"
    ]
    assert len(candidates) == 1
    return candidates[0]


class TestContentCaptureDisabled:
    """Default config (tracing on, capture off): spans carry no content attrs/events."""

    @pytest.mark.asyncio
    async def test_no_content_on_any_span(self, iorails_tracing, exporter):
        """Capture off → request, LLM, and rail spans carry no captured content."""
        _stub_deep_pipeline(iorails_tracing)

        await iorails_tracing.generate_async([{"role": "user", "content": "hi"}])

        spans = exporter.get_finished_spans()

        request_span = _request_span(spans)
        assert GuardrailsAttributes.REQUEST_INPUT not in request_span.attributes
        assert GuardrailsAttributes.REQUEST_OUTPUT not in request_span.attributes
        assert all(not e.name.startswith("gen_ai.") for e in request_span.events)

        llm_span = _main_llm_span(spans)
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in llm_span.attributes
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in llm_span.attributes
        assert all(not e.name.startswith("gen_ai.") for e in llm_span.events)

        for rail_span in (s for s in spans if s.name == "guardrails.rail"):
            assert GuardrailsAttributes.RAIL_INPUT not in rail_span.attributes
            assert GuardrailsAttributes.RAIL_REASON not in rail_span.attributes


class TestContentCaptureLegacyFormat:
    """Capture on + opt-in env unset: guardrails.request.* attrs on request span; gen_ai.* events on LLM span."""

    @pytest.mark.asyncio
    async def test_request_span_carries_guardrails_attrs(self, iorails_content_capture, exporter):
        """Request span carries guardrails.request.input/output attrs, not gen_ai.* events."""
        _stub_deep_pipeline(iorails_content_capture, main_llm_response="Hi back")

        await iorails_content_capture.generate_async([{"role": "user", "content": "hello"}])

        span = _request_span(exporter.get_finished_spans())
        assert json.loads(span.attributes[GuardrailsAttributes.REQUEST_INPUT]) == [{"role": "user", "content": "hello"}]
        assert span.attributes[GuardrailsAttributes.REQUEST_OUTPUT] == "Hi back"
        # The request span does not carry gen_ai.* content attrs or events
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
        assert all(not e.name.startswith("gen_ai.") for e in span.events)

    @pytest.mark.asyncio
    async def test_legacy_events_on_main_llm_span(self, iorails_content_capture, exporter):
        """The main LLM span carries the legacy gen_ai.* message/choice events.

        Verifies the engine→set_llm_call_content wiring; the exact event
        attributes are pinned by the unit tests for _set_llm_call_content_events.
        """
        _stub_deep_pipeline(iorails_content_capture)

        await iorails_content_capture.generate_async([{"role": "user", "content": "hello"}])

        span = _main_llm_span(exporter.get_finished_spans())
        event_names = [e.name for e in span.events]
        assert "gen_ai.user.message" in event_names
        assert "gen_ai.choice" in event_names

    @pytest.mark.asyncio
    async def test_refusal_message_captured_on_blocked_input(self, iorails_content_capture, exporter):
        """A blocked input records REFUSAL_MESSAGE as guardrails.request.output."""
        _stub_deep_pipeline(iorails_content_capture, input_safe=False)

        result = await iorails_content_capture.generate_async([{"role": "user", "content": "bad"}])
        assert result["content"] == REFUSAL_MESSAGE

        span = _request_span(exporter.get_finished_spans())
        assert span.attributes[GuardrailsAttributes.REQUEST_OUTPUT] == REFUSAL_MESSAGE

    @pytest.mark.asyncio
    async def test_refusal_message_captured_on_blocked_output(self, iorails_content_capture, exporter):
        """A blocked OUTPUT records REFUSAL_MESSAGE as guardrails.request.output.

        The LLM CLIENT span still records the raw model response while the
        request SERVER span records what the caller actually received (REFUSAL) —
        this is the semantic distinction that motivated the separate attr names.
        """
        iorails = iorails_content_capture
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="bad response"))
        iorails.rails_manager.is_output_safe = AsyncMock(
            return_value=RailResult(is_safe=False, reason="unsafe response")
        )

        result = await iorails.generate_async([{"role": "user", "content": "hi"}])
        assert result["content"] == REFUSAL_MESSAGE

        span = _request_span(exporter.get_finished_spans())
        assert span.attributes[GuardrailsAttributes.REQUEST_OUTPUT] == REFUSAL_MESSAGE

    @pytest.mark.asyncio
    async def test_llm_and_request_spans_diverge_on_output_block(self, iorails_content_capture, exporter):
        """On an output block, the LLM CLIENT span and request SERVER span hold different outputs.

        This is the core scenario that motivated separate guardrails.request.*
        attributes: the LLM span records the RAW model response (what the model
        produced) while the request span records REFUSAL_MESSAGE (what the caller
        actually received).  The main LLM call runs for real at the engine level
        so its CLIENT span is created and captures content; only the output rail
        is forced to block.
        """
        iorails = iorails_content_capture
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        # Mock at the engine level (not engine_registry.model_call) so the real
        # model_call wrapper runs, creating the LLM CLIENT span + capturing content.
        iorails.engine_registry._engines["main"].chat_completion = AsyncMock(
            return_value=LLMResponse(content="raw model answer")
        )
        iorails.rails_manager.is_output_safe = AsyncMock(
            return_value=RailResult(is_safe=False, reason="unsafe response")
        )

        result = await iorails.generate_async([{"role": "user", "content": "hi"}])
        assert result["content"] == REFUSAL_MESSAGE

        spans = exporter.get_finished_spans()
        # LLM CLIENT span: the raw model output (legacy gen_ai.choice event)
        llm_span = _main_llm_span(spans)
        choice = next(e for e in llm_span.events if e.name == "gen_ai.choice")
        assert dict(choice.attributes)["message.content"] == "raw model answer"
        # Request SERVER span: the refusal the caller actually received — divergent
        req_span = _request_span(spans)
        assert req_span.attributes[GuardrailsAttributes.REQUEST_OUTPUT] == REFUSAL_MESSAGE


class TestContentCaptureJsonFormat:
    """Capture on + OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental: JSON attrs."""

    @pytest.fixture(autouse=True)
    def _set_stability_opt_in(self, monkeypatch):
        monkeypatch.setenv(
            OtelContentCapture.STABILITY_OPT_IN_ENV,
            OtelContentCapture.STABILITY_OPT_IN_LATEST,
        )

    @pytest.mark.asyncio
    async def test_request_span_carries_guardrails_attrs(self, iorails_content_capture, exporter):
        """Opt-in set → request span carries guardrails.request.* attrs (plain strings)."""
        _stub_deep_pipeline(iorails_content_capture, main_llm_response="The answer")

        await iorails_content_capture.generate_async([{"role": "user", "content": "hello"}])

        span = _request_span(exporter.get_finished_spans())
        assert json.loads(span.attributes[GuardrailsAttributes.REQUEST_INPUT]) == [{"role": "user", "content": "hello"}]
        assert span.attributes[GuardrailsAttributes.REQUEST_OUTPUT] == "The answer"
        # The request span does not carry gen_ai.* content attrs or events
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
        assert all(not e.name.startswith("gen_ai.") for e in span.events)

    @pytest.mark.asyncio
    async def test_system_instructions_on_llm_span(self, iorails_content_capture, exporter):
        """System messages split to gen_ai.system_instructions on the LLM span."""
        _stub_deep_pipeline(iorails_content_capture)

        messages = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]
        await iorails_content_capture.generate_async(messages)

        llm_span = _main_llm_span(exporter.get_finished_spans())
        sysinst = json.loads(llm_span.attributes[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS])
        inputs = json.loads(llm_span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])

        assert sysinst == [{"type": "text", "content": "be helpful"}]
        assert [m["role"] for m in inputs] == ["user"]

        # Request span records the full raw input list (no split)
        req_span = _request_span(exporter.get_finished_spans())
        assert json.loads(req_span.attributes[GuardrailsAttributes.REQUEST_INPUT]) == messages

    @pytest.mark.asyncio
    async def test_json_attrs_on_main_llm_span(self, iorails_content_capture, exporter):
        """Opt-in set → the main LLM span carries JSON input/output message attrs."""
        _stub_deep_pipeline(iorails_content_capture)

        await iorails_content_capture.generate_async([{"role": "user", "content": "hello"}])

        span = _main_llm_span(exporter.get_finished_spans())
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in span.attributes
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in span.attributes


class TestContentCaptureEnvVarFallback:
    """OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT enables capture when config is unset."""

    @pytest.mark.asyncio
    async def test_env_var_enables_capture_when_config_unset(self, monkeypatch, tracer_from_provider, exporter):
        """Config without enable_content_capture + env=true → capture is active."""
        monkeypatch.setenv(OtelContentCapture.CAPTURE_CONTENT_ENV, "true")

        with patch.object(telemetry, "_tracer", tracer_from_provider):
            with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
                # _make_tracing_config has tracing on but enable_content_capture unset
                config = RailsConfig.from_content(config=_make_tracing_config())
                iorails = IORails(config)
            async with iorails:
                _stub_deep_pipeline(iorails)
                await iorails.generate_async([{"role": "user", "content": "hi"}])

        span = _request_span(exporter.get_finished_spans())
        # Request span carries guardrails.request.* attrs
        assert GuardrailsAttributes.REQUEST_INPUT in span.attributes
        assert GuardrailsAttributes.REQUEST_OUTPUT in span.attributes

    @pytest.mark.asyncio
    async def test_env_var_false_disables_capture_when_config_true(self, monkeypatch, tracer_from_provider, exporter):
        """config enable_content_capture=True + env=false → capture inactive.

        End-to-end counterpart to the unit-level
        test_env_var_falsy_disables_capture_even_when_config_true: confirms
        the env-var-wins semantic holds through the full IORails pipeline,
        not just the is_content_capture_enabled helper in isolation."""
        monkeypatch.setenv(OtelContentCapture.CAPTURE_CONTENT_ENV, "false")

        with patch.object(telemetry, "_tracer", tracer_from_provider):
            with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
                # _make_content_capture_config sets enable_content_capture=True
                config = RailsConfig.from_content(config=_make_content_capture_config())
                iorails = IORails(config)
            async with iorails:
                _stub_deep_pipeline(iorails)
                await iorails.generate_async([{"role": "user", "content": "hi"}])

        span = _request_span(exporter.get_finished_spans())
        # No content attrs despite config=True — env=false wins
        assert GuardrailsAttributes.REQUEST_INPUT not in span.attributes
        assert GuardrailsAttributes.REQUEST_OUTPUT not in span.attributes
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes


class TestRailContentCapture:
    """guardrails.rail.input on every rail span; guardrails.rail.reason on blocked rails only."""

    @pytest.mark.asyncio
    async def test_rail_input_recorded_on_passing_rail(self, iorails_content_capture, exporter):
        """Every rail span records its rail.input; passing rails carry no rail.reason."""
        _stub_deep_pipeline(iorails_content_capture)

        await iorails_content_capture.generate_async([{"role": "user", "content": "hi"}])

        rail_spans = [s for s in exporter.get_finished_spans() if s.name == "guardrails.rail"]
        assert len(rail_spans) >= 1
        for span in rail_spans:
            rail_input = json.loads(span.attributes[GuardrailsAttributes.RAIL_INPUT])
            assert "messages" in rail_input
            assert rail_input["messages"] == [{"role": "user", "content": "hi"}]
            # Passing rails carry no block reason
            assert GuardrailsAttributes.RAIL_REASON not in span.attributes

    @pytest.mark.asyncio
    async def test_rail_reason_set_on_blocked_rail_only(self, iorails_content_capture, exporter):
        """When an input rail blocks, its span gets a reason; later rails never run."""
        _stub_deep_pipeline(iorails_content_capture, input_safe=False)

        await iorails_content_capture.generate_async([{"role": "user", "content": "bad"}])

        rail_spans = [s for s in exporter.get_finished_spans() if s.name == "guardrails.rail"]
        blocked = [s for s in rail_spans if GuardrailsAttributes.RAIL_REASON in s.attributes]
        # Exactly one rail blocked (sequential mode short-circuits)
        assert len(blocked) == 1
        # The blocking rail's reason is a non-empty string
        assert blocked[0].attributes[GuardrailsAttributes.RAIL_REASON]


class TestStreamingContentCapture:
    """Streamed delta_content accumulates and lands on the request + LLM spans."""

    @pytest.mark.asyncio
    async def test_output_text_recorded_on_request_span(self, iorails_streaming_content_capture, exporter):
        """Streamed chunks accumulate and the joined text lands on the request span."""
        iorails = iorails_streaming_content_capture
        _stub_deep_streaming_pipeline(iorails)

        chunks = [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]
        delivered = "".join(c for c in chunks if isinstance(c, str))
        assert delivered  # sanity: stream produced something

        span = _request_span(exporter.get_finished_spans())
        # Request span: guardrails.request.* attrs carry the delivered text
        assert span.attributes[GuardrailsAttributes.REQUEST_OUTPUT] == delivered

    @pytest.mark.asyncio
    async def test_output_text_recorded_on_streaming_llm_span(self, iorails_streaming_content_capture, exporter):
        """Streamed chunks accumulate and the joined text lands on the LLM span too."""
        iorails = iorails_streaming_content_capture
        _stub_deep_streaming_pipeline(iorails)

        chunks = [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]
        delivered = "".join(c for c in chunks if isinstance(c, str))

        span = _main_llm_span(exporter.get_finished_spans())
        choice = next(e for e in span.events if e.name == "gen_ai.choice")
        assert dict(choice.attributes)["message.content"] == delivered

    @pytest.mark.asyncio
    async def test_blocked_input_records_refusal_as_output(self, iorails_streaming_content_capture, exporter):
        """Input-rail block: REFUSAL_MESSAGE is pushed through the streaming
        handler, so the consumer receives it and content capture records it
        as the assistant output on the request span (not empty)."""
        iorails = iorails_streaming_content_capture
        _stub_deep_streaming_pipeline(iorails, input_safe=False)

        [c async for c in iorails.stream_async([{"role": "user", "content": "bad"}])]

        span = _request_span(exporter.get_finished_spans())
        assert span.attributes[GuardrailsAttributes.REQUEST_OUTPUT] == REFUSAL_MESSAGE

    @pytest.mark.asyncio
    async def test_empty_delivered_records_no_output_attr(self, iorails_streaming_content_capture, exporter):
        """When the LLM yields zero content chunks, guardrails.request.output is absent.

        Guards against an empty-string output being recorded when the stream
        produced nothing — None output_text means no attribute is set."""
        iorails = iorails_streaming_content_capture

        async def _empty_stream(messages, **kwargs):
            if False:
                yield  # pragma: no cover

        _stub_deep_streaming_pipeline(iorails, main_stream=_empty_stream)

        [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]

        spans = exporter.get_finished_spans()
        request_span = _request_span(spans)
        # Input is captured; output is absent (empty delivered)
        assert GuardrailsAttributes.REQUEST_INPUT in request_span.attributes
        assert GuardrailsAttributes.REQUEST_OUTPUT not in request_span.attributes

        # LLM span also has no output — empty content_parts → None
        llm_span = _main_llm_span(spans)
        assert all(e.name != "gen_ai.choice" for e in llm_span.events)

    @pytest.mark.asyncio
    async def test_dict_chunks_with_include_metadata_get_captured(self, iorails_streaming_content_capture, exporter):
        """include_metadata=True streams dict chunks; capture extracts non-empty text fields.

        Covers the isinstance(chunk, dict) branch in _wrapped_iterator's
        accumulator.  Also verifies that dict chunks with an empty-string
        ``text`` field (metadata-only frames) are excluded from the captured
        output — they must not contribute empty strings to the join and must
        not cause a spurious empty assistant output where None is correct.
        """
        iorails = iorails_streaming_content_capture

        async def _stream_with_empty_frame(messages, **kwargs):
            """Inject an empty-text metadata frame between real content chunks."""
            yield LLMResponseChunk(delta_content="Hello")
            yield LLMResponseChunk(delta_content="")  # empty delta — metadata frame
            yield LLMResponseChunk(delta_content=" world")

        _stub_deep_streaming_pipeline(iorails, main_stream=_stream_with_empty_frame)

        chunks = [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}], include_metadata=True)]
        assert all(isinstance(c, dict) for c in chunks)
        # Expected: only the non-empty text parts joined; empty delta excluded
        expected = "Hello world"

        span = _request_span(exporter.get_finished_spans())
        assert span.attributes[GuardrailsAttributes.REQUEST_OUTPUT] == expected


class TestStreamingContentCaptureJsonFormat:
    """Streaming + capture on + OTEL_SEMCONV_STABILITY_OPT_IN: JSON attrs on both spans."""

    @pytest.fixture(autouse=True)
    def _set_stability_opt_in(self, monkeypatch):
        monkeypatch.setenv(
            OtelContentCapture.STABILITY_OPT_IN_ENV,
            OtelContentCapture.STABILITY_OPT_IN_LATEST,
        )

    @pytest.mark.asyncio
    async def test_json_attrs_on_request_span(self, iorails_streaming_content_capture, exporter):
        """Streaming + opt-in → request span carries JSON input/output attrs, no events."""
        iorails = iorails_streaming_content_capture
        _stub_deep_streaming_pipeline(iorails)

        chunks = [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]
        delivered = "".join(c for c in chunks if isinstance(c, str))

        span = _request_span(exporter.get_finished_spans())
        # Request span: guardrails.request.* plain-string attrs regardless of opt-in format
        assert json.loads(span.attributes[GuardrailsAttributes.REQUEST_INPUT]) == [{"role": "user", "content": "hi"}]
        assert span.attributes[GuardrailsAttributes.REQUEST_OUTPUT] == delivered
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes

    @pytest.mark.asyncio
    async def test_json_attrs_on_streaming_llm_span(self, iorails_streaming_content_capture, exporter):
        """Streaming + opt-in → the LLM span's JSON output.messages holds the joined stream."""
        iorails = iorails_streaming_content_capture
        _stub_deep_streaming_pipeline(iorails)

        chunks = [c async for c in iorails.stream_async([{"role": "user", "content": "hi"}])]
        delivered = "".join(c for c in chunks if isinstance(c, str))

        span = _main_llm_span(exporter.get_finished_spans())
        outputs = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        assert outputs[0]["parts"][0]["content"] == delivered

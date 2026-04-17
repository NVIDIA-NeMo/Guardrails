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
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode, format_trace_id

from nemoguardrails.guardrails import telemetry
from nemoguardrails.guardrails.guardrails_types import REQUEST_ID_HEX_CHARS, RailResult, get_request_id
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.rails.llm.config import RailsConfig
from tests.guardrails.test_data import NEMOGUARDS_CONFIG
from tests.guardrails.test_telemetry import _is_valid_hex_string


def _make_tracing_config():
    """Return a NEMOGUARDS_CONFIG copy with tracing enabled."""
    cfg = copy.deepcopy(NEMOGUARDS_CONFIG)
    cfg["tracing"] = {"enabled": True}
    return cfg


def _stub_safe_pipeline(iorails, llm_response="Hello"):
    """Mock input/output rails as safe and the LLM to return *llm_response*."""
    iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
    iorails.engine_registry.model_call = AsyncMock(return_value=llm_response)
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


@pytest.fixture
def iorails_tracing(tracer_from_provider):
    """IORails instance with tracing enabled, using a test tracer.

    Patches the module-level ``_tracer`` before constructing IORails so that
    ``IORails.__init__`` picks up the test tracer via ``get_tracer()`` and
    threads it through EngineRegistry/RailsManager/RailAction constructors.
    """
    with patch.object(telemetry, "_tracer", tracer_from_provider):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            config = RailsConfig.from_content(config=_make_tracing_config())
            iorails = IORails(config)
        yield iorails


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def iorails_no_tracing():
    """IORails instance with default config (tracing disabled)."""
    return IORails(RailsConfig.from_content(config=NEMOGUARDS_CONFIG))


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
            return "Generated response"

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
                engine.chat_completion = AsyncMock(return_value=main_llm_response)
            elif name == "content_safety":
                # Content safety output parser needs Response Safety field
                engine.chat_completion = AsyncMock(return_value=SAFE_OUTPUT_JSON if input_safe else input_json)
            else:
                engine.chat_completion = AsyncMock(return_value=input_json)
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
                engine.chat_completion = AsyncMock(return_value=SAFE_INPUT_JSON)

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
    "tracing": {"enabled": True},
}

_INPUT_ONLY_STREAMING_NO_TRACING_CONFIG = {
    **NEMOGUARDS_CONFIG,
    "rails": {
        **NEMOGUARDS_CONFIG["rails"],
        "output": {"flows": []},
    },
}


def _make_output_streaming_tracing_config(*, stream_first=True):
    """Config with output-rail streaming + tracing enabled."""
    base = copy.deepcopy(NEMOGUARDS_CONFIG)
    base["rails"]["output"]["streaming"] = {
        "enabled": True,
        "chunk_size": 5,
        "context_size": 2,
        "stream_first": stream_first,
    }
    base["tracing"] = {"enabled": True}
    return base


async def _mock_chunks_stream(model_type, messages, **kwargs):
    """stream_model_call-level mock yielding three string chunks."""
    for chunk in ["Hello", " ", "world"]:
        yield chunk


async def _engine_default_stream(messages, **kwargs):
    """ModelEngine.stream_chat_completion-level mock for the main LLM."""
    for chunk in ["Hello", " from", " the", " stream"]:
        yield chunk


async def _engine_failing_stream(messages, **kwargs):
    """ModelEngine.stream_chat_completion-level mock that raises mid-stream."""
    yield "Hello"
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
                engine.chat_completion = AsyncMock(return_value=SAFE_OUTPUT_JSON if input_safe else input_json)
            else:
                engine.chat_completion = AsyncMock(return_value=input_json)
        elif isinstance(engine, APIEngine):
            engine.call = AsyncMock(return_value={"jailbreak": False, "score": 0.01})


@pytest.fixture
def iorails_streaming_input_only_tracing(tracer_from_provider):
    """Input-rails only + streaming + tracing enabled."""
    with patch.object(telemetry, "_tracer", tracer_from_provider):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            config = RailsConfig.from_content(config=_INPUT_ONLY_STREAMING_TRACING_CONFIG)
            iorails = IORails(config)
        yield iorails


@pytest.fixture
def iorails_streaming_output_tracing(tracer_from_provider):
    """Full input+output streaming + tracing enabled."""
    with patch.object(telemetry, "_tracer", tracer_from_provider):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            config = RailsConfig.from_content(config=_make_output_streaming_tracing_config())
            iorails = IORails(config)
        yield iorails


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def iorails_streaming_no_tracing():
    """Input-rails only + streaming + tracing disabled."""
    return IORails(RailsConfig.from_content(config=_INPUT_ONLY_STREAMING_NO_TRACING_CONFIG))


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

            _stub_safe_pipeline(iorails)

            result = await iorails.generate_async([{"role": "user", "content": "hi"}])

            assert result == {"role": "assistant", "content": "Hello"}
            assert iorails._tracing_enabled is False
            assert len(exporter.get_finished_spans()) == 0

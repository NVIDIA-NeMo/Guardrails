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

"""Tests for speculative generation (M2): input rails race LLM generation."""

import asyncio
import copy
import json
import logging
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nemoguardrails.guardrails import telemetry
from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails, _is_stream_error_chunk
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.types import LLMResponse, LLMResponseChunk, UsageInfo
from tests.guardrails.async_helpers import started_iorails
from tests.guardrails.test_data import (
    NEMOGUARDS_CONFIG,
    NEMOGUARDS_SPECULATIVE_CONFIG,
    NEMOGUARDS_SPECULATIVE_STREAMING_CONFIG,
    NEMOGUARDS_SPECULATIVE_STREAMING_INPUT_ONLY_CONFIG,
    NEMOGUARDS_SPECULATIVE_STREAMING_SMALL_BUFFER_CONFIG,
)

MESSAGES = [{"role": "user", "content": "hi"}]


def _hi_model() -> AsyncMock:
    """Main-model mock returning a fixed 'Hi' response with usage."""
    return AsyncMock(
        return_value=LLMResponse(content="Hi", usage=UsageInfo(input_tokens=5, output_tokens=3, total_tokens=8))
    )


async def _slow_reject(messages, *, enabled=True):
    """Input rails that reject after a short delay (so generation finishes first)."""
    await asyncio.sleep(0.05)
    return RailResult(is_safe=False, reason="unsafe")


async def _immediate_reject(messages, *, enabled=True):
    """Input rails that reject immediately (so rails and generation finish in the same tick)."""
    return RailResult(is_safe=False, reason="unsafe")


@pytest_asyncio.fixture
async def iorails():
    async with started_iorails(NEMOGUARDS_SPECULATIVE_CONFIG) as instance:
        yield instance


@pytest_asyncio.fixture
async def iorails_sequential():
    async with started_iorails(NEMOGUARDS_CONFIG) as instance:
        yield instance


@pytest.fixture
def caplog_iorails(caplog):
    """Capture records from ``nemoguardrails.guardrails.iorails`` reliably.

    test_configure_logging.py sets ``propagate=False`` on the parent
    ``nemoguardrails.guardrails`` logger and only restores handlers (not
    propagation) on teardown, so once it runs first in a session caplog's
    root-attached handler stops seeing iorails records. Attach the handler
    directly to bypass the propagation gap, and locally disable propagation
    to prevent double-capture when the chain is intact.
    """
    iorails_logger = logging.getLogger("nemoguardrails.guardrails.iorails")
    original_propagate = iorails_logger.propagate
    iorails_logger.addHandler(caplog.handler)
    iorails_logger.propagate = False
    try:
        yield caplog
    finally:
        iorails_logger.removeHandler(caplog.handler)
        iorails_logger.propagate = original_propagate


class TestSpeculativeGeneration:
    """Speculative generation races input rails against LLM generation."""

    @pytest.mark.asyncio
    async def test_rails_first_pass(self, iorails):
        """Rails finish first and pass — generation is awaited, output rails run."""

        async def fast_rails(messages, *, enabled=True):
            return RailResult(is_safe=True)

        async def slow_llm(model_type, messages):
            await asyncio.sleep(0.05)
            return LLMResponse(content="Hello from LLM")

        iorails.rails_manager.is_input_safe = fast_rails
        iorails.engine_registry.model_call = slow_llm
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        result = await iorails.generate_async(messages=MESSAGES)

        assert result == {"role": "assistant", "content": "Hello from LLM"}

    @pytest.mark.asyncio
    async def test_rails_first_reject(self, iorails):
        """Rails finish first and reject — generation is cancelled, refusal returned."""
        llm_started = False
        llm_completed = False

        async def fast_reject(messages, *, enabled=True):
            return RailResult(is_safe=False, reason="unsafe")

        async def slow_llm(model_type, messages):
            nonlocal llm_started, llm_completed
            llm_started = True
            await asyncio.sleep(0.5)
            llm_completed = True
            return LLMResponse(content="Should not be used")

        iorails.rails_manager.is_input_safe = fast_reject
        iorails.engine_registry.model_call = slow_llm
        iorails.rails_manager.is_output_safe = AsyncMock()

        result = await iorails.generate_async(messages=MESSAGES)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        iorails.rails_manager.is_output_safe.assert_not_called()
        # The speculative LLM call must have been started but cancelled mid-flight.
        # Without these, the test would still pass on a regression where gen_task
        # silently completed in the background instead of being cancelled.
        assert llm_started, "LLM should have started speculatively"
        assert not llm_completed, "LLM should have been cancelled before completion"

    @pytest.mark.asyncio
    async def test_gen_first_pass(self, iorails):
        """Generation finishes first — rails verdict awaited, response served on pass."""

        async def slow_rails(messages, *, enabled=True):
            await asyncio.sleep(0.05)
            return RailResult(is_safe=True)

        async def fast_llm(model_type, messages):
            return LLMResponse(content="Fast LLM response")

        iorails.rails_manager.is_input_safe = slow_rails
        iorails.engine_registry.model_call = fast_llm
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        result = await iorails.generate_async(messages=MESSAGES)

        assert result == {"role": "assistant", "content": "Fast LLM response"}

    @pytest.mark.asyncio
    async def test_gen_first_reject(self, iorails):
        """Generation finishes first, then rails reject — response discarded."""

        async def slow_reject(messages, *, enabled=True):
            await asyncio.sleep(0.05)
            return RailResult(is_safe=False, reason="unsafe")

        async def fast_llm(model_type, messages):
            return LLMResponse(content="Should be discarded")

        iorails.rails_manager.is_input_safe = slow_reject
        iorails.engine_registry.model_call = fast_llm
        iorails.rails_manager.is_output_safe = AsyncMock()

        result = await iorails.generate_async(messages=MESSAGES)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        iorails.rails_manager.is_output_safe.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_error_cancels_rails(self, iorails):
        """LLM errors while rails still running — rails cancelled, error propagated."""

        async def slow_rails(messages, *, enabled=True):
            await asyncio.sleep(0.5)
            return RailResult(is_safe=True)

        iorails.rails_manager.is_input_safe = slow_rails
        iorails.engine_registry.model_call = AsyncMock(side_effect=RuntimeError("LLM crashed"))

        with pytest.raises(RuntimeError, match="LLM crashed"):
            await iorails.generate_async(messages=MESSAGES)

    @pytest.mark.asyncio
    async def test_rails_error_cancels_generation(self, iorails):
        """Rails error while LLM still running — generation cancelled, error propagated."""

        async def slow_llm(model_type, messages):
            await asyncio.sleep(0.5)
            return LLMResponse(content="Should not be used")

        iorails.rails_manager.is_input_safe = AsyncMock(side_effect=RuntimeError("Rails crashed"))
        iorails.engine_registry.model_call = slow_llm

        with pytest.raises(RuntimeError, match="Rails crashed"):
            await iorails.generate_async(messages=MESSAGES)

    @pytest.mark.asyncio
    async def test_rails_reject_with_simultaneous_llm_exception(self, iorails, caplog_iorails):
        """Rails reject + LLM raises in the same scheduling window — refusal returned, exception drained."""

        async def fast_reject(messages, *, enabled=True):
            return RailResult(is_safe=False, reason="unsafe")

        async def slow_raises(model_type, messages):
            # Yield once so rails wins the race, then raise — the cleanup path
            # must drain gen_task's stored exception via gather rather than
            # letting it leak through suppress(CancelledError).
            await asyncio.sleep(0)
            raise RuntimeError("LLM crashed late")

        iorails.rails_manager.is_input_safe = fast_reject
        iorails.engine_registry.model_call = slow_raises
        iorails.rails_manager.is_output_safe = AsyncMock()

        with caplog_iorails.at_level("WARNING", logger="nemoguardrails.guardrails.iorails"):
            result = await iorails.generate_async(messages=MESSAGES)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        iorails.rails_manager.is_output_safe.assert_not_called()
        assert any("LLM generation error suppressed" in rec.message for rec in caplog_iorails.records)

    @pytest.mark.asyncio
    async def test_both_tasks_raise_during_race(self, iorails, caplog_iorails):
        """Both rails and gen raise — outer cleanup logs the loser exception, winner propagates."""

        async def rails_raises(messages, *, enabled=True):
            raise RuntimeError("Rails crashed")

        async def gen_raises(model_type, messages):
            await asyncio.sleep(0)
            raise RuntimeError("LLM crashed too")

        iorails.rails_manager.is_input_safe = rails_raises
        iorails.engine_registry.model_call = gen_raises

        with caplog_iorails.at_level("WARNING", logger="nemoguardrails.guardrails.iorails"):
            with pytest.raises(RuntimeError):
                await iorails.generate_async(messages=MESSAGES)

        assert any("task error discarded during cleanup" in rec.message for rec in caplog_iorails.records)

    @pytest.mark.asyncio
    async def test_rails_first_reject_records_blocked_metric(self):
        """Rails-first-reject path increments record_request_blocked when metrics are on."""
        cfg = copy.deepcopy(NEMOGUARDS_SPECULATIVE_CONFIG)
        cfg["metrics"] = {"enabled": True}

        async def fast_reject(messages, *, enabled=True):
            return RailResult(is_safe=False, reason="unsafe")

        async def slow_llm(model_type, messages):
            await asyncio.sleep(0.5)
            return LLMResponse(content="Should not be used")

        async with started_iorails(cfg) as iorails:
            iorails.rails_manager.is_input_safe = fast_reject
            iorails.engine_registry.model_call = slow_llm
            iorails.rails_manager.is_output_safe = AsyncMock()

            with patch("nemoguardrails.guardrails.iorails.record_request_blocked") as record_mock:
                result = await iorails.generate_async(messages=MESSAGES)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        record_mock.assert_called_once()
        assert record_mock.call_args.args[0].name == "INPUT"

    @pytest.mark.asyncio
    async def test_gen_first_reject_records_blocked_metric(self):
        """Gen-first-reject path increments record_request_blocked when metrics are on."""
        cfg = copy.deepcopy(NEMOGUARDS_SPECULATIVE_CONFIG)
        cfg["metrics"] = {"enabled": True}

        async def slow_reject(messages, *, enabled=True):
            await asyncio.sleep(0.05)
            return RailResult(is_safe=False, reason="unsafe")

        async def fast_llm(model_type, messages):
            return LLMResponse(content="Should be discarded")

        async with started_iorails(cfg) as iorails:
            iorails.rails_manager.is_input_safe = slow_reject
            iorails.engine_registry.model_call = fast_llm
            iorails.rails_manager.is_output_safe = AsyncMock()

            with patch("nemoguardrails.guardrails.iorails.record_request_blocked") as record_mock:
                result = await iorails.generate_async(messages=MESSAGES)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        record_mock.assert_called_once()
        assert record_mock.call_args.args[0].name == "INPUT"

    @pytest.mark.asyncio
    async def test_flag_disabled_runs_sequentially(self, iorails_sequential):
        """When speculative_generation is false, pipeline runs sequentially."""
        call_order = []

        async def mock_input(messages, *, enabled=True):
            call_order.append("input")
            return RailResult(is_safe=True)

        async def mock_generate(model_type, messages):
            call_order.append("generate")
            return LLMResponse(content="response")

        async def mock_output(messages, response, *, enabled=True):
            call_order.append("output")
            return RailResult(is_safe=True)

        iorails_sequential.rails_manager.is_input_safe = mock_input
        iorails_sequential.engine_registry.model_call = mock_generate
        iorails_sequential.rails_manager.is_output_safe = mock_output

        await iorails_sequential.generate_async(messages=MESSAGES)
        assert call_order == ["input", "generate", "output"]


# ── OTEL fixtures for speculative-generation span attribute tests ──


@pytest.fixture
def span_exporter():
    return InMemorySpanExporter()


@pytest.fixture
def test_tracer(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider.get_tracer("test")


def _make_speculative_tracing_config():
    cfg = copy.deepcopy(NEMOGUARDS_SPECULATIVE_CONFIG)
    cfg["tracing"] = {"enabled": True}
    return cfg


@pytest_asyncio.fixture
async def iorails_speculative_tracing(test_tracer):
    """IORails with speculative generation + OTEL tracing, backed by an in-memory exporter."""
    with patch.object(telemetry, "_tracer", test_tracer):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            config = RailsConfig.from_content(config=_make_speculative_tracing_config())
            iorails = IORails(config)
        async with iorails:
            yield iorails


class TestSpeculativeGenerationTelemetry:
    """Verify OTEL span attributes for all (first_completed, first_rejector) permutations."""

    @pytest.mark.asyncio
    async def test_rails_first_pass_span_attrs(self, iorails_speculative_tracing, span_exporter):
        """Rails finish first and pass — first_completed=input_rails, first_rejector=none."""

        async def fast_rails(messages, *, enabled=True):
            return RailResult(is_safe=True)

        async def slow_llm(model_type, messages):
            await asyncio.sleep(0.05)
            return LLMResponse(content="Hello from LLM")

        iorails_speculative_tracing.rails_manager.is_input_safe = fast_rails
        iorails_speculative_tracing.engine_registry.model_call = slow_llm
        iorails_speculative_tracing.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        result = await iorails_speculative_tracing.generate_async(messages=MESSAGES)

        assert result == {"role": "assistant", "content": "Hello from LLM"}
        spans = span_exporter.get_finished_spans()
        request_spans = [s for s in spans if s.name == "guardrails.request"]
        assert len(request_spans) == 1
        attrs = dict(request_spans[0].attributes)
        assert attrs["speculative_generation.mode_active"] is True
        assert attrs["speculative_generation.first_completed"] == "input_rails"
        assert attrs["speculative_generation.first_rejector"] == "none"

    @pytest.mark.asyncio
    async def test_rails_first_reject_span_attrs(self, iorails_speculative_tracing, span_exporter):
        """Rails finish first and reject — first_completed=input_rails, first_rejector=input_rails."""

        async def fast_reject(messages, *, enabled=True):
            return RailResult(is_safe=False, reason="unsafe")

        async def slow_llm(model_type, messages):
            await asyncio.sleep(0.5)
            return LLMResponse(content="Should not be used")

        iorails_speculative_tracing.rails_manager.is_input_safe = fast_reject
        iorails_speculative_tracing.engine_registry.model_call = slow_llm
        iorails_speculative_tracing.rails_manager.is_output_safe = AsyncMock()

        result = await iorails_speculative_tracing.generate_async(messages=MESSAGES)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        spans = span_exporter.get_finished_spans()
        request_spans = [s for s in spans if s.name == "guardrails.request"]
        assert len(request_spans) == 1
        attrs = dict(request_spans[0].attributes)
        assert attrs["speculative_generation.mode_active"] is True
        assert attrs["speculative_generation.first_completed"] == "input_rails"
        assert attrs["speculative_generation.first_rejector"] == "input_rails"

    @pytest.mark.asyncio
    async def test_gen_first_pass_span_attrs(self, iorails_speculative_tracing, span_exporter):
        """Generation finishes first, rails pass — first_completed=generation, first_rejector=none."""

        async def slow_rails(messages, *, enabled=True):
            await asyncio.sleep(0.05)
            return RailResult(is_safe=True)

        async def fast_llm(model_type, messages):
            return LLMResponse(content="Fast LLM response")

        iorails_speculative_tracing.rails_manager.is_input_safe = slow_rails
        iorails_speculative_tracing.engine_registry.model_call = fast_llm
        iorails_speculative_tracing.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        result = await iorails_speculative_tracing.generate_async(messages=MESSAGES)

        assert result == {"role": "assistant", "content": "Fast LLM response"}
        spans = span_exporter.get_finished_spans()
        request_spans = [s for s in spans if s.name == "guardrails.request"]
        assert len(request_spans) == 1
        attrs = dict(request_spans[0].attributes)
        assert attrs["speculative_generation.mode_active"] is True
        assert attrs["speculative_generation.first_completed"] == "generation"
        assert attrs["speculative_generation.first_rejector"] == "none"

    @pytest.mark.asyncio
    async def test_gen_first_reject_span_attrs(self, iorails_speculative_tracing, span_exporter):
        """Generation finishes first, then rails reject — first_completed=generation, first_rejector=input_rails."""

        async def slow_reject(messages, *, enabled=True):
            await asyncio.sleep(0.05)
            return RailResult(is_safe=False, reason="unsafe")

        async def fast_llm(model_type, messages):
            return LLMResponse(content="Should be discarded")

        iorails_speculative_tracing.rails_manager.is_input_safe = slow_reject
        iorails_speculative_tracing.engine_registry.model_call = fast_llm
        iorails_speculative_tracing.rails_manager.is_output_safe = AsyncMock()

        result = await iorails_speculative_tracing.generate_async(messages=MESSAGES)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        spans = span_exporter.get_finished_spans()
        request_spans = [s for s in spans if s.name == "guardrails.request"]
        assert len(request_spans) == 1
        attrs = dict(request_spans[0].attributes)
        assert attrs["speculative_generation.mode_active"] is True
        assert attrs["speculative_generation.first_completed"] == "generation"
        assert attrs["speculative_generation.first_rejector"] == "input_rails"

    @pytest.mark.asyncio
    async def test_sequential_mode_has_no_speculative_attrs(self, test_tracer, span_exporter):
        """When speculative_generation is disabled, no speculative attributes are set."""
        cfg = copy.deepcopy(NEMOGUARDS_CONFIG)
        cfg["tracing"] = {"enabled": True}
        with patch.object(telemetry, "_tracer", test_tracer):
            with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
                iorails = IORails(RailsConfig.from_content(config=cfg))
            async with iorails:
                iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
                iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="response"))
                iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

                await iorails.generate_async(messages=MESSAGES)

        spans = span_exporter.get_finished_spans()
        request_spans = [s for s in spans if s.name == "guardrails.request"]
        assert len(request_spans) == 1
        attrs = dict(request_spans[0].attributes)
        assert "speculative_generation.mode_active" not in attrs
        assert "speculative_generation.first_completed" not in attrs
        assert "speculative_generation.first_rejector" not in attrs


# ── Streaming speculative generation (SG2) ──

STREAM_TOKENS = ["Hello", " from", " the", " streaming", " LLM", "!"]


def _token_stream(*, delay: float = 0.0, tokens=STREAM_TOKENS, emitted=None):
    """Build a mock LLM stream over ``tokens`` with an optional per-token delay.

    ``emitted`` (if given) records each token as it is produced, so tests can
    assert whether the generation ran to completion or was aborted mid-stream.
    """

    async def _stream(model_type, messages, **kwargs):
        for tok in tokens:
            if delay:
                await asyncio.sleep(delay)
            if emitted is not None:
                emitted.append(tok)
            yield LLMResponseChunk(delta_content=tok)

    return _stream


async def _collect(async_iter):
    return [chunk async for chunk in async_iter]


def _content_of(chunks):
    """Join the plain content chunks, excluding any error/violation payloads."""
    return "".join(c for c in chunks if isinstance(c, str) and not _is_stream_error_chunk(c))


def _error_chunks(chunks):
    return [c for c in chunks if isinstance(c, str) and _is_stream_error_chunk(c)]


def _wire_stream(iorails, *, input_safe=True, input_delay=0.0, output_safe=True, stream=None):
    """Attach input/tool/output rail mocks and a streaming LLM to an IORails instance."""

    async def _input(messages, *, enabled=True):
        if input_delay:
            await asyncio.sleep(input_delay)
        return RailResult(is_safe=input_safe, reason=None if input_safe else "blocked")

    iorails.rails_manager.are_tool_results_safe = AsyncMock(return_value=RailResult(is_safe=True))
    iorails.rails_manager.is_input_safe = _input
    iorails.rails_manager.is_output_safe = AsyncMock(
        return_value=RailResult(is_safe=output_safe, reason=None if output_safe else "blocked")
    )
    iorails.engine_registry.stream_model_call = stream if stream is not None else _token_stream()


@pytest_asyncio.fixture
async def iorails_spec_stream():
    """Speculative streaming with output rails (check-first)."""
    async with started_iorails(NEMOGUARDS_SPECULATIVE_STREAMING_CONFIG) as instance:
        yield instance


@pytest_asyncio.fixture
async def iorails_spec_stream_input_only():
    """Speculative streaming with input rails only (buffer-and-release path)."""
    async with started_iorails(NEMOGUARDS_SPECULATIVE_STREAMING_INPUT_ONLY_CONFIG) as instance:
        yield instance


@pytest_asyncio.fixture
async def iorails_spec_stream_small_buffer():
    """Speculative streaming with a 2-chunk release buffer (overflow path)."""
    async with started_iorails(NEMOGUARDS_SPECULATIVE_STREAMING_SMALL_BUFFER_CONFIG) as instance:
        yield instance


class TestSpeculativeStreaming:
    """Streaming speculation (SG2): input rails race the LLM + output rails."""

    @pytest.mark.asyncio
    async def test_streaming_pass_rails_first(self, iorails_spec_stream):
        """Fast passing input rails: all validated tokens are delivered, no error."""
        _wire_stream(iorails_spec_stream, stream=_token_stream(delay=0.005))
        chunks = await _collect(iorails_spec_stream.stream_async(MESSAGES))
        assert _content_of(chunks) == "".join(STREAM_TOKENS)
        assert not _error_chunks(chunks)

    @pytest.mark.asyncio
    async def test_streaming_pass_held_then_released(self, iorails_spec_stream):
        """Slow (but passing) input rails: tokens are held during speculation, then flushed."""
        _wire_stream(iorails_spec_stream, input_delay=0.2, stream=_token_stream(delay=0.0))
        chunks = await _collect(iorails_spec_stream.stream_async(MESSAGES))
        assert _content_of(chunks) == "".join(STREAM_TOKENS)
        assert not _error_chunks(chunks)

    @pytest.mark.asyncio
    async def test_streaming_input_reject_no_leak_and_aborts(self, iorails_spec_stream):
        """Input rails reject: no content leaks, refusal emitted, generation aborted mid-stream."""
        emitted = []
        many = [f"tok{i} " for i in range(50)]
        _wire_stream(
            iorails_spec_stream,
            input_safe=False,
            input_delay=0.02,
            stream=_token_stream(delay=0.01, tokens=many, emitted=emitted),
        )
        chunks = await asyncio.wait_for(_collect(iorails_spec_stream.stream_async(MESSAGES)), timeout=3.0)

        assert _content_of(chunks) == ""  # nothing leaked before the input verdict
        errs = _error_chunks(chunks)
        assert errs and json.loads(errs[0])["error"]["param"] == "input_rails"
        assert len(emitted) < len(many), "generation should be aborted, not run to completion"

    @pytest.mark.asyncio
    async def test_streaming_output_early_reject(self, iorails_spec_stream):
        """Output rails reject during speculation: refusal before input verdict, no leak."""
        emitted = []
        many = [f"tok{i} " for i in range(50)]
        _wire_stream(
            iorails_spec_stream,
            input_safe=True,
            input_delay=1.0,  # input rails still running when output rejects
            output_safe=False,
            stream=_token_stream(delay=0.01, tokens=many, emitted=emitted),
        )
        chunks = await asyncio.wait_for(_collect(iorails_spec_stream.stream_async(MESSAGES)), timeout=3.0)

        assert _content_of(chunks) == ""
        errs = _error_chunks(chunks)
        assert errs and json.loads(errs[0])["error"]["param"] == "output_rails"

    @pytest.mark.asyncio
    async def test_streaming_backpressure_no_drop(self, iorails_spec_stream_small_buffer):
        """A small release buffer applies backpressure without dropping tokens.

        The cap comes from config, not a patched private attribute, so this also
        covers the ``rails.input.speculative_max_buffered_chunks`` -> ``IORails``
        wiring.
        """
        io = iorails_spec_stream_small_buffer
        assert io._speculative_max_buffered_chunks == 2, "config value must reach the engine"

        _wire_stream(io, input_delay=0.05, stream=_token_stream(delay=0.0))
        chunks = await _collect(io.stream_async(MESSAGES))
        assert _content_of(chunks) == "".join(STREAM_TOKENS)

    @pytest.mark.asyncio
    async def test_streaming_input_only_buffer_and_release(self, iorails_spec_stream_input_only):
        """No output rails: raw tokens are held until input passes, then released."""
        _wire_stream(iorails_spec_stream_input_only, input_delay=0.1, stream=_token_stream(delay=0.0))
        chunks = await _collect(iorails_spec_stream_input_only.stream_async(MESSAGES))
        assert _content_of(chunks) == "".join(STREAM_TOKENS)

    @pytest.mark.asyncio
    async def test_streaming_input_only_reject_no_leak(self, iorails_spec_stream_input_only):
        """No output rails + input reject: no raw tokens leak, refusal emitted."""
        many = [f"t{i} " for i in range(30)]
        _wire_stream(
            iorails_spec_stream_input_only,
            input_safe=False,
            input_delay=0.02,
            stream=_token_stream(delay=0.01, tokens=many),
        )
        chunks = await asyncio.wait_for(_collect(iorails_spec_stream_input_only.stream_async(MESSAGES)), timeout=3.0)
        assert _content_of(chunks) == ""
        assert _error_chunks(chunks)

    @pytest.mark.asyncio
    async def test_streaming_tool_result_reject_uses_tool_input_param(self, iorails_spec_stream_input_only):
        """Tool-result-rail rejection surfaces param=tool_input_rails, matching the non-speculative path."""
        io = iorails_spec_stream_input_only
        io.rails_manager.are_tool_results_safe = AsyncMock(
            return_value=RailResult(is_safe=False, reason="unlinked tool result")
        )
        io.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        io.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))
        io.engine_registry.stream_model_call = _token_stream(delay=0.01)

        chunks = await asyncio.wait_for(_collect(io.stream_async(MESSAGES)), timeout=3.0)

        assert _content_of(chunks) == ""
        errs = _error_chunks(chunks)
        assert errs and json.loads(errs[0])["error"]["param"] == "tool_input_rails"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failing_rail", ["is_input_safe", "are_tool_results_safe"])
    async def test_streaming_rail_exception_reports_error_not_block(self, iorails_spec_stream_input_only, failing_rail):
        """A rail that *raises* is reported as an error, not as a policy refusal.

        The stream must not crash (``stream_async`` is an async generator, so by
        this point a server has already committed a 200 OK and a raise would
        truncate the SSE response), and it must not masquerade as a block: an
        outage is ``generation_error``/``generation_failed``, matching the
        non-speculative ``_generation_task`` path, while a refusal stays
        ``guardrails_violation``/``content_blocked``.
        """
        io = iorails_spec_stream_input_only
        io.rails_manager.are_tool_results_safe = AsyncMock(return_value=RailResult(is_safe=True))
        io.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        io.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))
        setattr(io.rails_manager, failing_rail, AsyncMock(side_effect=RuntimeError("boom")))
        io.engine_registry.stream_model_call = _token_stream(delay=0.01)

        chunks = await asyncio.wait_for(_collect(io.stream_async(MESSAGES)), timeout=3.0)

        assert _content_of(chunks) == ""  # nothing leaks on failure
        errs = _error_chunks(chunks)
        assert errs
        error = json.loads(errs[0])["error"]
        assert error["type"] == "generation_error"
        assert error["code"] == "generation_failed"
        assert "boom" in error["message"]
        # No ``param``: that field distinguishes rail families on a *block*, and
        # this is not one.  Matches the _generation_task payload shape exactly.
        assert "param" not in error

    @pytest.mark.asyncio
    async def test_streaming_json_with_string_error_value_does_not_crash(self, iorails_spec_stream_input_only):
        """A chunk of ``{"error": "..."}`` must not crash the gate.

        ``_is_stream_error_chunk`` only checks that an ``error`` key exists, so a
        non-dict value reaches the output-violation check. Reading ``error.param``
        off a string raised ``AttributeError`` — uncaught, since the handler only
        covered JSONDecodeError/TypeError — which escaped ``_gate_on_input`` and
        killed the stream.

        The chunk is still forwarded and still ends the stream (that is
        ``_is_stream_error_chunk``'s pre-existing behavior for anything shaped like
        an error payload); what this locks in is that it is not misclassified as an
        *output-rails violation*, and above all that it does not raise.
        """
        io = iorails_spec_stream_input_only
        content = '{"error": "connection refused"}'
        _wire_stream(io, stream=_token_stream(tokens=[content]))

        chunks = await asyncio.wait_for(_collect(io.stream_async(MESSAGES)), timeout=3.0)

        assert [c for c in chunks if isinstance(c, str)] == [content]

    @pytest.mark.asyncio
    async def test_streaming_both_rails_reject_first_rejector_wins(self, iorails_spec_stream):
        """Unsafe input AND unsafe output: whichever lands first wins, cleanly.

        Output rails run on the first buffered batch while input rails are still
        pending, so output rejects first and short-circuits before the input
        verdict — one error chunk, no leak, no second refusal appended.
        """
        many = [f"tok{i} " for i in range(50)]
        _wire_stream(
            iorails_spec_stream,
            input_safe=False,
            input_delay=1.0,
            output_safe=False,
            stream=_token_stream(delay=0.01, tokens=many),
        )
        chunks = await asyncio.wait_for(_collect(iorails_spec_stream.stream_async(MESSAGES)), timeout=3.0)

        assert _content_of(chunks) == ""
        errs = _error_chunks(chunks)
        assert len(errs) == 1, "exactly one rejection should reach the client"
        assert json.loads(errs[0])["error"]["param"] == "output_rails"

    @pytest.mark.asyncio
    async def test_output_rails_run_at_generation_rate_not_in_a_burst(self, iorails_spec_stream):
        """Output-rail calls are spread across generation, not clustered at release.

        This is the property that motivated Option C over buffer-then-release
        (design doc S4): output rails sit *upstream* of the hold buffer, so they
        process at the LLM's natural rate and the release flushes already-validated
        chunks without firing a burst of safety-model calls. Guards against a
        refactor that reorders the gate and the output rails.
        """
        call_times = []

        async def _timed_output(messages, chunk, *, enabled=True):
            call_times.append(asyncio.get_running_loop().time())
            return RailResult(is_safe=True)

        many = [f"tok{i} " for i in range(30)]
        _wire_stream(iorails_spec_stream, input_delay=0.25, stream=_token_stream(delay=0.01, tokens=many))
        iorails_spec_stream.rails_manager.is_output_safe = _timed_output

        await asyncio.wait_for(_collect(iorails_spec_stream.stream_async(MESSAGES)), timeout=5.0)

        assert len(call_times) >= 3, "need several batches to judge the spread"
        span = call_times[-1] - call_times[0]
        # 30 tokens at 10ms is ~300ms of generation. A burst would collapse every
        # call into one tick; spread across even a fraction of that window proves
        # the rails ran during generation rather than after the release.
        assert span > 0.05, f"output-rail calls clustered in {span:.3f}s — burst regression"


def _make_speculative_streaming_tracing_config():
    cfg = copy.deepcopy(NEMOGUARDS_SPECULATIVE_STREAMING_CONFIG)
    cfg["tracing"] = {"enabled": True}
    return cfg


@pytest_asyncio.fixture
async def iorails_spec_stream_tracing(test_tracer):
    """Speculative streaming with OTEL tracing, backed by an in-memory exporter."""
    with patch.object(telemetry, "_tracer", test_tracer):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            config = RailsConfig.from_content(config=_make_speculative_streaming_tracing_config())
            iorails = IORails(config)
        async with iorails:
            yield iorails


class TestSpeculativeStreamingTelemetry:
    """Verify speculative span attributes for streaming outcomes."""

    async def _request_attrs(self, span_exporter):
        spans = span_exporter.get_finished_spans()
        request_spans = [s for s in spans if s.name == "guardrails.request"]
        assert len(request_spans) == 1
        return dict(request_spans[0].attributes)

    @pytest.mark.asyncio
    async def test_streaming_pass_span_attrs(self, iorails_spec_stream_tracing, span_exporter):
        """Passing stream: mode_active, no rejector, time_saved recorded."""
        _wire_stream(iorails_spec_stream_tracing, stream=_token_stream(delay=0.0))
        await _collect(iorails_spec_stream_tracing.stream_async(MESSAGES))

        attrs = await self._request_attrs(span_exporter)
        assert attrs["speculative_generation.mode_active"] is True
        assert attrs["speculative_generation.first_rejector"] == "none"
        assert attrs["speculative_generation.first_completed"] == "input_rails"
        assert attrs["speculative_generation.output_rails_early_reject"] is False
        # both task durations recorded → time_saved present for a safe request
        assert attrs["speculative_generation.time_saved_ms"] >= 0.0

    @pytest.mark.asyncio
    async def test_streaming_input_reject_span_attrs(self, iorails_spec_stream_tracing, span_exporter):
        """Input reject: first_rejector=input_rails, generation cancelled."""
        many = [f"tok{i} " for i in range(50)]
        _wire_stream(
            iorails_spec_stream_tracing,
            input_safe=False,
            input_delay=0.02,
            stream=_token_stream(delay=0.01, tokens=many),
        )
        await asyncio.wait_for(_collect(iorails_spec_stream_tracing.stream_async(MESSAGES)), timeout=3.0)

        attrs = await self._request_attrs(span_exporter)
        assert attrs["speculative_generation.first_rejector"] == "input_rails"
        assert attrs["speculative_generation.cancellation_event"] == "generation_cancelled"

    @pytest.mark.asyncio
    async def test_streaming_output_early_reject_span_attrs(self, iorails_spec_stream_tracing, span_exporter):
        """Output early reject: first_rejector=output_rails, output_rails_early_reject=True."""
        many = [f"tok{i} " for i in range(50)]
        _wire_stream(
            iorails_spec_stream_tracing,
            input_safe=True,
            input_delay=1.0,
            output_safe=False,
            stream=_token_stream(delay=0.01, tokens=many),
        )
        await asyncio.wait_for(_collect(iorails_spec_stream_tracing.stream_async(MESSAGES)), timeout=3.0)

        attrs = await self._request_attrs(span_exporter)
        assert attrs["speculative_generation.first_rejector"] == "output_rails"
        assert attrs["speculative_generation.first_completed"] == "output_rails"
        assert attrs["speculative_generation.output_rails_early_reject"] is True
        assert attrs["speculative_generation.cancellation_event"] == "input_rails_cancelled"

    @pytest.mark.asyncio
    async def test_release_queue_attrs_recorded_when_buffer_fills(self, test_tracer, span_exporter):
        """Overflow path: the held buffer's size and hold time are both recorded."""
        cfg = copy.deepcopy(NEMOGUARDS_SPECULATIVE_STREAMING_SMALL_BUFFER_CONFIG)
        cfg["tracing"] = {"enabled": True}
        with patch.object(telemetry, "_tracer", test_tracer):
            with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
                iorails = IORails(RailsConfig.from_content(config=cfg))
            async with iorails:
                _wire_stream(iorails, input_delay=0.05, stream=_token_stream(delay=0.0))
                await asyncio.wait_for(_collect(iorails.stream_async(MESSAGES)), timeout=3.0)

        attrs = await self._request_attrs(span_exporter)
        assert attrs["speculative_generation.release_queue_token_count"] == 2  # the configured cap
        assert attrs["speculative_generation.release_queue_duration_ms"] > 0.0
        # No output rails configured, so the output-rails-specific flag is absent
        # rather than reported as a misleading False.
        assert "speculative_generation.output_rails_early_reject" not in attrs


class TestSpeculativeGenerationTiming:
    """The speculative path times the main call so its record and stats aren't left empty."""

    @pytest.mark.asyncio
    async def test_main_call_record_carries_timing(self, iorails):
        """On the speculative path, the generation call in the log has real timestamps and a duration."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.engine_registry.model_call = _hi_model()

        result = await iorails.generate_async(messages=MESSAGES, options={"log": {"llm_calls": True}})

        assert isinstance(result, GenerationResponse)
        assert result.log is not None
        gen_call = next(c for c in (result.log.llm_calls or []) if c.task == "general")
        assert gen_call.started_at is not None
        assert gen_call.finished_at is not None
        assert gen_call.duration is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reject_rails", [_slow_reject, _immediate_reject], ids=["gen-first", "rails-first-simultaneous"]
    )
    async def test_blocked_speculative_records_completed_call(self, iorails, reject_rails):
        """A speculative main call that completed before the input rails blocked is still logged —
        whether generation finished first or both finished in the same tick."""
        iorails.rails_manager.is_input_safe = reject_rails
        iorails.engine_registry.model_call = _hi_model()

        result = await iorails.generate_async(messages=MESSAGES, options={"log": {"llm_calls": True}})

        assert isinstance(result, GenerationResponse)
        assert result.log is not None
        gen_calls = [c for c in (result.log.llm_calls or []) if c.task == "general"]
        assert len(gen_calls) == 1
        assert gen_calls[0].duration is not None

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
import logging
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nemoguardrails.guardrails import telemetry
from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.types import LLMResponse
from tests.guardrails.async_helpers import started_iorails
from tests.guardrails.test_data import NEMOGUARDS_CONFIG, NEMOGUARDS_SPECULATIVE_CONFIG

MESSAGES = [{"role": "user", "content": "hi"}]


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

        async def fast_rails(messages):
            return RailResult(is_safe=True)

        async def slow_llm(model_type, messages):
            await asyncio.sleep(0.05)
            return LLMResponse(content="Hello from LLM")

        iorails.rails_manager.is_input_safe = fast_rails
        iorails.engine_registry.model_call = slow_llm
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        result = await iorails.generate_async(MESSAGES)

        assert result == {"role": "assistant", "content": "Hello from LLM"}

    @pytest.mark.asyncio
    async def test_rails_first_reject(self, iorails):
        """Rails finish first and reject — generation is cancelled, refusal returned."""
        llm_started = False
        llm_completed = False

        async def fast_reject(messages):
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

        result = await iorails.generate_async(MESSAGES)

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

        async def slow_rails(messages):
            await asyncio.sleep(0.05)
            return RailResult(is_safe=True)

        async def fast_llm(model_type, messages):
            return LLMResponse(content="Fast LLM response")

        iorails.rails_manager.is_input_safe = slow_rails
        iorails.engine_registry.model_call = fast_llm
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        result = await iorails.generate_async(MESSAGES)

        assert result == {"role": "assistant", "content": "Fast LLM response"}

    @pytest.mark.asyncio
    async def test_gen_first_reject(self, iorails):
        """Generation finishes first, then rails reject — response discarded."""

        async def slow_reject(messages):
            await asyncio.sleep(0.05)
            return RailResult(is_safe=False, reason="unsafe")

        async def fast_llm(model_type, messages):
            return LLMResponse(content="Should be discarded")

        iorails.rails_manager.is_input_safe = slow_reject
        iorails.engine_registry.model_call = fast_llm
        iorails.rails_manager.is_output_safe = AsyncMock()

        result = await iorails.generate_async(MESSAGES)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        iorails.rails_manager.is_output_safe.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_error_cancels_rails(self, iorails):
        """LLM errors while rails still running — rails cancelled, error propagated."""

        async def slow_rails(messages):
            await asyncio.sleep(0.5)
            return RailResult(is_safe=True)

        iorails.rails_manager.is_input_safe = slow_rails
        iorails.engine_registry.model_call = AsyncMock(side_effect=RuntimeError("LLM crashed"))

        with pytest.raises(RuntimeError, match="LLM crashed"):
            await iorails.generate_async(MESSAGES)

    @pytest.mark.asyncio
    async def test_rails_error_cancels_generation(self, iorails):
        """Rails error while LLM still running — generation cancelled, error propagated."""

        async def slow_llm(model_type, messages):
            await asyncio.sleep(0.5)
            return LLMResponse(content="Should not be used")

        iorails.rails_manager.is_input_safe = AsyncMock(side_effect=RuntimeError("Rails crashed"))
        iorails.engine_registry.model_call = slow_llm

        with pytest.raises(RuntimeError, match="Rails crashed"):
            await iorails.generate_async(MESSAGES)

    @pytest.mark.asyncio
    async def test_rails_reject_with_simultaneous_llm_exception(self, iorails, caplog_iorails):
        """Rails reject + LLM raises in the same scheduling window — refusal returned, exception drained."""

        async def fast_reject(messages):
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
            result = await iorails.generate_async(MESSAGES)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        iorails.rails_manager.is_output_safe.assert_not_called()
        assert any("LLM generation error suppressed" in rec.message for rec in caplog_iorails.records)

    @pytest.mark.asyncio
    async def test_both_tasks_raise_during_race(self, iorails, caplog_iorails):
        """Both rails and gen raise — outer cleanup logs the loser exception, winner propagates."""

        async def rails_raises(messages):
            raise RuntimeError("Rails crashed")

        async def gen_raises(model_type, messages):
            await asyncio.sleep(0)
            raise RuntimeError("LLM crashed too")

        iorails.rails_manager.is_input_safe = rails_raises
        iorails.engine_registry.model_call = gen_raises

        with caplog_iorails.at_level("WARNING", logger="nemoguardrails.guardrails.iorails"):
            with pytest.raises(RuntimeError):
                await iorails.generate_async(MESSAGES)

        assert any("task error discarded during cleanup" in rec.message for rec in caplog_iorails.records)

    @pytest.mark.asyncio
    async def test_rails_first_reject_records_blocked_metric(self):
        """Rails-first-reject path increments record_request_blocked when metrics are on."""
        cfg = copy.deepcopy(NEMOGUARDS_SPECULATIVE_CONFIG)
        cfg["metrics"] = {"enabled": True}

        async def fast_reject(messages):
            return RailResult(is_safe=False, reason="unsafe")

        async def slow_llm(model_type, messages):
            await asyncio.sleep(0.5)
            return LLMResponse(content="Should not be used")

        async with started_iorails(cfg) as iorails:
            iorails.rails_manager.is_input_safe = fast_reject
            iorails.engine_registry.model_call = slow_llm
            iorails.rails_manager.is_output_safe = AsyncMock()

            with patch("nemoguardrails.guardrails.iorails.record_request_blocked") as record_mock:
                result = await iorails.generate_async(MESSAGES)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        record_mock.assert_called_once()
        assert record_mock.call_args.args[0].name == "INPUT"

    @pytest.mark.asyncio
    async def test_gen_first_reject_records_blocked_metric(self):
        """Gen-first-reject path increments record_request_blocked when metrics are on."""
        cfg = copy.deepcopy(NEMOGUARDS_SPECULATIVE_CONFIG)
        cfg["metrics"] = {"enabled": True}

        async def slow_reject(messages):
            await asyncio.sleep(0.05)
            return RailResult(is_safe=False, reason="unsafe")

        async def fast_llm(model_type, messages):
            return LLMResponse(content="Should be discarded")

        async with started_iorails(cfg) as iorails:
            iorails.rails_manager.is_input_safe = slow_reject
            iorails.engine_registry.model_call = fast_llm
            iorails.rails_manager.is_output_safe = AsyncMock()

            with patch("nemoguardrails.guardrails.iorails.record_request_blocked") as record_mock:
                result = await iorails.generate_async(MESSAGES)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        record_mock.assert_called_once()
        assert record_mock.call_args.args[0].name == "INPUT"

    @pytest.mark.asyncio
    async def test_flag_disabled_runs_sequentially(self, iorails_sequential):
        """When speculative_generation is false, pipeline runs sequentially."""
        call_order = []

        async def mock_input(messages):
            call_order.append("input")
            return RailResult(is_safe=True)

        async def mock_generate(model_type, messages):
            call_order.append("generate")
            return LLMResponse(content="response")

        async def mock_output(messages, response):
            call_order.append("output")
            return RailResult(is_safe=True)

        iorails_sequential.rails_manager.is_input_safe = mock_input
        iorails_sequential.engine_registry.model_call = mock_generate
        iorails_sequential.rails_manager.is_output_safe = mock_output

        await iorails_sequential.generate_async(MESSAGES)
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

        async def fast_rails(messages):
            return RailResult(is_safe=True)

        async def slow_llm(model_type, messages):
            await asyncio.sleep(0.05)
            return LLMResponse(content="Hello from LLM")

        iorails_speculative_tracing.rails_manager.is_input_safe = fast_rails
        iorails_speculative_tracing.engine_registry.model_call = slow_llm
        iorails_speculative_tracing.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        result = await iorails_speculative_tracing.generate_async(MESSAGES)

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

        async def fast_reject(messages):
            return RailResult(is_safe=False, reason="unsafe")

        async def slow_llm(model_type, messages):
            await asyncio.sleep(0.5)
            return LLMResponse(content="Should not be used")

        iorails_speculative_tracing.rails_manager.is_input_safe = fast_reject
        iorails_speculative_tracing.engine_registry.model_call = slow_llm
        iorails_speculative_tracing.rails_manager.is_output_safe = AsyncMock()

        result = await iorails_speculative_tracing.generate_async(MESSAGES)

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

        async def slow_rails(messages):
            await asyncio.sleep(0.05)
            return RailResult(is_safe=True)

        async def fast_llm(model_type, messages):
            return LLMResponse(content="Fast LLM response")

        iorails_speculative_tracing.rails_manager.is_input_safe = slow_rails
        iorails_speculative_tracing.engine_registry.model_call = fast_llm
        iorails_speculative_tracing.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        result = await iorails_speculative_tracing.generate_async(MESSAGES)

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

        async def slow_reject(messages):
            await asyncio.sleep(0.05)
            return RailResult(is_safe=False, reason="unsafe")

        async def fast_llm(model_type, messages):
            return LLMResponse(content="Should be discarded")

        iorails_speculative_tracing.rails_manager.is_input_safe = slow_reject
        iorails_speculative_tracing.engine_registry.model_call = fast_llm
        iorails_speculative_tracing.rails_manager.is_output_safe = AsyncMock()

        result = await iorails_speculative_tracing.generate_async(MESSAGES)

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

                await iorails.generate_async(MESSAGES)

        spans = span_exporter.get_finished_spans()
        request_spans = [s for s in spans if s.name == "guardrails.request"]
        assert len(request_spans) == 1
        attrs = dict(request_spans[0].attributes)
        assert "speculative_generation.mode_active" not in attrs
        assert "speculative_generation.first_completed" not in attrs
        assert "speculative_generation.first_rejector" not in attrs

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

"""Unit tests for the OTEL metrics API in nemoguardrails.guardrails.telemetry."""

import asyncio
from typing import cast
from unittest.mock import Mock, patch

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider

from nemoguardrails.guardrails import telemetry
from nemoguardrails.guardrails.async_work_queue import AsyncWorkQueue
from nemoguardrails.guardrails.guardrails_types import RailDirection
from nemoguardrails.guardrails.telemetry import (
    _ensure_request_instruments,
    are_metrics_enabled,
    get_meter,
    record_nonstream_rejected,
    record_request_blocked,
    record_request_error,
    record_stream_rejected,
    register_nonstream_saturation_gauges,
    request_metrics,
    stream_active_metric,
    traced_request,
)
from nemoguardrails.rails.llm.config import MetricsConfig
from nemoguardrails.tracing import constants as tracing_constants
from nemoguardrails.tracing.constants import (
    SystemConstants,
    _ensure_llm_instruments,
    llm_operation_duration,
    record_time_per_output_chunk,
    record_time_to_first_chunk,
    record_token_usage,
)
from nemoguardrails.types import UsageInfo
from tests.guardrails.metric_helpers import collect_metric_points


@pytest.fixture(autouse=True)
def reset_metrics_singletons():
    """Reset module-level meter + instrument + tracer singletons between
    tests.  ``_tracer`` is included even though tests in this file are
    metric-focused — leaks of the cached tracer would otherwise affect
    later test files that exercise the OTEL adapter.
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
def meter_reader():
    """Install a test-local Meter on the telemetry module, return the reader."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    telemetry._meter = provider.get_meter(
        SystemConstants.SYSTEM_NAME,
        version="0.0.0-dev",
        schema_url="https://opentelemetry.io/schemas/1.26.0",
    )
    yield reader


@pytest.fixture
def tracer():
    """Provide a real Tracer (no exporter — tests here care about metrics, not spans)."""
    provider = TracerProvider()
    return provider.get_tracer("test")


class TestGetMeter:
    def test_returns_meter(self):
        meter = get_meter()
        assert meter is not None

    def test_returns_same_instance(self):
        m1 = get_meter()
        m2 = get_meter()
        assert m1 is m2

    def test_returns_none_without_otel(self):
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._meter = None
            assert get_meter() is None


class TestEnsureRequestInstruments:
    """``_ensure_request_instruments()`` lazily creates and caches the full
    set of OTEL instruments used by the request lifecycle."""

    def test_creates_all_instruments(self, meter_reader):
        """First call returns a populated ``RequestInstruments`` with every
        core + saturation instrument set."""
        result = _ensure_request_instruments()
        assert result is not None
        # Core request-level metrics
        assert result.requests is not None
        assert result.errors is not None
        assert result.blocked is not None
        assert result.duration is not None
        # Saturation metrics
        assert result.requests_active is not None
        assert result.nonstream_rejections is not None
        assert result.stream_active is not None
        assert result.stream_rejections is not None

    def test_returns_same_instruments_on_second_call(self, meter_reader):
        first = _ensure_request_instruments()
        second = _ensure_request_instruments()
        assert first is second

    def test_returns_none_without_otel(self):
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._meter = None
            assert _ensure_request_instruments() is None


class TestEnsureLLMInstruments:
    """``_ensure_llm_instruments()`` lazily creates and caches the
    OTEL GenAI standard LLM-call-scope instruments."""

    def test_creates_all_instruments(self, meter_reader):
        """First call returns a populated ``LLMInstruments`` with all
        four OTEL GenAI standard client-side metrics."""
        result = _ensure_llm_instruments()
        assert result is not None
        assert result.token_usage is not None
        assert result.operation_duration is not None
        assert result.time_to_first_chunk is not None
        assert result.time_per_output_chunk is not None

    def test_returns_same_instruments_on_second_call(self, meter_reader):
        """Caching: second call returns the same struct as the first.
        Important because each call to ``meter.create_histogram`` with
        the same name produces a duplicate-instrument warning, and
        we'd lose data points if the SDK actually allocated two."""
        first = _ensure_llm_instruments()
        second = _ensure_llm_instruments()
        assert first is second

    def test_returns_none_without_otel(self):
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._meter = None
            assert _ensure_llm_instruments() is None

    def test_independent_from_request_instruments(self, meter_reader):
        """``_ensure_llm_instruments`` and ``_ensure_request_instruments``
        cache separately; calling one does not populate the other."""
        _ensure_llm_instruments()
        assert tracing_constants._llm_instruments is not None
        assert telemetry._request_instruments is None
        _ensure_request_instruments()
        assert telemetry._request_instruments is not None


class TestRecordTokenUsage:
    """``record_token_usage`` emits two ``gen_ai.client.token.usage``
    Histogram observations distinguished by ``gen_ai.token.type``."""

    def test_emits_one_input_and_one_output_observation(self, meter_reader):
        usage = UsageInfo(input_tokens=42, output_tokens=17, total_tokens=59)
        record_token_usage("model-x", "openai", "chat", usage)
        points = collect_metric_points(meter_reader)
        observations = points["gen_ai.client.token.usage"]
        assert len(observations) == 2
        # Both observations carry the same model/provider/operation labels;
        # they're distinguished by gen_ai.token.type only.
        types = {obs.attributes["gen_ai.token.type"] for obs in observations}
        assert types == {"input", "output"}

    def test_label_set_includes_model_provider_operation(self, meter_reader):
        record_token_usage("model-x", "openai", "chat", UsageInfo(input_tokens=1, output_tokens=1))
        points = collect_metric_points(meter_reader)
        attrs = points["gen_ai.client.token.usage"][0].attributes
        assert attrs["gen_ai.request.model"] == "model-x"
        assert attrs["gen_ai.provider.name"] == "openai"
        assert attrs["gen_ai.operation.name"] == "chat"

    def test_no_op_when_usage_is_none(self, meter_reader):
        """Skipping when usage is None preserves the "no observation"
        vs "0 tokens" distinction — operators can tell the difference
        between a successful zero-token call and a call where the
        provider didn't return usage info."""
        record_token_usage("model-x", "openai", "chat", None)
        points = collect_metric_points(meter_reader)
        assert "gen_ai.client.token.usage" not in points

    def test_no_op_when_otel_unavailable(self):
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._meter = None
            tracing_constants._llm_instruments = None
            record_token_usage("m", "p", "chat", UsageInfo(input_tokens=1, output_tokens=1))
            # No meter to assert against; just verify no exception.

    def test_records_zero_tokens_when_provider_returns_zeros(self, meter_reader):
        """A ``UsageInfo(0, 0, 0)`` is real data (provider explicitly
        reported zero) — distinct from ``usage=None``.  We record the
        zeros."""
        record_token_usage("model-x", "openai", "chat", UsageInfo(input_tokens=0, output_tokens=0))
        points = collect_metric_points(meter_reader)
        assert len(points["gen_ai.client.token.usage"]) == 2

    def test_input_tokens_only_emits_both_observations(self, meter_reader):
        """``UsageInfo(input_tokens=10, output_tokens=0)`` — typical of
        an LLM call that errored after consuming prompt tokens but
        before generating any output.  Both observations are recorded
        (with output=0) so the operator sees the prompt cost without a
        gap in the output series."""
        record_token_usage("model-x", "openai", "chat", UsageInfo(input_tokens=10, output_tokens=0))
        sums_by_type = _histogram_sums_by_token_type(meter_reader)
        assert sums_by_type == {"input": 10, "output": 0}

    def test_output_tokens_only_emits_both_observations(self, meter_reader):
        """``UsageInfo(input_tokens=0, output_tokens=10)`` — pathological
        but possible (e.g. cached prompt that the provider counts as 0
        input).  Both observations are recorded for symmetry with the
        input-only case."""
        record_token_usage("model-x", "openai", "chat", UsageInfo(input_tokens=0, output_tokens=10))
        sums_by_type = _histogram_sums_by_token_type(meter_reader)
        assert sums_by_type == {"input": 0, "output": 10}


def _histogram_sums_by_token_type(reader):
    """Walk the SDK's collected metrics for ``gen_ai.client.token.usage``
    and return ``{token_type: sum}`` so callers can assert that input
    and output observations carried the correct values.

    ``collect_metric_points`` only exposes the histogram *count*, not
    *sum* per data point — so we read the SDK's data points directly
    here.  Kept local to this test class because no other test needs
    per-label histogram sums.
    """
    data = reader.get_metrics_data()
    out = {}
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name != "gen_ai.client.token.usage":
                    continue
                for dp in metric.data.data_points:
                    out[dp.attributes["gen_ai.token.type"]] = dp.sum
    return out


class TestLLMOperationDuration:
    """``llm_operation_duration`` context manager records the wrapped
    block's duration into ``gen_ai.client.operation.duration``,
    adding ``error.type`` only on exception."""

    def test_records_one_observation_on_success(self, meter_reader):
        with llm_operation_duration("model-x", "openai", "chat"):
            pass
        points = collect_metric_points(meter_reader)
        # Histogram count, not sum.
        assert points["gen_ai.client.operation.duration"][0].value == 1

    def test_label_set_on_success_has_no_error_type(self, meter_reader):
        with llm_operation_duration("model-x", "openai", "chat"):
            pass
        points = collect_metric_points(meter_reader)
        attrs = points["gen_ai.client.operation.duration"][0].attributes
        assert attrs["gen_ai.request.model"] == "model-x"
        assert attrs["gen_ai.provider.name"] == "openai"
        assert attrs["gen_ai.operation.name"] == "chat"
        assert "error.type" not in attrs

    def test_records_observation_on_exception_with_error_type(self, meter_reader):
        with pytest.raises(RuntimeError):
            with llm_operation_duration("model-x", "openai", "chat"):
                raise RuntimeError("boom")
        points = collect_metric_points(meter_reader)
        assert len(points["gen_ai.client.operation.duration"]) == 1
        attrs = points["gen_ai.client.operation.duration"][0].attributes
        assert attrs["error.type"] == "RuntimeError"

    def test_success_and_failure_split_by_error_type_label(self, meter_reader):
        """Two calls — one ok, one raising — produce two distinct
        data points distinguished by presence/value of ``error.type``."""
        with llm_operation_duration("model-x", "openai", "chat"):
            pass
        with pytest.raises(ValueError):
            with llm_operation_duration("model-x", "openai", "chat"):
                raise ValueError("nope")
        points = collect_metric_points(meter_reader)
        observations = points["gen_ai.client.operation.duration"]
        assert len(observations) == 2
        error_types = {obs.attributes.get("error.type") for obs in observations}
        assert error_types == {None, "ValueError"}

    def test_records_error_type_on_cancelled_error(self, meter_reader):
        """Consumer-cancelled streams raise ``asyncio.CancelledError`` —
        a ``BaseException`` subclass — inside the context manager.  The
        duration record must still carry ``error.type=CancelledError``
        so dashboards can distinguish cancelled streams from successful
        ones.
        """
        with pytest.raises(asyncio.CancelledError):
            with llm_operation_duration("model-x", "openai", "chat"):
                raise asyncio.CancelledError()
        points = collect_metric_points(meter_reader)
        assert len(points["gen_ai.client.operation.duration"]) == 1
        attrs = points["gen_ai.client.operation.duration"][0].attributes
        assert attrs["error.type"] == "CancelledError"

    def test_records_error_type_on_generator_exit(self, meter_reader):
        """``GeneratorExit`` (raised when an async generator is closed)
        is a ``BaseException`` subclass and must also tag the duration
        record with ``error.type=GeneratorExit``.
        """
        with pytest.raises(GeneratorExit):
            with llm_operation_duration("model-x", "openai", "chat"):
                raise GeneratorExit()
        points = collect_metric_points(meter_reader)
        assert len(points["gen_ai.client.operation.duration"]) == 1
        attrs = points["gen_ai.client.operation.duration"][0].attributes
        assert attrs["error.type"] == "GeneratorExit"

    def test_cancelled_error_propagates(self, meter_reader):
        """The context manager must re-raise ``CancelledError`` so the
        surrounding asyncio task is actually cancelled — swallowing it
        would break cancel semantics.
        """
        with pytest.raises(asyncio.CancelledError):
            with llm_operation_duration("model-x", "openai", "chat"):
                raise asyncio.CancelledError()

    def test_no_op_when_otel_unavailable(self):
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._meter = None
            tracing_constants._llm_instruments = None
            with llm_operation_duration("m", "p", "chat"):
                pass  # must not raise


class TestLLMOperationDurationBestEffort:
    """The ``finally`` emission must be best-effort: a meter SDK that raises
    while recording the duration must never mask the original exception, nor
    turn a successful call into a failure"""

    def test_finally_record_failure_does_not_mask_original_exception(self):
        broken = Mock()
        broken.operation_duration.record.side_effect = RuntimeError("meter SDK down")

        with patch.object(tracing_constants, "_ensure_llm_instruments", return_value=broken):
            # The original CancelledError must propagate, NOT the meter's
            # RuntimeError raised from ``finally``.
            with pytest.raises(asyncio.CancelledError):
                with llm_operation_duration("model-x", "openai", "chat"):
                    raise asyncio.CancelledError()

        broken.operation_duration.record.assert_called_once()

    def test_finally_record_failure_swallowed_on_success_path(self):
        broken = Mock()
        broken.operation_duration.record.side_effect = RuntimeError("meter SDK down")

        with patch.object(tracing_constants, "_ensure_llm_instruments", return_value=broken):
            # A broken meter must not turn a successful block into a failure.
            with llm_operation_duration("model-x", "openai", "chat"):
                pass

        broken.operation_duration.record.assert_called_once()


class TestRecordTimeToFirstChunk:
    """``record_time_to_first_chunk`` records a single observation onto
    ``gen_ai.client.operation.time_to_first_chunk`` with the standard
    label set."""

    def test_records_observation(self, meter_reader):
        record_time_to_first_chunk("model-x", "openai", "chat", 0.123)
        points = collect_metric_points(meter_reader)
        # Histogram value here is the recording count.
        assert points["gen_ai.client.operation.time_to_first_chunk"][0].value == 1

    def test_label_set(self, meter_reader):
        record_time_to_first_chunk("model-x", "openai", "chat", 0.05)
        points = collect_metric_points(meter_reader)
        attrs = points["gen_ai.client.operation.time_to_first_chunk"][0].attributes
        assert attrs["gen_ai.request.model"] == "model-x"
        assert attrs["gen_ai.provider.name"] == "openai"
        assert attrs["gen_ai.operation.name"] == "chat"

    def test_no_op_when_otel_unavailable(self):
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._meter = None
            tracing_constants._llm_instruments = None
            record_time_to_first_chunk("m", "p", "chat", 0.01)
            # No exception, no meter to assert against.


class TestRecordTimePerOutputChunk:
    """``record_time_per_output_chunk`` records the inter-chunk
    interval onto ``gen_ai.client.operation.time_per_output_chunk``.
    Caller is responsible for skipping the first chunk and any
    non-content frames; the helper itself just records."""

    def test_each_call_records_one_observation(self, meter_reader):
        for interval in (0.02, 0.04, 0.03):
            record_time_per_output_chunk("model-x", "openai", "chat", interval)
        points = collect_metric_points(meter_reader)
        # Three calls → three recordings on the same data point.
        assert points["gen_ai.client.operation.time_per_output_chunk"][0].value == 3

    def test_label_set(self, meter_reader):
        record_time_per_output_chunk("model-x", "openai", "chat", 0.05)
        points = collect_metric_points(meter_reader)
        attrs = points["gen_ai.client.operation.time_per_output_chunk"][0].attributes
        assert attrs["gen_ai.request.model"] == "model-x"
        assert attrs["gen_ai.provider.name"] == "openai"
        assert attrs["gen_ai.operation.name"] == "chat"

    def test_no_op_when_otel_unavailable(self):
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._meter = None
            tracing_constants._llm_instruments = None
            record_time_per_output_chunk("m", "p", "chat", 0.01)


class TestRequestMetrics:
    """``request_metrics()`` context manager increments the requests
    counter on entry, records duration on exit, and bumps the errors
    counter (split by ``error.type``) on exception."""

    def test_requests_counter_increments_on_entry(self, meter_reader):
        with request_metrics():
            pass
        points = collect_metric_points(meter_reader)
        assert points["guardrails.requests"][0].value == 1

    def test_counter_accumulates_across_calls(self, meter_reader):
        for _ in range(3):
            with request_metrics():
                pass
        points = collect_metric_points(meter_reader)
        assert points["guardrails.requests"][0].value == 3

    def test_duration_histogram_records_on_exit(self, meter_reader):
        with request_metrics():
            pass
        points = collect_metric_points(meter_reader)
        # Histogram value here is the count of recordings, not the sum.
        assert points["guardrails.request.duration"][0].value == 1

    def test_errors_counter_increments_on_exception(self, meter_reader):
        with pytest.raises(ValueError):
            with request_metrics():
                raise ValueError("boom")
        points = collect_metric_points(meter_reader)
        assert points["guardrails.requests.errors"][0].value == 1
        assert points["guardrails.requests.errors"][0].attributes["error.type"] == "ValueError"

    def test_errors_counter_labels_split_by_error_type(self, meter_reader):
        """Errors counter splits into separate series keyed by
        ``error.type`` attribute."""
        with pytest.raises(ValueError):
            with request_metrics():
                raise ValueError("a")
        with pytest.raises(RuntimeError):
            with request_metrics():
                raise RuntimeError("b")
        points = collect_metric_points(meter_reader)
        error_types = {point.attributes["error.type"] for point in points["guardrails.requests.errors"]}
        assert error_types == {"ValueError", "RuntimeError"}

    def test_duration_still_recorded_on_exception(self, meter_reader):
        """Duration histogram fires from ``finally``, so a raising scope
        still produces a recording."""
        with pytest.raises(ValueError):
            with request_metrics():
                raise ValueError("boom")
        points = collect_metric_points(meter_reader)
        assert points["guardrails.request.duration"][0].value == 1

    def test_errors_counter_increments_on_cancelled_error(self, meter_reader):
        """When a consumer cancels a streaming request the surrounding
        ``request_metrics`` scope exits via ``asyncio.CancelledError``
        (a ``BaseException`` subclass).  The errors counter must still
        bump with ``error.type=CancelledError`` so cancellation volume
        is visible on the dashboard.
        """
        with pytest.raises(asyncio.CancelledError):
            with request_metrics():
                raise asyncio.CancelledError()
        points = collect_metric_points(meter_reader)
        assert len(points["guardrails.requests.errors"]) == 1
        assert points["guardrails.requests.errors"][0].value == 1
        assert points["guardrails.requests.errors"][0].attributes["error.type"] == "CancelledError"

    def test_errors_counter_increments_on_generator_exit(self, meter_reader):
        """A closed async generator surfaces as ``GeneratorExit`` inside
        the ``request_metrics`` scope.  The errors counter must bump
        with ``error.type=GeneratorExit``.
        """
        with pytest.raises(GeneratorExit):
            with request_metrics():
                raise GeneratorExit()
        points = collect_metric_points(meter_reader)
        assert len(points["guardrails.requests.errors"]) == 1
        assert points["guardrails.requests.errors"][0].attributes["error.type"] == "GeneratorExit"

    def test_cancelled_error_propagates(self, meter_reader):
        """``request_metrics`` must re-raise ``CancelledError`` — swallowing
        it would prevent the surrounding asyncio task from being
        cancelled.
        """
        with pytest.raises(asyncio.CancelledError):
            with request_metrics():
                raise asyncio.CancelledError()

    def test_requests_active_nets_to_zero_after_completed_scope(self, meter_reader):
        """+1 on entry, -1 on exit → net 0 for a completed scope."""
        with request_metrics():
            pass
        points = collect_metric_points(meter_reader)
        assert points["guardrails.requests.active"][0].value == 0

    def test_requests_active_nets_to_zero_after_exception(self, meter_reader):
        """Exception path still decrements — -1 lives in ``finally``."""
        with pytest.raises(ValueError):
            with request_metrics():
                raise ValueError("boom")
        points = collect_metric_points(meter_reader)
        assert points["guardrails.requests.active"][0].value == 0

    def test_requests_active_reflects_concurrent_scopes_mid_flight(self, meter_reader):
        """Two overlapping ``request_metrics()`` scopes → counter reads 2
        mid-flight.  Simulates mid-flight observation without actually
        running two real requests."""
        with request_metrics():
            with request_metrics():
                mid = collect_metric_points(meter_reader)
                assert mid["guardrails.requests.active"][0].value == 2
        final = collect_metric_points(meter_reader)
        assert final["guardrails.requests.active"][0].value == 0

    def test_no_metrics_when_otel_unavailable(self):
        """``request_metrics()`` is a silent no-op when OTEL isn't installed —
        the context manager must not raise."""
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._meter = None
            with request_metrics():
                pass
            # Just verify no crash; there's no reader to check against.


class TestRequestMetricsBestEffort:
    """The ``finally`` emissions must be best-effort: a meter SDK that raises
    while decrementing the active gauge or recording duration must never mask
    the original exception, nor turn a successful request into a failure.
    Each emit is guarded independently so a failure in one still attempts the
    other — the active-gauge decrement must run to avoid leaking the gauge
    """

    @staticmethod
    def _broken_instruments():
        """A RequestInstruments-shaped mock whose ``finally`` emits both fail.

        ``requests_active.add`` succeeds on the ``+1`` entry bump (so the
        scope is entered normally) but fails on the ``-1`` finally decrement,
        and ``duration.record`` always fails.
        """
        broken = Mock()

        def _add(delta, *args, **kwargs):
            if delta < 0:
                raise RuntimeError("meter SDK down")

        broken.requests_active.add.side_effect = _add
        broken.duration.record.side_effect = RuntimeError("meter SDK down")
        return broken

    def test_finally_emit_failure_does_not_mask_original_exception(self):
        broken = self._broken_instruments()

        with patch.object(telemetry, "_ensure_request_instruments", return_value=broken):
            # The original CancelledError must propagate, NOT a RuntimeError
            # from either failing ``finally`` emit.
            with pytest.raises(asyncio.CancelledError):
                with request_metrics():
                    raise asyncio.CancelledError()

        # Both finally emits were attempted despite the first one failing.
        broken.requests_active.add.assert_any_call(-1)
        broken.duration.record.assert_called_once()

    def test_finally_emit_failure_swallowed_on_success_path(self):
        broken = self._broken_instruments()

        with patch.object(telemetry, "_ensure_request_instruments", return_value=broken):
            # A broken meter must not turn a successful request into a failure.
            with request_metrics():
                pass

        broken.requests_active.add.assert_any_call(-1)
        broken.duration.record.assert_called_once()


class TestTracedRequestMetrics:
    """``traced_request(tracer, metrics_enabled)`` gates the two signals
    independently.  All four combinations exercised here.
    """

    def test_both_enabled_emits_metrics(self, meter_reader, tracer):
        with traced_request(tracer, metrics_enabled=True):
            pass
        points = collect_metric_points(meter_reader)
        assert points["guardrails.requests"][0].value == 1
        assert points["guardrails.request.duration"][0].value == 1

    def test_metrics_only_emits_metrics(self, meter_reader):
        """tracer=None, metrics_enabled=True — the cost-optimized setup."""
        with traced_request(None, metrics_enabled=True):
            pass
        points = collect_metric_points(meter_reader)
        assert points["guardrails.requests"][0].value == 1
        assert points["guardrails.request.duration"][0].value == 1

    def test_tracing_only_emits_no_metrics(self, meter_reader, tracer):
        """tracer!=None, metrics_enabled=False — span emits (not asserted
        here; see span tests) but no metric data points are recorded.
        """
        with traced_request(tracer, metrics_enabled=False):
            pass
        points = collect_metric_points(meter_reader)
        assert points == {}

    def test_both_disabled_emits_nothing(self, meter_reader):
        with traced_request(None, metrics_enabled=False):
            pass
        points = collect_metric_points(meter_reader)
        assert points == {}

    def test_errors_counter_on_exception_metrics_only(self, meter_reader):
        """Exception through a metrics-only traced_request still bumps the
        errors counter — the errors counter follows metrics_enabled, not
        tracer presence.
        """
        with pytest.raises(ValueError):
            with traced_request(None, metrics_enabled=True):
                raise ValueError("boom")
        points = collect_metric_points(meter_reader)
        assert points["guardrails.requests.errors"][0].value == 1
        assert points["guardrails.requests.errors"][0].attributes["error.type"] == "ValueError"

    def test_errors_counter_on_exception_both_enabled(self, meter_reader, tracer):
        with pytest.raises(ValueError):
            with traced_request(tracer, metrics_enabled=True):
                raise ValueError("boom")
        points = collect_metric_points(meter_reader)
        assert points["guardrails.requests.errors"][0].value == 1
        assert points["guardrails.requests.errors"][0].attributes["error.type"] == "ValueError"

    def test_no_errors_counter_when_metrics_disabled(self, meter_reader, tracer):
        """Exception through tracing-only traced_request does NOT bump the
        errors counter — metrics are off.
        """
        with pytest.raises(ValueError):
            with traced_request(tracer, metrics_enabled=False):
                raise ValueError("boom")
        points = collect_metric_points(meter_reader)
        assert points == {}


class TestNoMeterProviderConfigured:
    """OTEL API is available but the host has not configured a MeterProvider.

    The OTEL API returns proxy/no-op instruments in this case; emissions should
    be silent passthroughs with no exceptions raised.
    """

    def test_request_metrics_does_not_raise(self):
        # No meter_reader fixture — get_meter() will produce the API default
        # (proxy/no-op) meter, and instrument .add()/.record() calls are no-ops.
        with request_metrics():
            pass

    def test_request_metrics_does_not_raise_on_exception(self):
        with pytest.raises(ValueError):
            with request_metrics():
                raise ValueError("boom")

    def test_ensure_request_instruments_returns_populated_struct(self):
        # Even without a MeterProvider, the API returns a meter, so instrument
        # creation still succeeds and returns a populated RequestInstruments.
        result = _ensure_request_instruments()
        assert result is not None
        assert result.requests is not None
        assert result.errors is not None
        assert result.duration is not None


class TestRecordRequestError:
    """Direct coverage for ``record_request_error``.

    Exercised indirectly by the streaming-failure tests in the integration
    suite and by ``request_metrics``'s except branch, but the
    OTEL-unavailable short-circuit is unreachable from either — that path
    only fires when callers invoke the helper directly with no OTEL.
    """

    def test_no_op_when_otel_unavailable(self):
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._meter = None
            telemetry._request_instruments = None
            # Must not raise; must not crash on attribute access.
            record_request_error(ValueError("boom"))

    def test_swallows_sdk_failure(self):
        """A meter SDK that raises while bumping the errors counter must not
        propagate — ``record_request_error`` is best-effort so it can be
        called from an ``except`` branch handling a cancellation without
        masking it.
        """
        broken = Mock()
        broken.errors.add.side_effect = RuntimeError("meter SDK down")

        with patch.object(telemetry, "_ensure_request_instruments", return_value=broken):
            # Must return cleanly, not raise.
            record_request_error(ValueError("orig"))

        broken.errors.add.assert_called_once()

    def test_does_not_swallow_base_exception_from_sdk(self):
        """Only ``Exception`` is suppressed.  A ``BaseException`` raised by the
        meter SDK must still propagate rather than be silently dropped.
        """
        broken = Mock()
        broken.errors.add.side_effect = KeyboardInterrupt()

        with patch.object(telemetry, "_ensure_request_instruments", return_value=broken):
            with pytest.raises(KeyboardInterrupt):
                record_request_error(ValueError("orig"))


class TestRecordRequestBlocked:
    def test_input_block_labels_rail_type_input(self, meter_reader):
        record_request_blocked(RailDirection.INPUT)
        points = collect_metric_points(meter_reader)
        assert points["guardrails.requests.blocked"][0].value == 1
        assert points["guardrails.requests.blocked"][0].attributes["rail.type"] == "Input"

    def test_output_block_labels_rail_type_output(self, meter_reader):
        record_request_blocked(RailDirection.OUTPUT)
        points = collect_metric_points(meter_reader)
        assert points["guardrails.requests.blocked"][0].value == 1
        assert points["guardrails.requests.blocked"][0].attributes["rail.type"] == "Output"

    def test_labels_split_points_by_direction(self, meter_reader):
        record_request_blocked(RailDirection.INPUT)
        record_request_blocked(RailDirection.OUTPUT)
        record_request_blocked(RailDirection.INPUT)
        points = collect_metric_points(meter_reader)
        by_type = {p.attributes["rail.type"]: p.value for p in points["guardrails.requests.blocked"]}
        assert by_type == {"Input": 2, "Output": 1}

    def test_no_op_when_otel_unavailable(self):
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._meter = None
            telemetry._request_instruments = None
            record_request_blocked(RailDirection.INPUT)  # must not raise


class TestAreMetricsEnabled:
    """``are_metrics_enabled`` gates purely on ``config.metrics.enabled`` and
    OTEL availability — it does NOT consult any tracing state.
    """

    def test_returns_true_when_config_enabled(self):
        """``MetricsConfig(enabled=True)`` with OTEL available → metrics on."""
        assert are_metrics_enabled(MetricsConfig(enabled=True)) is True

    def test_returns_false_when_config_disabled(self):
        """``MetricsConfig(enabled=False)`` → metrics off."""
        assert are_metrics_enabled(MetricsConfig(enabled=False)) is False

    def test_returns_false_when_config_none(self):
        """Missing ``MetricsConfig`` → metrics off."""
        assert are_metrics_enabled(None) is False

    def test_returns_false_when_otel_unavailable(self):
        """Even with ``enabled=True``, metrics stay off and a UserWarning
        fires when the OTEL API package isn't installed."""
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            with pytest.warns(UserWarning, match="opentelemetry-api package is not installed"):
                assert are_metrics_enabled(MetricsConfig(enabled=True)) is False


class TestRecordStreamRejected:
    """``record_stream_rejected()`` bumps the stream-rejections counter."""

    def test_increments_counter(self, meter_reader):
        """Each call increments ``guardrails.stream.rejections`` by 1."""
        record_stream_rejected()
        record_stream_rejected()
        points = collect_metric_points(meter_reader)
        assert points["guardrails.stream.rejections"][0].value == 2

    def test_no_op_when_otel_unavailable(self):
        """Silent no-op when OTEL isn't installed — must not raise."""
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._meter = None
            telemetry._request_instruments = None
            record_stream_rejected()  # must not raise


class TestRecordNonstreamRejected:
    """``record_nonstream_rejected()`` bumps the nonstream-rejections counter."""

    def test_increments_counter(self, meter_reader):
        """Each call increments ``guardrails.nonstream.rejections`` by 1."""
        record_nonstream_rejected()
        record_nonstream_rejected()
        record_nonstream_rejected()
        points = collect_metric_points(meter_reader)
        assert points["guardrails.nonstream.rejections"][0].value == 3

    def test_no_op_when_otel_unavailable(self):
        """Silent no-op when OTEL isn't installed — must not raise."""
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._meter = None
            telemetry._request_instruments = None
            record_nonstream_rejected()  # must not raise


class TestStreamActiveMetric:
    """``stream_active_metric()`` context manager nets to zero across
    completed and failed scopes and reflects concurrent streams."""

    def test_nets_to_zero_after_completed_scope(self, meter_reader):
        """UpDownCounter +1 on enter, -1 on exit → net 0 for a completed scope."""
        with stream_active_metric():
            pass
        points = collect_metric_points(meter_reader)
        assert points["guardrails.stream.active"][0].value == 0

    def test_nets_to_zero_after_exception(self, meter_reader):
        """Exception path still decrements — -1 lives in ``finally``."""
        with pytest.raises(ValueError):
            with stream_active_metric():
                raise ValueError("boom")
        points = collect_metric_points(meter_reader)
        assert points["guardrails.stream.active"][0].value == 0

    def test_reflects_concurrent_streams_mid_flight(self, meter_reader):
        """Observe mid-flight: two overlapping streams → counter reads 2."""
        with stream_active_metric():
            with stream_active_metric():
                mid = collect_metric_points(meter_reader)
                assert mid["guardrails.stream.active"][0].value == 2
        final = collect_metric_points(meter_reader)
        assert final["guardrails.stream.active"][0].value == 0

    def test_no_op_when_otel_unavailable(self):
        """Silent no-op when OTEL isn't installed — must not raise."""
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._meter = None
            telemetry._request_instruments = None
            with stream_active_metric():
                pass  # must not raise


class _FakeQueue:
    """Stand-in for AsyncWorkQueue used by gauge-registration unit tests.

    The real ``AsyncWorkQueue`` needs an asyncio event loop to start; the
    gauge callbacks only need ``num_pending()`` and ``num_busy_workers()``,
    so a tiny stub keeps these tests synchronous.
    """

    def __init__(self, queued: int = 0, active: int = 0) -> None:
        """Initialize with fixed counts the gauge callbacks will read back."""
        self._queued = queued
        self._active = active

    def num_pending(self) -> int:
        """Return the configured queued-item count."""
        return self._queued

    def num_busy_workers(self) -> int:
        """Return the configured busy-worker count."""
        return self._active


class TestRegisterNonstreamSaturationGauges:
    """``register_nonstream_saturation_gauges()`` wires the
    ``nonstream.queued`` / ``nonstream.active`` ObservableGauges to a
    queue and an ``is_running`` flag."""

    def test_registers_both_gauges(self, meter_reader):
        """After registration, both gauge series exist in the collected
        metric points."""
        fake = _FakeQueue(queued=0, active=0)
        register_nonstream_saturation_gauges(cast(AsyncWorkQueue, fake), is_running=lambda: True)
        points = collect_metric_points(meter_reader)
        assert "guardrails.nonstream.queued" in points
        assert "guardrails.nonstream.active" in points

    def test_gauges_reflect_live_queue_state(self, meter_reader):
        """Callbacks re-read the queue on each collection — bumping the
        fake's counters between collections shows up immediately."""
        fake = _FakeQueue(queued=3, active=7)
        register_nonstream_saturation_gauges(cast(AsyncWorkQueue, fake), is_running=lambda: True)

        first = collect_metric_points(meter_reader)
        assert first["guardrails.nonstream.queued"][0].value == 3
        assert first["guardrails.nonstream.active"][0].value == 7

        fake._queued = 11
        fake._active = 2
        second = collect_metric_points(meter_reader)
        assert second["guardrails.nonstream.queued"][0].value == 11
        assert second["guardrails.nonstream.active"][0].value == 2

    def test_disabled_flag_suppresses_observations(self, meter_reader):
        """When ``is_running()`` returns False the callbacks return [] and
        no data points are exported (soft-disable behaviour used after
        ``IORails.stop()``)."""
        fake = _FakeQueue(queued=5, active=5)
        enabled = True
        register_nonstream_saturation_gauges(cast(AsyncWorkQueue, fake), is_running=lambda: enabled)

        enabled_points = collect_metric_points(meter_reader)
        assert enabled_points["guardrails.nonstream.queued"][0].value == 5
        assert enabled_points["guardrails.nonstream.active"][0].value == 5

        enabled = False
        disabled_points = collect_metric_points(meter_reader)
        # Callback returned [] — the SDK may either omit the metric entirely
        # or include it with zero data points.  Assert "no observations",
        # which is what consumers actually care about.
        assert disabled_points.get("guardrails.nonstream.queued", []) == []
        assert disabled_points.get("guardrails.nonstream.active", []) == []

    def test_flag_flip_back_on_resumes_observations(self, meter_reader):
        """After soft-disable, flipping the flag back to True resumes
        collection on the same callbacks — matches the stop → start cycle."""
        fake = _FakeQueue(queued=1, active=0)
        enabled = False
        register_nonstream_saturation_gauges(cast(AsyncWorkQueue, fake), is_running=lambda: enabled)

        off = collect_metric_points(meter_reader)
        # SDK omits gauges entirely when callbacks return [] (no data points
        # → no metric in the export).  Assert "no observations" defensively
        # so the test doesn't depend on which of the two shapes the SDK picks.
        assert off.get("guardrails.nonstream.queued", []) == []

        enabled = True
        on = collect_metric_points(meter_reader)
        assert on["guardrails.nonstream.queued"][0].value == 1

    def test_no_op_when_meter_is_none(self):
        """No MeterProvider configured → no registration, no exception."""
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._meter = None
            fake = _FakeQueue()
            register_nonstream_saturation_gauges(cast(AsyncWorkQueue, fake), is_running=lambda: True)  # must not raise

    def test_accepts_real_async_work_queue(self):
        """Smoke test: the helper accepts an actual ``AsyncWorkQueue`` — no
        asyncio loop needed for registration, only for start/stop."""
        real_queue = AsyncWorkQueue(name="t", max_queue_size=4, max_concurrency=2)
        # No meter_reader fixture — we only care that registration doesn't
        # raise on the real class.  Skipped when no meter is configured.
        register_nonstream_saturation_gauges(real_queue, is_running=lambda: False)

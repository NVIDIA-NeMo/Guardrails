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

from typing import cast
from unittest.mock import patch

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
from nemoguardrails.tracing.constants import SystemConstants
from tests.guardrails.metric_helpers import collect_metric_points


@pytest.fixture(autouse=True)
def reset_metrics_singletons():
    """Reset module-level meter + instrument singletons between tests."""
    telemetry._meter = None
    telemetry._request_instruments = None
    yield
    telemetry._meter = None
    telemetry._request_instruments = None


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

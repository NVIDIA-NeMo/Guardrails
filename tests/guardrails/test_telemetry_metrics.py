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

from unittest.mock import patch

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider

from nemoguardrails.guardrails import telemetry
from nemoguardrails.guardrails.guardrails_types import RailDirection
from nemoguardrails.guardrails.telemetry import (
    _ensure_request_instruments,
    are_metrics_enabled,
    get_meter,
    record_request_blocked,
    record_request_error,
    request_metrics,
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
    def test_creates_three_instruments(self, meter_reader):
        result = _ensure_request_instruments()
        assert result is not None
        assert result.requests is not None
        assert result.errors is not None
        assert result.duration is not None

    def test_returns_same_instruments_on_second_call(self, meter_reader):
        first = _ensure_request_instruments()
        second = _ensure_request_instruments()
        assert first is second

    def test_returns_none_without_otel(self):
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._meter = None
            assert _ensure_request_instruments() is None


class TestRequestMetrics:
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
        with pytest.raises(ValueError):
            with request_metrics():
                raise ValueError("boom")
        points = collect_metric_points(meter_reader)
        assert points["guardrails.request.duration"][0].value == 1

    def test_no_metrics_when_otel_unavailable(self):
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
        assert are_metrics_enabled(MetricsConfig(enabled=True)) is True

    def test_returns_false_when_config_disabled(self):
        assert are_metrics_enabled(MetricsConfig(enabled=False)) is False

    def test_returns_false_when_config_none(self):
        assert are_metrics_enabled(None) is False

    def test_returns_false_when_otel_unavailable(self):
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            with pytest.warns(UserWarning, match="opentelemetry-api package is not installed"):
                assert are_metrics_enabled(MetricsConfig(enabled=True)) is False

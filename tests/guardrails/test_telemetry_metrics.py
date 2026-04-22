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

from typing import Dict, List, NamedTuple
from unittest.mock import patch

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider

from nemoguardrails.guardrails import telemetry
from nemoguardrails.guardrails.telemetry import (
    _ensure_request_instruments,
    get_meter,
    request_metrics,
    traced_request,
)
from nemoguardrails.tracing.constants import SystemConstants


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


class MetricPoint(NamedTuple):
    """One OTEL metric data point flattened from the SDK's nested shape.

    ``value`` is the counter's cumulative sum OR the histogram's recording
    count, depending on instrument type.  ``attributes`` holds the label
    key-values for this data point's label-set.
    """

    value: float
    attributes: Dict[str, str]


def _point_value(data_point) -> float:
    """Return the counter's cumulative sum OR the histogram's recording count.

    OTEL SDK data points carry ``value`` on counters (monotonic sum) and
    ``count`` on histograms (number of ``record()`` calls).  The SDK doesn't
    expose a unified accessor, so we sniff.
    """
    value = getattr(data_point, "value", None)
    if value is not None:
        return value
    return getattr(data_point, "count", 0)


def _collect_points(reader: InMemoryMetricReader) -> Dict[str, List[MetricPoint]]:
    """Flatten SDK-collected metric data into ``{metric_name: [MetricPoint, ...]}``.

    The SDK groups points under ``Resource → Scope → Metric``; this walk
    flattens that and keys on metric name, which is what tests care about.
    One data point per unique label-set.
    """
    out: Dict[str, List[MetricPoint]] = {}
    data = reader.get_metrics_data()
    if data is None:
        return out
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                out[metric.name] = [
                    MetricPoint(
                        value=_point_value(data_point),
                        attributes=dict(data_point.attributes or {}),
                    )
                    for data_point in metric.data.data_points
                ]
    return out


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
        points = _collect_points(meter_reader)
        assert points["guardrails.requests"][0].value == 1

    def test_counter_accumulates_across_calls(self, meter_reader):
        for _ in range(3):
            with request_metrics():
                pass
        points = _collect_points(meter_reader)
        assert points["guardrails.requests"][0].value == 3

    def test_duration_histogram_records_on_exit(self, meter_reader):
        with request_metrics():
            pass
        points = _collect_points(meter_reader)
        # Histogram value here is the count of recordings, not the sum.
        assert points["guardrails.request.duration"][0].value == 1

    def test_errors_counter_increments_on_exception(self, meter_reader):
        with pytest.raises(ValueError):
            with request_metrics():
                raise ValueError("boom")
        points = _collect_points(meter_reader)
        assert points["guardrails.requests.errors"][0].value == 1
        assert points["guardrails.requests.errors"][0].attributes["error.type"] == "ValueError"

    def test_errors_counter_labels_split_by_error_type(self, meter_reader):
        with pytest.raises(ValueError):
            with request_metrics():
                raise ValueError("a")
        with pytest.raises(RuntimeError):
            with request_metrics():
                raise RuntimeError("b")
        points = _collect_points(meter_reader)
        error_types = {point.attributes["error.type"] for point in points["guardrails.requests.errors"]}
        assert error_types == {"ValueError", "RuntimeError"}

    def test_duration_still_recorded_on_exception(self, meter_reader):
        with pytest.raises(ValueError):
            with request_metrics():
                raise ValueError("boom")
        points = _collect_points(meter_reader)
        assert points["guardrails.request.duration"][0].value == 1

    def test_no_metrics_when_otel_unavailable(self):
        with patch.object(telemetry, "_OTEL_AVAILABLE", False):
            telemetry._meter = None
            with request_metrics():
                pass
            # Just verify no crash; there's no reader to check against.


class TestTracedRequestMetrics:
    def test_traced_request_emits_metrics_when_tracer_present(self, meter_reader, tracer):
        with traced_request(tracer):
            pass
        points = _collect_points(meter_reader)
        assert points["guardrails.requests"][0].value == 1
        assert points["guardrails.request.duration"][0].value == 1

    def test_traced_request_emits_no_metrics_when_tracer_none(self, meter_reader):
        with traced_request(None):
            pass
        points = _collect_points(meter_reader)
        assert points == {}

    def test_traced_request_errors_counter_on_exception(self, meter_reader, tracer):
        with pytest.raises(ValueError):
            with traced_request(tracer):
                raise ValueError("boom")
        points = _collect_points(meter_reader)
        assert points["guardrails.requests.errors"][0].value == 1
        assert points["guardrails.requests.errors"][0].attributes["error.type"] == "ValueError"


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

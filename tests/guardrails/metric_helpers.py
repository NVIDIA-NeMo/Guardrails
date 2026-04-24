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

"""Shared test helpers for inspecting OTEL metrics data.

Both ``test_telemetry_metrics.py`` (unit) and ``test_iorails_telemetry.py``
(integration) need to walk the SDK's ``Resource → Scope → Metric`` nesting
and flatten it into something assertable.  These helpers live here so the
flattening logic has a single source of truth.
"""

from typing import Dict, List, NamedTuple

from opentelemetry.sdk.metrics.export import InMemoryMetricReader


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


def collect_histogram_sum(reader: InMemoryMetricReader, metric_name: str) -> float:
    """Return the aggregated ``sum`` across all data points of a histogram.

    ``collect_metric_points`` flattens histograms to their *count* (number
    of recordings) — fine for "did it fire?" tests.  When a test also
    cares about the magnitude (e.g. asserting duration includes queue-
    wait), it needs ``sum``.  Returns ``0.0`` if the metric hasn't been
    recorded yet.
    """
    data = reader.get_metrics_data()
    if data is None:
        return 0.0
    total = 0.0
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name != metric_name:
                    continue
                for data_point in metric.data.data_points:
                    total += getattr(data_point, "sum", 0.0) or 0.0
    return total


def collect_metric_points(reader: InMemoryMetricReader) -> Dict[str, List[MetricPoint]]:
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

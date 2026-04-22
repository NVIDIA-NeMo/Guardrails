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

"""Constants for the inline IORails OTEL instrumentation.

Separate from :mod:`nemoguardrails.tracing.constants` (which serves the legacy
LLMRails post-hoc tracing path) so the inline IORails signal surface can
evolve independently.

Tests deliberately assert on the raw metric-name strings rather than these
constants — the names are part of the library's public API (customers point
dashboards and alerts at them), so test assertions should verify the wire
contract, not re-reference the same symbol the production code uses.
"""


class MetricNames:
    """OTEL metric names emitted by the IORails engine."""

    REQUESTS = "guardrails.requests"
    REQUESTS_ERRORS = "guardrails.requests.errors"
    REQUESTS_BLOCKED = "guardrails.requests.blocked"
    REQUEST_DURATION = "guardrails.request.duration"

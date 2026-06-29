# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Session-aware egress monitor — a PROTOTYPE / SPIKE, not production code.

Where `policy.py` authorizes one `(tool, args, principal)` triple in isolation,
this watches the *sequence* of outbound (egress) tool calls within a session and
vetoes aggregate behavior that no single-call rule can see: too many requests,
too much cumulative outbound volume, too many distinct destination hosts, or a
burst rate. It is the layer the context-exfiltration finding was triaged to (see
`synthesis/TRIAGE.md`).

It is a coarse heuristic backstop — a rate/volume proxy, NOT exfiltration
detection: a patient attacker stays under any static threshold, and a legitimate
data-sync agent can trip it. Like `policy.py`, it has no Guardrails dependency,
so it can be exercised offline and wired into either a Guardrails action/output
rail or an orchestrator (see `authorize_with_egress`). State is in-memory and
single-process; a real deployment would back it with a shared store.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urlparse

from policy import Principal, ToolCall, ToolCallGuard


@dataclass(frozen=True)
class EgressLimits:
    """Per-principal, per-session ceilings on aggregate egress behavior."""

    max_requests: int = 20  # total egress calls in the session
    max_cumulative_bytes: int = 100_000  # total outbound payload
    max_distinct_hosts: int = 5  # fan-out across destinations
    max_requests_per_window: int = 5  # burst rate
    window_seconds: float = 10.0
    egress_tools: frozenset = frozenset({"http_request"})  # which tools count as egress
    exempt_hosts: frozenset = frozenset()  # internal hosts that don't count toward cardinality


@dataclass(frozen=True)
class EgressDecision:
    allowed: bool
    reason: str


@dataclass
class _SessionState:
    count: int = 0
    cumulative_bytes: int = 0
    hosts: set = field(default_factory=set)
    times: list = field(default_factory=list)  # monotonic timestamps of allowed egress calls


def _host_of(call: ToolCall) -> Optional[str]:
    url = call.args.get("url")
    return urlparse(url).hostname if isinstance(url, str) else None


def _estimate_bytes(call: ToolCall) -> int:
    # Proxy for outbound payload: the size of the string arguments carried.
    return sum(len(v) for v in call.args.values() if isinstance(v, str))


class EgressMonitor:
    """Tracks egress per (session, principal) and vetoes when a ceiling is crossed.

    A call that would cross a limit is blocked and NOT recorded — it does not
    happen, so it must not count toward the tally. An allowed call commits its
    increments. The clock is injectable so the rate window is deterministic in
    tests.
    """

    def __init__(self, limits: EgressLimits = EgressLimits(), clock: Callable[[], float] = time.monotonic):
        self._limits = limits
        self._clock = clock
        self._sessions: dict[tuple[str, str], _SessionState] = {}

    def check(self, session_id: str, call: ToolCall, principal: Principal) -> EgressDecision:
        limits = self._limits
        if call.tool not in limits.egress_tools:
            return EgressDecision(True, f"{call.tool} is not an egress-tracked tool")

        state = self._sessions.setdefault((session_id, principal.id), _SessionState())
        now = self._clock()
        host = _host_of(call)
        size = _estimate_bytes(call)

        # Evaluate against prospective totals; commit only if the call is allowed.
        prospective_count = state.count + 1
        prospective_bytes = state.cumulative_bytes + size
        prospective_hosts = set(state.hosts)
        if host and host not in limits.exempt_hosts:
            prospective_hosts.add(host)
        recent = [t for t in state.times if t >= now - limits.window_seconds]
        prospective_window = len(recent) + 1

        if prospective_count > limits.max_requests:
            return EgressDecision(
                False, f"egress count {prospective_count} exceeds session limit {limits.max_requests}"
            )
        if prospective_bytes > limits.max_cumulative_bytes:
            return EgressDecision(
                False, f"cumulative egress {prospective_bytes}B exceeds session limit {limits.max_cumulative_bytes}B"
            )
        if len(prospective_hosts) > limits.max_distinct_hosts:
            return EgressDecision(
                False,
                f"distinct egress hosts {len(prospective_hosts)} exceeds session limit {limits.max_distinct_hosts}",
            )
        if prospective_window > limits.max_requests_per_window:
            return EgressDecision(
                False,
                f"egress rate {prospective_window} in {limits.window_seconds:g}s exceeds limit "
                f"{limits.max_requests_per_window}",
            )

        state.count = prospective_count
        state.cumulative_bytes = prospective_bytes
        state.hosts = prospective_hosts
        recent.append(now)
        state.times = recent
        return EgressDecision(True, "within egress limits")

    def reset(self, session_id: str, principal_id: str) -> None:
        """Drop a session's tally (e.g. when a conversation ends)."""
        self._sessions.pop((session_id, principal_id), None)


def authorize_with_egress(
    guard: ToolCallGuard,
    monitor: EgressMonitor,
    session_id: str,
    call: ToolCall,
    principal: Principal,
) -> EgressDecision:
    """Compose the two layers an orchestrator (or Guardrails action) applies in
    place of a raw tool: the stateless per-call guard authorizes the single call
    first, and only if it passes does the session-aware egress monitor get to
    veto on aggregate behavior."""
    decision = guard.authorize(call, principal)
    if not decision.allowed:
        return EgressDecision(False, decision.reason)
    return monitor.check(session_id, call, principal)

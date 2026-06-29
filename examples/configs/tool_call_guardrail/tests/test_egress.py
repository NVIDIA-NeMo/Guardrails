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

"""Offline tests for the session-aware egress monitor (PROTOTYPE / SPIKE).

Pins the aggregate signals (count, volume, host cardinality, burst rate), the
"blocked calls are not recorded" invariant, per-session/per-principal isolation,
and the two-layer composition in `authorize_with_egress`. Stdlib only — no LLM,
no network, no Guardrails install.
"""

from egress import EgressDecision, EgressLimits, EgressMonitor, authorize_with_egress
from example_policies import HARDENED_GUARD
from policy import Principal, ToolCall

ALICE = Principal(
    "dev-alice", roles=frozenset({"developer"}), attributes={"owned_repos": frozenset(), "elevated": True}
)
BOB = Principal("dev-bob", roles=frozenset({"developer"}), attributes={"elevated": False})


def _req(host="api.example.com", **extra):
    return ToolCall("http_request", {"url": f"https://{host}/", **extra})


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_non_egress_tool_is_not_tracked():
    mon = EgressMonitor(EgressLimits(max_requests=1))
    # write_file is not in egress_tools -> always passes the monitor, never counted.
    for _ in range(5):
        d = mon.check("s", ToolCall("write_file", {"path": "/workspace/alice/x"}), ALICE)
        assert d.allowed


def test_request_count_limit():
    mon = EgressMonitor(EgressLimits(max_requests=2, max_distinct_hosts=99, max_requests_per_window=99))
    assert mon.check("s", _req("a.com"), ALICE).allowed
    assert mon.check("s", _req("a.com"), ALICE).allowed
    blocked = mon.check("s", _req("a.com"), ALICE)
    assert not blocked.allowed and "count" in blocked.reason


def test_cumulative_volume_limit():
    mon = EgressMonitor(
        EgressLimits(max_cumulative_bytes=5000, max_requests=99, max_distinct_hosts=99, max_requests_per_window=99)
    )
    assert mon.check("s", _req("a.com", body="x" * 2000), ALICE).allowed
    assert mon.check("s", _req("a.com", body="x" * 2000), ALICE).allowed
    blocked = mon.check("s", _req("a.com", body="x" * 2000), ALICE)
    assert not blocked.allowed and "cumulative" in blocked.reason


def test_distinct_host_cardinality_limit():
    mon = EgressMonitor(EgressLimits(max_distinct_hosts=3, max_requests=99, max_requests_per_window=99))
    for host in ("a.com", "b.com", "c.com"):
        assert mon.check("s", _req(host), ALICE).allowed
    blocked = mon.check("s", _req("d.com"), ALICE)
    assert not blocked.allowed and "hosts" in blocked.reason
    # A previously-seen host does not add cardinality, so it is still allowed.
    assert mon.check("s", _req("a.com"), ALICE).allowed


def test_blocked_call_is_not_recorded():
    # One over the byte limit is blocked; a later small call that fits is still allowed,
    # proving the blocked call did not consume budget.
    mon = EgressMonitor(
        EgressLimits(max_cumulative_bytes=1000, max_requests=99, max_distinct_hosts=99, max_requests_per_window=99)
    )
    assert not mon.check("s", _req("a.com", body="x" * 2000), ALICE).allowed
    assert mon.check("s", _req("a.com", body="x" * 500), ALICE).allowed


def test_burst_rate_window_blocks_and_recovers():
    clk = FakeClock()
    mon = EgressMonitor(
        EgressLimits(
            max_requests_per_window=3,
            window_seconds=10,
            max_requests=99,
            max_distinct_hosts=99,
            max_cumulative_bytes=10**9,
        ),
        clock=clk,
    )
    for t in (0, 1, 2):
        clk.t = float(t)
        assert mon.check("s", _req(f"h{t}.com"), ALICE).allowed
    clk.t = 3.0
    blocked = mon.check("s", _req("h3.com"), ALICE)
    assert not blocked.allowed and "rate" in blocked.reason
    # Once the window slides past the early calls, egress is allowed again.
    clk.t = 20.0
    assert mon.check("s", _req("h4.com"), ALICE).allowed


def test_sessions_and_principals_are_isolated():
    mon = EgressMonitor(EgressLimits(max_requests=1, max_distinct_hosts=99, max_requests_per_window=99))
    assert mon.check("s1", _req(), ALICE).allowed
    assert not mon.check("s1", _req(), ALICE).allowed  # alice exhausted in s1
    assert mon.check("s2", _req(), ALICE).allowed  # different session, fresh
    assert mon.check("s1", _req(), BOB).allowed  # different principal, fresh


def test_reset_clears_a_session():
    mon = EgressMonitor(EgressLimits(max_requests=1, max_distinct_hosts=99, max_requests_per_window=99))
    assert mon.check("s", _req(), ALICE).allowed
    assert not mon.check("s", _req(), ALICE).allowed
    mon.reset("s", "dev-alice")
    assert mon.check("s", _req(), ALICE).allowed


def test_exempt_hosts_do_not_count_toward_cardinality():
    mon = EgressMonitor(
        EgressLimits(
            max_distinct_hosts=1,
            max_requests=99,
            max_requests_per_window=99,
            exempt_hosts=frozenset({"internal.example.com"}),
        )
    )
    assert mon.check("s", _req("internal.example.com"), ALICE).allowed
    assert mon.check("s", _req("internal.example.com"), ALICE).allowed
    assert mon.check("s", _req("a.com"), ALICE).allowed  # first external host (cardinality 1)
    assert not mon.check("s", _req("b.com"), ALICE).allowed  # second external host -> over limit


def test_authorize_with_egress_runs_percall_guard_first():
    mon = EgressMonitor()
    # The metadata URL is blocked by HARDENED_GUARD's deny_arg_matching before the
    # monitor sees it -> the monitor records nothing.
    blocked = authorize_with_egress(HARDENED_GUARD, mon, "s", _req("169.254.169.254"), ALICE)
    assert isinstance(blocked, EgressDecision) and not blocked.allowed
    # An external host passes the guard and is then recorded by the monitor.
    assert authorize_with_egress(HARDENED_GUARD, mon, "s", _req("api.example.com"), ALICE).allowed


def test_authorize_with_egress_monitor_can_veto_allowed_calls():
    mon = EgressMonitor(EgressLimits(max_requests=1, max_distinct_hosts=99, max_requests_per_window=99))
    assert authorize_with_egress(HARDENED_GUARD, mon, "s", _req("a.com"), ALICE).allowed
    vetoed = authorize_with_egress(HARDENED_GUARD, mon, "s", _req("b.com"), ALICE)
    assert not vetoed.allowed and "count" in vetoed.reason

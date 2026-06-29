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

"""Offline demo of the session-aware egress monitor (PROTOTYPE / SPIKE).

Drives sequences of outbound tool calls through `authorize_with_egress` — the
stateless per-call guard (HARDENED_GUARD) followed by the stateful EgressMonitor —
and prints each verdict. Shows the four aggregate signals plus the layer ordering.
Needs only the standard library: no LLM, no API key, no Guardrails install.

    python demo_egress.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from egress import EgressLimits, EgressMonitor, authorize_with_egress  # noqa: E402
from example_policies import HARDENED_GUARD, PRINCIPALS  # noqa: E402
from policy import Principal, ToolCall  # noqa: E402


def _abbrev(args: dict) -> str:
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 24:
            parts.append(f"{k}=<{len(v)}B>")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _run(title, monitor, session, principal_id, steps):
    print(f"\n=== {title} ===")
    principal = PRINCIPALS.get(principal_id, Principal(principal_id))
    for call, note in steps:
        decision = authorize_with_egress(HARDENED_GUARD, monitor, session, call, principal)
        flag = "ALLOW" if decision.allowed else "BLOCK"
        print(f"  {flag:5}  http_request({_abbrev(call.args)})  — {note}")
        print(f"         {decision.reason}")


def _req(host, **extra):
    return ToolCall("http_request", {"url": f"https://{host}/", **extra})


def main() -> int:
    # 1. Distinct-host fan-out: each request is individually fine, but contacting
    #    too many endpoints in one session is the exfiltration tell.
    mon = EgressMonitor(
        EgressLimits(max_distinct_hosts=3, max_requests=100, max_cumulative_bytes=10**9, max_requests_per_window=100)
    )
    _run(
        "1. Distinct-host fan-out (limit 3)",
        mon,
        "sess-1",
        "dev-alice",
        [
            (_req("a.example.com"), "host #1"),
            (_req("b.example.com"), "host #2"),
            (_req("c.example.com"), "host #3"),
            (_req("d.example.com"), "host #4 -> blocked"),
            (_req("a.example.com"), "already-seen host -> still allowed"),
        ],
    )

    # 2. Cumulative outbound volume: slow drip of data adds up across the session.
    mon = EgressMonitor(
        EgressLimits(max_cumulative_bytes=5000, max_requests=100, max_distinct_hosts=100, max_requests_per_window=100)
    )
    _run(
        "2. Cumulative outbound volume (limit 5000B)",
        mon,
        "sess-2",
        "dev-alice",
        [
            (_req("sink.example.com", body="x" * 2000), "2000B (total 2000)"),
            (_req("sink.example.com", body="x" * 2000), "2000B (total 4000)"),
            (_req("sink.example.com", body="x" * 2000), "2000B (total 6000) -> blocked"),
        ],
    )

    # 3. Layer ordering: the per-call guard blocks a metadata-egress URL before the
    #    monitor is ever consulted (defense in depth).
    mon = EgressMonitor()
    _run(
        "3. Per-call guard runs first",
        mon,
        "sess-3",
        "dev-alice",
        [
            (_req("169.254.169.254"), "cloud metadata -> blocked by HARDENED_GUARD, monitor not consulted"),
            (_req("api.example.com"), "external host -> guard allows, monitor records"),
        ],
    )

    # 4. Burst rate, using a controllable clock so the window is deterministic.
    class _Clock:
        t = 0.0

        def __call__(self):
            return self.t

    clk = _Clock()
    mon = EgressMonitor(
        EgressLimits(
            max_requests_per_window=3,
            window_seconds=10,
            max_requests=100,
            max_distinct_hosts=100,
            max_cumulative_bytes=10**9,
        ),
        clock=clk,
    )
    print("\n=== 4. Burst rate (limit 3 per 10s) ===")
    alice = PRINCIPALS["dev-alice"]
    for i, (t, note) in enumerate(
        [
            (0, "t=0"),
            (1, "t=1"),
            (2, "t=2"),
            (3, "t=3 -> blocked (4 in window)"),
            (20, "t=20 -> window cleared, allowed"),
        ]
    ):
        clk.t = float(t)
        d = authorize_with_egress(HARDENED_GUARD, mon, "sess-4", _req(f"h{i}.example.com"), alice)
        print(f"  {'ALLOW' if d.allowed else 'BLOCK':5}  {note}")
        print(f"         {d.reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

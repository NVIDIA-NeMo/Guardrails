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

"""Offline proof of principle for the agent tool-call guardrail.

Runs the policy core (policy.py) against a small ground-truth set of tool-call
attempts and prints each verdict plus a pass/fail summary. Requires nothing
beyond the standard library — no LLM, no API key, no Guardrails install — so the
guardrail's correctness can be checked deterministically.

    python demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from example_policies import GUARD, PRINCIPALS  # noqa: E402
from policy import Principal, ToolCall  # noqa: E402

# Each case: (principal_id, ToolCall, ground_truth_allowed, note).
CASES = [
    ("dev-alice", ToolCall("read_file", {"path": "src/app.py"}), True, "developer reads a workspace file"),
    ("anon", ToolCall("read_file", {"path": "src/app.py"}), False, "no role permits read_file"),
    (
        "dev-alice",
        ToolCall("run_shell", {"command": "pytest -q", "timeout_seconds": 120}),
        True,
        "in-bounds shell command",
    ),
    (
        "dev-alice",
        ToolCall("run_shell", {"command": "sleep 99999", "timeout_seconds": 7200}),
        False,
        "timeout exceeds the 3600s ceiling",
    ),
    (
        "ci-bot",
        ToolCall("run_shell", {"command": "make build", "timeout_seconds": 60}),
        False,
        "ci role may not run shell commands",
    ),
    (
        "dev-alice",
        ToolCall("write_file", {"path": "src/app.py", "content": "..."}),
        False,
        "write_file is unpoliced, so default-deny",
    ),
    ("dev-alice", ToolCall("deploy_service", {}), False, "unknown tool is denied by default"),
]


def main() -> int:
    print(f"{'result':>7}  {'truth':>6}  tool / note")
    print("-" * 72)
    mismatches = 0
    for principal_id, call, truth, note in CASES:
        principal = PRINCIPALS.get(principal_id, Principal(principal_id))
        decision = GUARD.authorize(call, principal)
        ok = decision.allowed == truth
        mismatches += not ok
        flag = "ALLOW" if decision.allowed else "BLOCK"
        mark = "ok" if ok else "MISMATCH"
        print(f"{flag:>7}  {('allow' if truth else 'block'):>6}  {call.tool} — {note}  [{mark}]")
        print(f"{'':>7}  {'':>6}  reason: {decision.reason}")

    total = len(CASES)
    print("-" * 72)
    print(
        f"{total - mismatches}/{total} match ground truth" + ("" if mismatches == 0 else f"  ({mismatches} MISMATCH)")
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())

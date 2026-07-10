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

"""Apply the human-APPROVED candidates from a review queue, and show before/after.

Phase two of the adoption loop: after a reviewer flips rows to `"approved": true`
in `review_queue.json`, this loads ONLY those, applies them to the guard, and
prints a few before/after authorizations so the effect is visible. Un-approved
candidates are ignored — nothing takes effect until a human says so.

    python3 apply_guardrails.py --queue review_queue.json
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from example_policies import PRINCIPALS, VULNERABLE_GUARD  # noqa: E402
from policy import Principal, ToolCall, ToolCallGuard  # noqa: E402
from synthesis.review import apply, load_approved  # noqa: E402

# Representative calls for the two catalogued candidates, so whichever the reviewer
# approves shows a visible flip. git_push is the clip's headline before/after.
CHECKS = [
    (
        "dev-bob",
        ToolCall("git_push", {"remote": "origin", "branch": "main"}),
        "git_push to a remote dev-bob doesn't own",
    ),
    ("dev-alice", ToolCall("read_file", {"path": "../../etc/passwd"}), "read_file with a path-traversal argument"),
]


def main() -> int:
    p = argparse.ArgumentParser(description="Apply approved candidates from a review queue; show before/after.")
    p.add_argument("--queue", default="review_queue.json", help="the reviewed queue to apply")
    args = p.parse_args()

    approved = load_approved(args.queue)
    if not approved:
        print(f'no approved candidates in {args.queue} — flip a row to "approved": true first, then re-run.')
        return 0

    hardened = ToolCallGuard(apply(approved, VULNERABLE_GUARD))
    print(
        "applied "
        + str(len(approved))
        + " approved rule(s): "
        + ", ".join(f"{c.tool} ← {c.factory_key}" for c in approved)
    )
    print("before → after:")
    for pid, call, note in CHECKS:
        principal = PRINCIPALS.get(pid, Principal(pid))
        before = "ALLOW" if VULNERABLE_GUARD.authorize(call, principal).allowed else "BLOCK"
        decision = hardened.authorize(call, principal)
        after = "ALLOW" if decision.allowed else "BLOCK"
        flip = "  ← now blocked" if before == "ALLOW" and after == "BLOCK" else ""
        print(f"  {note}: {before} → {after}{flip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

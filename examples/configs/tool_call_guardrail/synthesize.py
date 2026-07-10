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

"""Turn scanner findings into a human review queue, and apply approved rules.

Mirrors the notebook's Act 3. `scanner/scan.py` emits findings; this synthesizes
the catalogued ones into rule candidates (each starts unapproved), records the
rest for human triage, and writes a review queue. `--apply` loads only the rows a
human flipped to approved and shows the before/after on the guard.

    python3 scanner/scan.py --docs advisories/ --out findings.json
    python3 synthesize.py findings.json --out review_queue.json
    # a human edits review_queue.json, flipping approved: false -> true
    python3 synthesize.py --apply review_queue.json
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from example_policies import PRINCIPALS, TOOL_REGISTRY, VULNERABLE_GUARD
from policy import Principal, ToolCall, ToolCallGuard
from synthesis.findings import load_findings
from synthesis.proposals import dropped_findings, find_gaps, synthesize
from synthesis.review import apply, load_approved, write_review_queue


def cmd_synthesize(findings_path: str, out_path: str) -> int:
    findings = load_findings(findings_path)
    candidates = synthesize(findings)
    uncatalogued = dropped_findings(findings)
    gaps = find_gaps(VULNERABLE_GUARD, TOOL_REGISTRY)
    write_review_queue(candidates, gaps, out_path, uncatalogued=uncatalogued)

    print(
        f"{len(findings)} techniques found — {len(candidates)} map to vetted rule factories, {len(uncatalogued)} novel."
    )
    for c in candidates:
        print(f"  candidate  {c.tool} <- {c.factory_key}  (approved: false)")
    for f in uncatalogued:
        tool = f.affected_tools[0] if f.affected_tools else "?"
        print(f"  triage     {tool}: {f.title}  (triaged: false)")
    print(f"Nothing applied. Wrote {out_path}; every proposal starts unapproved.")
    return 0


def cmd_apply(queue_path: str) -> int:
    approved = load_approved(queue_path)
    if not approved:
        print("No approved candidates in the queue — nothing to apply.")
        return 0

    new_guard = ToolCallGuard(apply(approved, VULNERABLE_GUARD))
    call = ToolCall("git_push", {"remote": "origin", "branch": "main"})
    principal = PRINCIPALS.get("dev-bob", Principal("dev-bob"))
    before = VULNERABLE_GUARD.authorize(call, principal)
    after = new_guard.authorize(call, principal)

    print(f"Applied {len(approved)} approved rule(s).")
    print("git_push to a remote dev-bob does not own:")
    print(f"  before: {'ALLOW' if before.allowed else 'BLOCK'} · {before.reason}")
    print(f"  after:  {'ALLOW' if after.allowed else 'BLOCK'} · {after.reason}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthesize scanner findings into a review queue, or apply approved rules."
    )
    parser.add_argument("findings", nargs="?", help="findings JSON from scanner/scan.py --out")
    parser.add_argument("--out", default="review_queue.json", help="review queue path to write")
    parser.add_argument(
        "--apply",
        metavar="QUEUE",
        help="apply approved candidates from QUEUE and show the before/after",
    )
    args = parser.parse_args()

    if args.apply:
        return cmd_apply(args.apply)
    if not args.findings:
        parser.error("provide a findings JSON path, or --apply QUEUE")
    return cmd_synthesize(args.findings, args.out)


if __name__ == "__main__":
    raise SystemExit(main())

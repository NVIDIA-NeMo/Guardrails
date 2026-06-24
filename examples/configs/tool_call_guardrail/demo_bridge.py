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

"""Offline proof of the full scanner -> policy pipeline.

Runs the whole pipeline with no LLM, no network, no Guardrails install — starting
from raw source documents, not a hand-written findings file:

    source docs  ->  findings  ->  rule candidates  ->  review queue  ->  approve
                 ->  applied policies  ->  before/after authorization

The offline `KeywordExtractor` stands in for the production LLM extractor, so the
real scanner and the bridge are exercised together, end-to-end and
deterministically.

    python demo_bridge.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from example_policies import (  # noqa: E402
    GUARD,
    PRINCIPAL_ATTRS,
    PRINCIPALS,
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
)
from policy import Principal, ToolCall, ToolCallGuard  # noqa: E402
from scanner.scan import KeywordExtractor, ScanContext, scan  # noqa: E402
from synthesis.catalog import (  # noqa: E402
    CLASS_DESCRIPTIONS,
    CLASS_REQUIRED_PARAMS,
    CLASS_TO_FACTORY,
)
from synthesis.proposals import (  # noqa: E402
    dropped_findings,
    find_gaps,
    synthesize,
)
from synthesis.review import apply, load_approved, write_review_queue  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DOCS = os.path.join(HERE, "scanner", "sample_docs")


def _rule(label: str) -> None:
    print(f"\n=== {label} ===")


def main() -> int:
    scan_ctx = ScanContext(
        docs_dir=SAMPLE_DOCS,
        tool_registry=dict(TOOL_REGISTRY),
        taxonomy=tuple(CLASS_TO_FACTORY),
        class_definitions=dict(CLASS_DESCRIPTIONS),
        class_params=dict(CLASS_REQUIRED_PARAMS),
        tool_schemas=dict(TOOL_SCHEMAS),
        principal_attrs=tuple(PRINCIPAL_ATTRS),
    )
    findings = scan(scan_ctx, KeywordExtractor())

    _rule("1. Findings produced by the offline scanner over sample_docs")
    for f in findings:
        print(f"  - [{f.attack_class}] {f.id}: {f.title}")

    _rule("2. Coverage gaps in the current guard")
    gaps = find_gaps(GUARD, TOOL_REGISTRY)
    for g in gaps:
        print(f"  - {g.tool}: {g.missing}")

    _rule("3. Candidates synthesized (unknown classes dropped, fail-closed)")
    candidates = synthesize(findings)
    for c in candidates:
        print(f"  - {c.tool} <- {c.factory_key}({c.params})  [from {c.finding_id}]")
    for d in dropped_findings(findings):
        print(f"  - DROPPED {d.id}: attack_class '{d.attack_class}' not in catalog")

    queue_path = os.path.join(tempfile.mkdtemp(prefix="t5_bridge_"), "review.json")
    write_review_queue(candidates, gaps, queue_path)
    _rule("4. Review queue written for the human gate")
    print(f"  {queue_path}")
    print('  (every candidate starts "approved": false)')

    # --- Human gate (simulated) --------------------------------------------
    # In reality a person edits the queue file and flips the entries they trust.
    # Here we approve every *valid* candidate to drive the rest of the demo.
    with open(queue_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    for entry in payload["candidates"]:
        entry["approved"] = True
    with open(queue_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    approved = load_approved(queue_path)
    _rule("5. Reviewer approved (simulated): candidates loaded back")
    for c in approved:
        print(f"  - {c.tool} <- {c.factory_key}({c.params})")

    updated = apply(approved, GUARD)
    new_guard = ToolCallGuard(updated)

    _rule("6. Before vs. after authorization")
    checks = [
        (
            "dev-bob",
            ToolCall("git_push", {"remote": "origin", "branch": "main"}),
            "push to a remote dev-bob does not own (ownership rule added)",
        ),
        (
            "dev-alice",
            ToolCall("run_shell", {"command": "sleep 600", "timeout_seconds": 600}),
            "shell with 600s timeout (was under old 3600 ceiling)",
        ),
        (
            "dev-alice",
            ToolCall("install_package", {"name": "leftpad-evil", "version": "1.0"}),
            "install a denylisted package (denylist added to an existing tool)",
        ),
        (
            "dev-alice",
            ToolCall("write_file", {"path": "src/app.py", "content": "..."}),
            "write_file (newly-policied tool, fail-closed on roles)",
        ),
    ]
    for principal_id, call, note in checks:
        principal = PRINCIPALS.get(principal_id, Principal(principal_id))
        before = GUARD.authorize(call, principal)
        after = new_guard.authorize(call, principal)
        print(f"  - {note}")
        print(f"      before: {'ALLOW' if before.allowed else 'BLOCK'} — {before.reason}")
        print(f"      after:  {'ALLOW' if after.allowed else 'BLOCK'} — {after.reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

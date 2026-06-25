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

"""Verify a Kibana JSON export against a smoke-run manifest.

Offline verifier for environments where the operator has Kibana
Discover access but cannot issue an API key. Workflow:

  1. Run ``scripts/telemetry_smoke.py``. Note the manifest path it
     prints.
  2. Open Kibana Discover at the staging telemetry data view
     (``<staging-telemetry-data-view>``). Filter on the exact
     ``kibana_filter`` printed by the smoke driver and saved in the
     manifest, for example:

         client.sessionId : ("id1" or "id2" or "id3")

  3. Share -> get the JSON of the visible documents. Save the file
     locally (e.g. ``kibana.json``).
  4. Run this script:

         poetry run python scripts/kibana_verify_export.py \\
             --manifest <run_dir>/manifest.json \\
             --export kibana.json

The script ignores any documents in the export whose ``eventName``
is not ``guardrails_usage_event`` (Kibana exports tend to include
sibling event types if the filter is broad).

Per-scenario logic:

  - Positive scenarios: assert the number of documents with a
    matching exact startup session ID is >= ``expected_event_count``.
  - Negative scenarios (``opt_out_*``, expected count 0): assert
    exactly zero documents for their recorded session IDs.
  - Scenarios whose local smoke verdict was FAIL: report FAIL. Receiver
    verification cannot make a broken smoke scenario successful.

Exit codes:
  0  every scenario verified
  1  one or more scenarios failed
  2  configuration error (missing files, malformed manifest)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVENT_NAME = "guardrails_usage_event"


def _load_json(path: Path, what: str) -> Any:
    if not path.exists():
        sys.stderr.write(f"{what} not found: {path}\n")
        sys.exit(2)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"{what} is not valid JSON ({path}): {exc}\n")
        sys.exit(2)


def _extract_session_id(doc: dict[str, Any]) -> str:
    """Pull the guardrails sessionId from a Kibana export document.

    Kibana export shape: ``doc["fields"]["client.sessionId"]`` is a
    list because Kibana wraps every field value in a list. This is the
    staging field exposed for smoke-run filtering.
    """
    fields = doc.get("fields", {})
    values = fields.get("client.sessionId", [])
    if not values:
        return ""
    return str(values[0])


def _is_guardrails_event(doc: dict[str, Any]) -> bool:
    fields = doc.get("fields", {})
    event_names = fields.get("eventName", [])
    return bool(event_names) and event_names[0] == EVENT_NAME


def _bucket_docs_by_session_id(docs: list[dict[str, Any]], session_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Group guardrails docs by exact client.sessionId."""
    session_id_set = set(session_ids)
    buckets: dict[str, list[dict[str, Any]]] = {session_id: [] for session_id in session_id_set}
    for doc in docs:
        if not _is_guardrails_event(doc):
            continue
        session_id = _extract_session_id(doc)
        if session_id in session_id_set:
            buckets[session_id].append(doc)
    return buckets


def _scenario_session_ids(scenario: dict[str, Any]) -> list[str]:
    return list(scenario.get("startup_session_ids", []))


def _verify(manifest: dict[str, Any], docs: list[dict[str, Any]]) -> tuple[int, int]:
    scenarios = manifest.get("results", [])
    if not scenarios:
        raise ValueError("manifest has no results to verify")

    all_session_ids = sorted({session_id for scenario in scenarios for session_id in _scenario_session_ids(scenario)})
    buckets = _bucket_docs_by_session_id(docs, all_session_ids)

    pass_count = 0
    fail_count = 0

    for scenario in scenarios:
        name = scenario["name"]
        expected = scenario["expected_event_count"]
        session_ids = _scenario_session_ids(scenario)

        if scenario.get("verdict") == "FAIL":
            print(f"[{name}] FAIL  (smoke verdict was FAIL)")
            fail_count += 1
            continue

        actual = sum(len(buckets.get(session_id, [])) for session_id in session_ids)
        if expected == 0:
            ok = actual == 0
            comparator = "=="
        else:
            ok = actual >= expected
            comparator = ">="

        verdict = "PASS" if ok else "FAIL"
        print(f"[{name}] {verdict}  (kibana={actual}, expected{comparator}{expected})")
        if ok:
            pass_count += 1
        else:
            fail_count += 1

    return pass_count, fail_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to manifest.json from a smoke run.")
    parser.add_argument(
        "--export",
        required=True,
        help="Path to the JSON file downloaded from Kibana Discover (array of documents).",
    )
    args = parser.parse_args()

    manifest = _load_json(Path(args.manifest), "manifest")
    docs = _load_json(Path(args.export), "Kibana export")

    if not isinstance(docs, list):
        sys.stderr.write(f"Kibana export must be a JSON array of documents; got {type(docs).__name__}\n")
        sys.exit(2)

    guardrails_docs = sum(1 for doc in docs if _is_guardrails_event(doc))
    print(
        f"manifest: {args.manifest}\n"
        f"export:   {args.export} ({len(docs)} docs total, {guardrails_docs} guardrails_usage_event)\n"
    )

    try:
        pass_count, fail_count = _verify(manifest, docs)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    print()
    print(f"=== summary: {pass_count} PASS, {fail_count} FAIL ===")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

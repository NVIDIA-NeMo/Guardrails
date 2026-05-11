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

"""Extract the guardrails-only slice from an upstream anonymous_events.json.

Used to refresh ``schemas/anonymous_events.snapshot.json`` from the latest
copy of the shared nemo-telemetry schema. Keeps only:

- ``definitions.events.guardrails_usage_event``
- ``definitions.types.<types referenced from the event>`` (transitively)
- ``oneOf`` entry pointing at ``guardrails_usage_event``

The set of types is discovered by walking the event definition for
``$ref`` strings under ``#/definitions/types/...``, then walking those
types for further ``$ref``s, so adding a new reference upstream pulls
the corresponding type definition automatically. Fails loudly if a
referenced type is not present in the upstream's ``definitions.types``.

Intended for use in a sync workflow: fetch upstream's
``anonymous_events.json``, run this script, ``git diff`` the snapshot;
non-empty diff means the upstream contract moved.

Usage:
    poetry run python scripts/extract_telemetry_snapshot.py UPSTREAM_PATH
    poetry run python scripts/extract_telemetry_snapshot.py UPSTREAM_PATH OUTPUT_PATH
    poetry run python scripts/extract_telemetry_snapshot.py UPSTREAM_PATH -

If ``OUTPUT_PATH`` is omitted, writes to
``schemas/anonymous_events.snapshot.json`` next to the repo root.
If ``OUTPUT_PATH`` is ``-``, writes to stdout.
"""

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "schemas" / "anonymous_events.snapshot.json"
EVENT_NAME = "guardrails_usage_event"
TYPE_REF_PREFIX = "#/definitions/types/"


def _collect_type_refs(node: Any, found: set[str]) -> None:
    """Walk ``node`` recursively and add any ``#/definitions/types/<Name>``
    refs found into ``found``.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str) and value.startswith(TYPE_REF_PREFIX):
                found.add(value[len(TYPE_REF_PREFIX) :])
            else:
                _collect_type_refs(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_type_refs(item, found)


def slim(upstream: dict[str, Any]) -> dict[str, Any]:
    """Build the slimmed snapshot from a full upstream document.

    Raises ``SystemExit`` if the upstream document does not contain
    ``guardrails_usage_event`` or references a type that is not defined
    in ``definitions.types``.
    """
    events = upstream.get("definitions", {}).get("events", {})
    if EVENT_NAME not in events:
        raise SystemExit(f"upstream schema does not contain event '{EVENT_NAME}'")
    event_def = events[EVENT_NAME]

    available_types = upstream.get("definitions", {}).get("types", {})
    used_types: set[str] = set()
    _collect_type_refs(event_def, used_types)

    closed: set[str] = set()
    while used_types - closed:
        target = next(iter(used_types - closed))
        closed.add(target)
        if target in available_types:
            _collect_type_refs(available_types[target], used_types)

    missing = used_types - set(available_types)
    if missing:
        raise SystemExit(f"upstream schema is missing referenced types: {sorted(missing)}")

    slim_types = {name: available_types[name] for name in available_types if name in used_types}

    return {
        "$schema": upstream["$schema"],
        "description": upstream["description"],
        "schemaMeta": upstream["schemaMeta"],
        "definitions": {
            "types": slim_types,
            "events": {EVENT_NAME: event_def},
        },
        "oneOf": [{"$ref": f"#/definitions/events/{EVENT_NAME}"}],
    }


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        sys.stderr.write(__doc__ or "")
        return 0 if args and args[0] in ("-h", "--help") else 2

    upstream_path = Path(args[0])
    output_arg = args[1] if len(args) >= 2 else None

    upstream = json.loads(upstream_path.read_text())
    payload = json.dumps(slim(upstream), indent=4) + "\n"

    if output_arg == "-":
        sys.stdout.write(payload)
        return 0

    out = Path(output_arg) if output_arg else DEFAULT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload)
    sys.stderr.write(f"wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

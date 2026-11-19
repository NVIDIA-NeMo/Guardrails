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

"""The decision record: every decision the hooks server gives, one JSON line each, chained by hash.

The record is one append-only file, `hooks_server.decision_record`. Each line is an entry: when, what kind of check
(an agent start, a tool call or a model call), which agent by name and package digest, which run, what was asked,
the outcome, the reason and which rule decided. Each entry carries `prev`, the `hash` of the entry before it (the
first entry's `prev` is all zeros), and its own `hash`, the sha256 of the entry's canonical JSON without `hash`. A
line changed, removed or put in another place after it was written breaks the chain, and `check` names where.

The chain shows that the file was not changed after it was written, as long as its last hash is kept somewhere the
writer of the file can't change. It doesn't stop a writer from writing a new chain from the start.
"""

import fcntl
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

GENESIS = "sha256:" + "0" * 64


class DecisionRecordError(Exception):
    """The record can't be read or written; a check that has to be recorded can't be answered."""


def entry_hash(entry: dict[str, Any]) -> str:
    """The hash of *entry*: sha256 over its canonical JSON, without its own `hash`."""
    body = {key: value for key, value in entry.items() if key != "hash"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class DecisionRecord:
    """The record file. Several server processes on one host can share it: each append takes an exclusive lock and
    reads the last line again under it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        # Refuse a record that can't be read at start, as the quarantine store does.
        result = check(path)
        if not result["intact"]:
            raise DecisionRecordError(f"the decision record {path} is not intact: {result['problems'][0]}")

    def append(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Add one entry after the last one, flushed to disk before the check answers, and return it as written."""
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a+", encoding="utf-8") as record:
                    fcntl.flock(record, fcntl.LOCK_EX)
                    try:
                        last = _last_entry(record)
                        written = {
                            "seq": 1 if last is None else int(last["seq"]) + 1,
                            "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                            **entry,
                            "prev": GENESIS if last is None else last["hash"],
                        }
                        written["hash"] = entry_hash(written)
                        record.seek(0, os.SEEK_END)
                        record.write(json.dumps(written, sort_keys=True, ensure_ascii=False) + "\n")
                        record.flush()
                        os.fsync(record.fileno())
                    finally:
                        fcntl.flock(record, fcntl.LOCK_UN)
            except (OSError, ValueError, KeyError) as exc:
                raise DecisionRecordError(f"the decision record {self.path} can't be written: {exc}") from exc
            return written


def _last_entry(record) -> Optional[dict[str, Any]]:
    """The last entry of an open record file, or None when it is empty."""
    record.seek(0)
    last = None
    for line in record:
        if line.strip():
            last = line
    return None if last is None else json.loads(last)


def read(path: Path) -> list[dict[str, Any]]:
    """Every entry of the record, in order. A record that doesn't exist yet has none."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise DecisionRecordError(f"the decision record {path} can't be read: {exc}") from exc
    entries = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except ValueError as exc:
            raise DecisionRecordError(f"line {number} of the decision record {path} isn't JSON") from exc
    return entries


def check(path: Path, head: Optional[str] = None) -> dict[str, Any]:
    """Whether the chain holds: each entry's hash is its own, and each `prev` is the hash before it.

    With *head*, the hash the last entry must have, kept apart from the file: a record cut short or written again from
    the start doesn't end on it."""
    problems: list[str] = []
    entries: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    except OSError as exc:
        lines = []
        problems.append(f"the file can't be read: {exc}")
    prev = GENESIS
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            problems.append(f"line {number}: not JSON")
            break
        if not isinstance(entry, dict) or "hash" not in entry or "prev" not in entry:
            problems.append(f"line {number}: not a decision entry")
            break
        if entry["prev"] != prev:
            problems.append(f"line {number}: prev is not the hash of the line before it")
        if entry_hash(entry) != entry["hash"]:
            problems.append(f"line {number}: the entry was changed after it was written")
        prev = entry["hash"]
        entries.append(entry)
    if head is not None and prev != head:
        problems.append(f"the last entry's hash is {prev}, not the head {head}")
    return {"file": str(path), "entries": len(entries), "head": prev, "intact": not problems, "problems": problems}


def select(
    entries: Iterable[dict[str, Any]],
    *,
    kind: Optional[str] = None,
    outcome: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    """The entries that match every filter given."""
    wanted = {"kind": kind, "outcome": outcome, "run_id": run_id, "agent_name": agent_name}
    return [
        entry for entry in entries if all(value is None or entry.get(key) == value for key, value in wanted.items())
    ]

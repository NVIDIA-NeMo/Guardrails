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

"""Quarantined agents: an agent, named with the digest of its package, that may not start or run until released.

The store is one append-only file of JSON lines, `hooks_server.quarantine_store`. Each line is an event: an agent was
quarantined, or released. Nothing is ever removed, so the file is the history, and the state is what the events add
up to. A server that restarts reads the file again. Several server processes on one host can share the file: each
write takes an exclusive lock, and each check reads the file again when it changed. Servers on several hosts need a
shared store with the same events instead of a local file.
"""

import fcntl
import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

QUARANTINED = "quarantined"
RELEASED = "released"


class QuarantineStoreError(Exception):
    """The store can't be read or written; a check that needs it can't finish."""


@dataclass(frozen=True)
class QuarantineEvent:
    """One line of the store."""

    event: Literal["quarantined", "released"]
    agent_name: str
    agent_digest: str
    reason: str
    by: str
    at: str
    # What asked: `rule` for a rule of the server's config, with the rule's text in `rule`; `request` for a person or
    # the platform, with no violation.
    source: Literal["rule", "request"] = "request"
    rule: Optional[str] = None


class QuarantineStore:
    """The quarantine file, and the state its events add up to."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._seen: Optional[tuple[int, int]] = None
        self._active: dict[tuple[str, str], QuarantineEvent] = {}
        self._load()

    def _load(self) -> None:
        """Read the file again when it changed since the last read. Raises `QuarantineStoreError` on a bad line."""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self._seen, self._active = None, {}
            return
        except OSError as exc:
            raise QuarantineStoreError(f"the quarantine store {self.path} can't be read: {exc}") from exc
        seen = (stat.st_mtime_ns, stat.st_size)
        if seen == self._seen:
            return
        active: dict[tuple[str, str], QuarantineEvent] = {}
        try:
            with self.path.open(encoding="utf-8") as lines:
                for number, line in enumerate(lines, start=1):
                    if not line.strip():
                        continue
                    try:
                        event = QuarantineEvent(**json.loads(line))
                    except (ValueError, TypeError) as exc:
                        raise QuarantineStoreError(
                            f"line {number} of the quarantine store {self.path} isn't a quarantine event"
                        ) from exc
                    key = (event.agent_name, event.agent_digest)
                    if event.event == QUARANTINED:
                        active[key] = event
                    else:
                        active.pop(key, None)
        except OSError as exc:
            raise QuarantineStoreError(f"the quarantine store {self.path} can't be read: {exc}") from exc
        self._seen, self._active = seen, active

    def _append(self, event: QuarantineEvent) -> None:
        """Write one event under an exclusive lock, flushed to disk before the check answers."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as store:
                fcntl.flock(store, fcntl.LOCK_EX)
                try:
                    store.write(json.dumps(asdict(event), sort_keys=True) + "\n")
                    store.flush()
                    os.fsync(store.fileno())
                finally:
                    fcntl.flock(store, fcntl.LOCK_UN)
        except OSError as exc:
            raise QuarantineStoreError(f"the quarantine store {self.path} can't be written: {exc}") from exc

    def find(self, agent_name: str, agent_digest: Optional[str]) -> Optional[QuarantineEvent]:
        """The quarantine that holds the agent: of that exact digest, or, with no digest given, of any version of it."""
        with self._lock:
            self._load()
            if agent_digest is not None:
                return self._active.get((agent_name, agent_digest))
            return next((event for (name, _), event in self._active.items() if name == agent_name), None)

    def active(self) -> list[QuarantineEvent]:
        """Every quarantine in force, oldest first."""
        with self._lock:
            self._load()
            return sorted(self._active.values(), key=lambda event: event.at)

    def quarantine(
        self,
        agent_name: str,
        agent_digest: str,
        reason: str,
        by: str,
        source: Literal["rule", "request"],
        rule: Optional[str] = None,
    ) -> QuarantineEvent:
        """Quarantine the agent, or return the quarantine already in force for it."""
        with self._lock:
            self._load()
            held = self._active.get((agent_name, agent_digest))
            if held is not None:
                return held
            event = QuarantineEvent(QUARANTINED, agent_name, agent_digest, reason, by, _now(), source, rule)
            self._append(event)
            self._load()
            return event

    def release(self, agent_name: str, agent_digest: str, reason: str, by: str) -> Optional[QuarantineEvent]:
        """Release the agent and record who did and why; None when it isn't quarantined."""
        with self._lock:
            self._load()
            if (agent_name, agent_digest) not in self._active:
                return None
            event = QuarantineEvent(RELEASED, agent_name, agent_digest, reason, by, _now())
            self._append(event)
            self._load()
            return event


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

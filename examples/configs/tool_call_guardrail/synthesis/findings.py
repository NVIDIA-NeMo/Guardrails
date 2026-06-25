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

"""The scanner's output and the coverage analyzer's output.

A `Finding` is whatever a field-scanning agent emits about an agent-exploitation
technique. It is treated as *untrusted data*: nothing in it is executed, and its
`evidence` text is for a human reviewer's eyes only. A `CoverageGap` is the
inverse signal — a tool the guard does not yet protect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Finding:
    """One agent-exploitation technique surfaced by the scanner (untrusted)."""

    id: str  # stable, e.g. "2026-confused-deputy-arg-injection"
    title: str
    source: str  # citation/URL — provenance, carried through to review
    attack_class: str  # taxonomy key, mapped to a rule factory in catalog.py
    affected_tools: Sequence[str] = ()  # tool names this concerns
    suggested_params: Mapping[str, object] = field(default_factory=dict)
    evidence: str = ""  # raw text for the human reviewer; never executed

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Finding":
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            source=str(data.get("source", "")),
            attack_class=str(data["attack_class"]),
            affected_tools=tuple(data.get("affected_tools", ())),
            suggested_params=dict(data.get("suggested_params", {})),
            evidence=str(data.get("evidence", "")),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "attack_class": self.attack_class,
            "affected_tools": list(self.affected_tools),
            "suggested_params": dict(self.suggested_params),
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class CoverageGap:
    """A tool the guard does not protect, or an attack class it does not cover."""

    tool: str
    missing: str  # human-readable description of what is missing


@dataclass(frozen=True)
class NovelCluster:
    """Uncatalogued findings aggregated for one tool — a prompt to consider a new
    rule factory. A reporting signal only; it proposes no rule and is never acted
    on automatically."""

    tool: str
    count: int
    attack_classes: Sequence[str]  # the distinct uncatalogued classes seen
    finding_ids: Sequence[str]
    examples: Sequence[str]  # a few finding titles for human context


def load_findings(path: str) -> list[Finding]:
    """Load a JSON array of findings (the scanner's serialized output)."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [Finding.from_dict(item) for item in raw]

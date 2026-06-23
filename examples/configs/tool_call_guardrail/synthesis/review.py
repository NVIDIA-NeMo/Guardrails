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

"""The human gate.

`write_review_queue` serializes proposals to a file a person reads and edits:
every candidate lands with `"approved": false`, and only the ones a reviewer
flips to `true` are ever loaded by `load_approved`. `apply` then merges the
approved candidates into a fresh set of policies — adding rules to existing tool
policies, and creating *fail-closed* (no allowed roles) policies for new tools so
an approved rule can never, by itself, open access.
"""

from __future__ import annotations

import json
from typing import Sequence

from policy import ToolCallGuard, ToolPolicy

from .catalog import RuleCandidate
from .findings import CoverageGap


def write_review_queue(
    candidates: Sequence[RuleCandidate],
    gaps: Sequence[CoverageGap],
    path: str,
) -> str:
    """Write proposals to `path` for human review. All start unapproved."""
    payload = {
        "candidates": [c.to_dict() for c in candidates],
        "coverage_gaps": [{"tool": g.tool, "missing": g.missing} for g in gaps],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return path


def load_approved(path: str) -> list[RuleCandidate]:
    """Load only the candidates a human marked `"approved": true` and that are
    valid against the vetted catalog."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    approved: list[RuleCandidate] = []
    for entry in payload.get("candidates", []):
        if not entry.get("approved"):
            continue
        candidate = RuleCandidate.from_dict(entry)
        if candidate.is_valid():
            approved.append(candidate)
    return approved


def apply(approved: Sequence[RuleCandidate], guard: ToolCallGuard) -> dict[str, ToolPolicy]:
    """Merge approved candidates into a new policy map derived from `guard`.

    Existing tools keep their `allowed_roles` and gain the new rule. A candidate
    for a tool with no existing policy creates one with *no* allowed roles, so the
    tool stays default-deny on roles until a human grants access explicitly.
    """
    policies: dict[str, ToolPolicy] = {tool: guard.policy_for(tool) for tool in guard.registered_tools()}
    for candidate in approved:
        rule = candidate.materialize()
        existing = policies.get(candidate.tool)
        if existing is not None:
            policies[candidate.tool] = ToolPolicy(
                allowed_roles=existing.allowed_roles,
                rules=[*existing.rules, rule],
            )
        else:
            policies[candidate.tool] = ToolPolicy(allowed_roles=frozenset(), rules=[rule])
    return policies

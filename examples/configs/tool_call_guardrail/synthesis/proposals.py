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

"""Turn findings into rule candidates, and find what the guard does not cover.

`synthesize` is the proposal side of the bridge: it maps each finding's attack
class to a vetted factory and emits one `RuleCandidate` per affected tool,
silently dropping any finding whose class is not in the catalog (fail closed).
`find_gaps` is the inverse: it reports tools the guard has no policy for, and
`cluster_uncatalogued` is the other inverse: it reports tools under repeated
pressure from techniques the catalog cannot yet express.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from policy import ToolCallGuard

from .catalog import CLASS_TO_FACTORY, RuleCandidate
from .findings import CoverageGap, Finding, NovelCluster

UNTARGETED = "(untargeted)"


def synthesize(findings: Sequence[Finding]) -> list[RuleCandidate]:
    """Map findings to rule candidates via the vetted catalog.

    Findings whose `attack_class` is not in `CLASS_TO_FACTORY` produce no
    candidate — an unrecognized technique is never silently auto-acted upon.
    """
    candidates: list[RuleCandidate] = []
    for finding in findings:
        factory_key = CLASS_TO_FACTORY.get(finding.attack_class)
        if factory_key is None:
            continue  # fail closed: unknown attack class -> no proposal
        for tool in finding.affected_tools:
            candidates.append(
                RuleCandidate(
                    finding_id=finding.id,
                    source=finding.source,
                    tool=tool,
                    factory_key=factory_key,
                    params=finding.suggested_params,
                    rationale=finding.title,
                )
            )
    return candidates


def dropped_findings(findings: Sequence[Finding]) -> list[Finding]:
    """Findings that produced no candidate because their class is uncatalogued."""
    return [f for f in findings if f.attack_class not in CLASS_TO_FACTORY]


def find_gaps(guard: ToolCallGuard, tool_registry: Iterable[str]) -> list[CoverageGap]:
    """Tools present in the agent's registry but lacking any guard policy."""
    covered = guard.registered_tools()
    return [CoverageGap(tool, "no policy registered (default-deny)") for tool in tool_registry if tool not in covered]


def cluster_uncatalogued(findings: Sequence[Finding], min_count: int = 1) -> list[NovelCluster]:
    """Aggregate uncatalogued findings by affected tool.

    The catalog could not express these techniques, so they became no rule (see
    `dropped_findings`). Grouping them by tool turns scattered `novel` hits into a
    ranked signal: a tool with several uncatalogued findings is a candidate for a
    new rule factory. A finding naming no tool is bucketed under `UNTARGETED`.
    Clusters below `min_count` are omitted. This only *reports* — it proposes no
    rule and nothing here is ever auto-applied.
    """
    by_tool: dict[str, list[Finding]] = {}
    for finding in dropped_findings(findings):
        tools = finding.affected_tools or (UNTARGETED,)
        for tool in tools:
            by_tool.setdefault(tool, []).append(finding)

    clusters = [
        NovelCluster(
            tool=tool,
            count=len(group),
            attack_classes=tuple(sorted({f.attack_class for f in group})),
            finding_ids=tuple(f.id for f in group),
            examples=tuple(f.title for f in group[:3]),
        )
        for tool, group in by_tool.items()
        if len(group) >= min_count
    ]
    clusters.sort(key=lambda c: (-c.count, c.tool))
    return clusters


def format_factory_prompt(clusters: Sequence[NovelCluster]) -> str:
    """Render `cluster_uncatalogued` output as a human-facing 'consider a new
    factory' report. Returns a message even when there is nothing to act on."""
    if not clusters:
        return "No uncatalogued findings: the catalog covers every surfaced technique."

    lines = [
        "Uncatalogued pressure — techniques the catalog cannot express yet.",
        "Each line is a prompt to consider designing a new rule factory (a "
        "deliberate, human-authored code change); nothing here is auto-applied.",
        "",
    ]
    for cluster in clusters:
        classes = ", ".join(cluster.attack_classes)
        lines.append(
            f"  {cluster.tool}: {cluster.count} uncatalogued finding(s) [{classes}] "
            "— consider a new rule factory for this tool"
        )
        lines.extend(f"      - {title}" for title in cluster.examples)
    return "\n".join(lines)

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
`find_gaps` is the inverse: it reports tools the guard has no policy for.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from policy import ToolCallGuard

from .catalog import CLASS_TO_FACTORY, RuleCandidate
from .findings import CoverageGap, Finding


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

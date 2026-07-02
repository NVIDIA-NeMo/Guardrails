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

"""Adapter: a garak audit hitlog -> scanner Findings (deterministic, no LLM).

A garak ``.hitlog.jsonl`` is pre-structured intelligence: garak ran a probe
against a live target and a detector judged the response a hit. Each line is a
serialized garak ``Attempt``. This turns those hits into `Finding`s for the same
synthesis -> human-gate pipeline the literature scanner feeds, so a vulnerability
Auditor (garak) demonstrates lands in the guardrail's review queue with full
evidence instead of becoming a rule automatically.

The mapping is deliberately conservative. A hitlog says *what* was demonstrated
and carries the probe's OWASP/payload tags, but not the concrete rule parameters
(a pattern, a ceiling, a denylist) nor, reliably, which registry tool is
affected. So every hit is routed to the uncatalogued/triage path — never
auto-parameterized into a rule — carrying the probe's OWASP tags and the
guardrail classes that share them as hints. A human turns that evidence into a
policy at the review gate, exactly as for a literature-sourced `novel` finding.

The probe -> tags lookup is injected (`ProbeTagResolver`) so this stays free of
any garak dependency: a live run backs it with garak_api's plugin cache; tests
stub it.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Iterable, Sequence

from synthesis.catalog import classes_for_owasp
from synthesis.findings import Finding

# A garak hit is evidence, not a catalogued technique with known parameters, so
# it is routed to the human-triage path — any class outside CLASS_TO_FACTORY is.
GARAK_HIT_CLASS = "garak-audit"

# Resolves a garak probe classname (e.g. "agent_breaker.AgentBreaker") to its
# tags (e.g. "owasp:llm06", "payload:agentic:exploitation").
ProbeTagResolver = Callable[[str], Sequence[str]]


def load_hitlog(path: str) -> list[dict]:
    """Parse a garak ``.hitlog.jsonl`` into attempt dicts, skipping blank lines,
    malformed lines, and any line without a ``probe_classname`` (garak reports
    also carry non-attempt header/digest lines)."""
    entries: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("probe_classname"):
                entries.append(obj)
    return entries


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower() or "x"


def _snippet(value: object, limit: int = 200) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _owasp(tags: Sequence[str]) -> list[str]:
    return [t for t in tags if t.startswith("owasp:")]


def hitlog_to_findings(
    entries: Iterable[dict],
    resolve_tags: ProbeTagResolver,
    target: str,
) -> list[Finding]:
    """Turn garak hitlog attempts into triage `Finding`s, one per distinct probe.

    Attempts from the same probe are aggregated into one finding (with the hit
    count recorded) so a chatty probe does not flood the queue. Findings are
    returned sorted by probe name for determinism.
    """
    by_probe: dict[str, list[dict]] = {}
    for entry in entries:
        probe = str(entry.get("probe_classname", "")).strip()
        if probe:
            by_probe.setdefault(probe, []).append(entry)

    findings: list[Finding] = []
    for probe in sorted(by_probe):
        group = by_probe[probe]
        tags = list(resolve_tags(probe) or ())
        owasp = _owasp(tags)
        related = sorted({cls for tag in owasp for cls in classes_for_owasp(tag)})
        first = group[0]
        goal = str(first.get("goal", "")).strip()

        evidence = [
            f"garak probe: {probe}",
            f"hits: {len(group)}",
            f"tags: {', '.join(tags) or '(none resolved)'}",
        ]
        if goal:
            evidence.append(f"goal: {goal}")
        if related:
            evidence.append("guardrail classes sharing these OWASP tags (candidates for a rule): " + ", ".join(related))
        if first.get("prompt") is not None:
            evidence.append(f"sample prompt: {_snippet(first.get('prompt'))}")
        outputs = first.get("outputs") or []
        if outputs:
            evidence.append(f"sample output: {_snippet(outputs[0])}")
        detectors = sorted((first.get("detector_results") or {}).keys())
        if detectors:
            evidence.append(f"detectors: {', '.join(detectors)}")

        title = f"garak probe {probe} landed against {target}"
        if goal:
            title += f" — {goal}"

        findings.append(
            Finding(
                id=f"garak-{_slug(target)}-{_slug(probe)}",
                title=title,
                source=f"garak-audit:{target}",
                attack_class=GARAK_HIT_CLASS,
                affected_tools=(),  # a hitlog does not name a registry tool; the human maps it
                suggested_params={},
                evidence="\n".join(evidence),
            )
        )
    return findings

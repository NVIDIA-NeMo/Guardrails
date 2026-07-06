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
from typing import Callable, Iterable, Mapping, Sequence

from synthesis.catalog import classes_for_owasp, classes_for_payload
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


def plugin_cache_resolver(cache_path: str) -> ProbeTagResolver:
    """Build a `ProbeTagResolver` backed by garak's ``plugin_cache.json`` (ships
    with garak / the ``garak_api`` package). Reads the file once; resolves a
    probe's tags trying both the bare classname and the ``probes.``-prefixed key
    garak uses in the cache, so it works whichever form the hitlog records.
    Depends only on the data file, not the ``garak_api`` package — so it runs
    wherever a copy of the cache is reachable."""
    with open(cache_path, "r", encoding="utf-8") as fh:
        cache = json.load(fh)
    probes = cache.get("probes", {}) if isinstance(cache, dict) else {}

    def resolve(probe_classname: str) -> tuple[str, ...]:
        entry = probes.get(probe_classname) or probes.get(f"probes.{probe_classname}")
        if not isinstance(entry, dict):
            return ()
        return tuple(t for t in entry.get("tags", []) if isinstance(t, str))

    return resolve


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower() or "x"


def _snippet(value: object, limit: int = 200) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _attempt_tool(entry: Mapping) -> str:
    """The specific tool an attempt targeted, if the probe records one.

    Agentic probes name the tool under attack in the attempt notes (AgentBreaker
    uses ``notes.current_target``); most probes carry none. An empty string means
    "not tool-specific" and the attempt groups at the probe level, as before."""
    notes = entry.get("notes") or {}
    for key in ("current_target", "target_tool", "tool"):
        value = notes.get(key)
        if value:
            return str(value).strip()
    return ""


def _is_hit(entry: Mapping, threshold: float) -> bool:
    """True when a detector scored this attempt at/above `threshold` — i.e. the
    attack actually landed. A garak *report* carries non-hit attempts too, so this
    is what separates a demonstrated vulnerability from an attempt that failed."""
    results = entry.get("detector_results") or {}
    for scores in results.values():
        if any((score or 0) >= threshold for score in (scores or [])):
            return True
    return False


def _vulnerability_info(entry: Mapping) -> str:
    """A probe's human-readable description of the demonstrated weakness, if any
    (AgentBreaker records one per tool in ``notes.vulnerability_info``)."""
    return str((entry.get("notes") or {}).get("vulnerability_info", "")).strip()


def hitlog_to_findings(
    entries: Iterable[dict],
    resolve_tags: ProbeTagResolver,
    target: str,
    hit_threshold: float = 0.5,
) -> list[Finding]:
    """Turn garak attempts into triage `Finding`s, one per distinct (probe, tool).

    Only attempts a detector actually scored as hits (``>= hit_threshold``) become
    findings — a report's non-hit attempts are not demonstrated vulnerabilities.
    Hits are grouped by (probe, tool): agentic probes name the tool under attack
    (see `_attempt_tool`), so AgentBreaker yields one reviewer-ready finding per
    tool (with that tool in `affected_tools`) instead of a single vague probe-level
    blob; tool-less probes still aggregate at the probe level. Sorted for
    determinism.
    """
    by_key: dict[tuple[str, str], list[dict]] = {}
    for entry in entries:
        probe = str(entry.get("probe_classname", "")).strip()
        if not probe or not _is_hit(entry, hit_threshold):
            continue
        by_key.setdefault((probe, _attempt_tool(entry)), []).append(entry)

    findings: list[Finding] = []
    for probe, tool in sorted(by_key):
        group = by_key[(probe, tool)]
        tags = list(resolve_tags(probe) or ())
        # Suggest candidate guardrail classes from BOTH the OWASP tags and garak's
        # payload taxonomy: agentic probes carry `payload:agentic:*` but unreliable
        # OWASP numbers, so the OWASP join alone misses them (see catalog.py).
        related = sorted(
            {cls for tag in tags for cls in classes_for_owasp(tag)}
            | {cls for tag in tags for cls in classes_for_payload(tag)}
        )
        first = group[0]
        goal = str(first.get("goal", "")).strip()

        evidence = [f"garak probe: {probe}"]
        if tool:
            evidence.append(f"target tool: {tool}")
        evidence.append(f"hits: {len(group)}")
        evidence.append(f"tags: {', '.join(tags) or '(none resolved)'}")
        if goal:
            evidence.append(f"goal: {goal}")
        if related:
            evidence.append(
                "guardrail classes matching this probe's tags (candidates for a rule): " + ", ".join(related)
            )
        vuln = _vulnerability_info(first)
        if vuln:
            evidence.append(f"vulnerability: {_snippet(vuln, 400)}")
        if first.get("prompt") is not None:
            evidence.append(f"sample prompt: {_snippet(first.get('prompt'))}")
        outputs = first.get("outputs") or []
        if outputs:
            evidence.append(f"sample output: {_snippet(outputs[0])}")
        detectors = sorted((first.get("detector_results") or {}).keys())
        if detectors:
            evidence.append(f"detectors: {', '.join(detectors)}")

        slug_parts = [target, probe] + ([tool] if tool else [])
        title = f"garak probe {probe} landed against {target}"
        if tool:
            title += f" via {tool}"
        if goal:
            title += f" — {goal}"

        findings.append(
            Finding(
                id="garak-" + "-".join(_slug(p) for p in slug_parts),
                title=title,
                source=f"garak-audit:{target}",
                # a hitlog names the agent's tool, not a registry tool; the human
                # confirms the mapping at triage, so this stays uncatalogued.
                attack_class=GARAK_HIT_CLASS,
                affected_tools=((tool,) if tool else ()),
                suggested_params={},
                evidence="\n".join(evidence),
            )
        )
    return findings

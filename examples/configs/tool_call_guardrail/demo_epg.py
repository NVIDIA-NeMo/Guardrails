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

"""Narrated end-to-end demo for EPG Demo Day — the agent tool-call guardrail.

Stitches the four standalone offline proofs into one story, in narrative order:

    Act 1  The gap          — an unguarded agent executes every dangerous call it decides to make
    Act 2  The guardrail     — the same calls authorized first; attacks blocked, legit work flows
                               (plus a ground-truth correctness proof, no model in the loop)
    Act 3  Keeping it current — a field scanner reads attack papers; a human approves; rules merge in
    Act 4  The next layer     — a novel technique the per-call catalog can't express, caught by the
                               session-aware egress backstop

Standard library only: no LLM, no network, no Guardrails install. Deterministic,
so it runs identically on stage every time. The same run can emit a structured
JSON event log (`--emit-trace`) that a dashboard/web UI can replay.

    python demo_epg.py                          # the whole story
    python demo_epg.py --act 2 --act 4          # selected acts (rehearsal)
    python demo_epg.py --pause                  # wait for Enter between acts (presenter pacing)
    python demo_epg.py --emit-trace trace.json  # also write the replayable event log
    python demo_epg.py --no-color               # plain text (for capture/piping)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo import CASES  # noqa: E402  (ground-truth correctness table)
from demo_guarded_vs_unguarded import SCENARIOS  # noqa: E402  (attack + legit tool calls)
from egress import EgressLimits, EgressMonitor, authorize_with_egress  # noqa: E402
from example_policies import (  # noqa: E402
    HARDENED_GUARD,
    PRINCIPAL_ATTRS,
    PRINCIPALS,
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    VULNERABLE_GUARD,
)
from policy import Principal, ToolCall, ToolCallGuard  # noqa: E402
from scanner.scan import KeywordExtractor, ScanContext, scan  # noqa: E402
from synthesis.catalog import CLASS_DESCRIPTIONS, CLASS_REQUIRED_PARAMS, CLASS_TO_FACTORY  # noqa: E402
from synthesis.proposals import (  # noqa: E402
    cluster_uncatalogued,
    dropped_findings,
    find_gaps,
    format_factory_prompt,
    synthesize,
)
from synthesis.review import apply, load_approved, write_review_queue  # noqa: E402
from tools import TOOLS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DOCS = os.path.join(HERE, "scanner", "sample_docs")


# --- presentation helpers --------------------------------------------------
class Style:
    """Minimal ANSI styling, a no-op when color is disabled."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _c(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.enabled else s

    def bold(self, s: str) -> str:
        return self._c("1", s)

    def dim(self, s: str) -> str:
        return self._c("2", s)

    def red(self, s: str) -> str:
        return self._c("31", s)

    def green(self, s: str) -> str:
        return self._c("32", s)

    def yellow(self, s: str) -> str:
        return self._c("33", s)

    def cyan(self, s: str) -> str:
        return self._c("36", s)


def banner(style: Style, n: int, title: str, subtitle: str) -> None:
    bar = "━" * 74
    print()
    print(style.cyan("┏" + bar))
    print(style.cyan("┃ ") + style.bold(f"ACT {n} · {title}"))
    print(style.cyan("┃ ") + style.dim(subtitle))
    print(style.cyan("┗" + bar))


def fmt_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 40:
            parts.append(f"{k}=<{len(v)}B>")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)


class Trace:
    """Accumulates a replayable JSON event log across the acts that run."""

    def __init__(self):
        self.acts: list[dict] = []

    def add_act(self, act_id: str, n: int, title: str, subtitle: str, kind: str) -> dict:
        act = {"id": act_id, "act": n, "title": title, "subtitle": subtitle, "kind": kind, "events": []}
        self.acts.append(act)
        return act

    def dump(self, path: str) -> None:
        payload = {
            "demo": "epg-tool-call-guardrail",
            "generated_by": os.path.basename(__file__),
            "acts": self.acts,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


# --- Act 1: the gap --------------------------------------------------------
def act_gap(style: Style, trace: Trace) -> None:
    act = trace.add_act(
        "act1",
        1,
        "The gap",
        "An autonomous agent with no authorization layer: every tool call it decides to make, executes.",
        "attack_feed",
    )
    banner(style, 1, "The gap", "No policy between the agent's decision and the tool's execution.")
    attacks = [(label, pid, call) for (label, pid, call) in SCENARIOS if not label.startswith("legitimate")]
    for label, pid, call in attacks:
        result = TOOLS[call.tool](**call.args)
        print(f"\n  {style.red('RAN')}  {pid}: {call.tool}({fmt_args(call.args)})")
        print(f"       {style.dim('→ ' + result)}  {style.dim('· ' + label)}")
        act["events"].append(
            {"label": label, "principal": pid, "tool": call.tool, "args": call.args, "executed": True, "result": result}
        )
    print(
        "\n  "
        + style.red(f"{len(attacks)}/{len(attacks)} dangerous calls executed")
        + style.dim(" — the tools are mocks here; in production each is real damage.")
    )


# --- Act 2: the guardrail --------------------------------------------------
def act_guardrail(style: Style, trace: Trace) -> None:
    act = trace.add_act(
        "act2",
        2,
        "The guardrail",
        "The same calls, authorized against the policy guard before dispatch. Attacks blocked; legit work flows.",
        "guarded_feed",
    )
    banner(style, 2, "The guardrail", "Authorize the (tool, args, principal) triple; dispatch only if allowed.")
    for label, pid, call in SCENARIOS:
        principal = PRINCIPALS.get(pid, Principal(pid))
        decision = HARDENED_GUARD.authorize(call, principal)
        if decision.allowed:
            result = TOOLS[call.tool](**call.args)
            print(f"\n  {style.cyan('ALLOWED')}  {pid}: {call.tool}({fmt_args(call.args)})")
            print(f"           {style.dim('→ ' + result)}  {style.dim('· ' + label)}")
            act["events"].append(
                {
                    "label": label,
                    "principal": pid,
                    "tool": call.tool,
                    "args": call.args,
                    "allowed": True,
                    "result": result,
                }
            )
        else:
            print(f"\n  {style.green('BLOCKED')}  {pid}: {call.tool}({fmt_args(call.args)})")
            print(f"           {style.dim('→ ' + decision.reason)}  {style.dim('· ' + label)}")
            act["events"].append(
                {
                    "label": label,
                    "principal": pid,
                    "tool": call.tool,
                    "args": call.args,
                    "allowed": False,
                    "reason": decision.reason,
                }
            )

    # Correctness proof: the policy engine against a ground-truth table, no model.
    print("\n  " + style.bold("Proof — policy engine vs. a ground-truth table (deterministic, no model):"))
    proof = []
    mismatches = 0
    for pid, call, truth, note in CASES:
        principal = PRINCIPALS.get(pid, Principal(pid))
        d = VULNERABLE_GUARD.authorize(call, principal)
        ok = d.allowed == truth
        mismatches += not ok
        mark = style.green("ok") if ok else style.red("MISMATCH")
        verdict = "ALLOW" if d.allowed else "BLOCK"
        print(f"      [{mark}] {verdict:>5} (truth {'allow' if truth else 'block'})  {call.tool} — {note}")
        proof.append(
            {
                "principal": pid,
                "tool": call.tool,
                "args": call.args,
                "verdict": "allow" if d.allowed else "block",
                "truth": "allow" if truth else "block",
                "match": ok,
                "reason": d.reason,
            }
        )
    tally = f"{len(CASES) - mismatches}/{len(CASES)} match ground truth"
    print("  " + (style.green(tally) if mismatches == 0 else style.red(tally)))
    act["proof"] = {"cases": proof, "matched": len(CASES) - mismatches, "total": len(CASES)}


# --- Act 3: keeping it current ---------------------------------------------
def act_pipeline(style: Style, trace: Trace) -> None:
    act = trace.add_act(
        "act3",
        3,
        "Keeping it current",
        "Where Act 2's rules came from: a field scanner reads attack papers, a human approves, rules merge in.",
        "pipeline",
    )
    banner(style, 3, "Keeping it current", "documents → findings → candidates → human gate → applied rules")

    scan_ctx = ScanContext(
        docs_dir=SAMPLE_DOCS,
        tool_registry=dict(TOOL_REGISTRY),
        taxonomy=tuple(CLASS_TO_FACTORY),
        class_definitions=dict(CLASS_DESCRIPTIONS),
        class_params=dict(CLASS_REQUIRED_PARAMS),
        tool_schemas=dict(TOOL_SCHEMAS),
        principal_attrs=tuple(PRINCIPAL_ATTRS),
    )
    findings = scan(scan_ctx, KeywordExtractor())
    print("\n  " + style.bold("1. Findings from the scanner over sample attack docs"))
    for f in findings:
        print(f"     - [{f.attack_class}] {f.title}")
    act["events"].append(
        {
            "stage": "findings",
            "items": [{"attack_class": f.attack_class, "id": f.id, "title": f.title} for f in findings],
        }
    )

    gaps = find_gaps(VULNERABLE_GUARD, TOOL_REGISTRY)
    act["events"].append({"stage": "gaps", "items": [{"tool": g.tool, "missing": g.missing} for g in gaps]})

    candidates = synthesize(findings)
    uncatalogued = dropped_findings(findings)
    print("\n  " + style.bold("2. Candidates synthesized (unknown classes drop out, fail-closed)"))
    for c in candidates:
        print(f"     - {c.tool} ← {c.factory_key}({c.params})")
    for d in uncatalogued:
        print(
            f"     - {style.yellow('UNCATALOGUED')} [{d.attack_class}] {d.title} → queued for human triage, never auto-applied"
        )
    act["events"].append(
        {
            "stage": "candidates",
            "items": [
                {"tool": c.tool, "factory_key": c.factory_key, "params": c.params, "finding_id": c.finding_id}
                for c in candidates
            ],
            "uncatalogued": [{"id": d.id, "attack_class": d.attack_class, "title": d.title} for d in uncatalogued],
        }
    )

    queue_path = os.path.join(tempfile.mkdtemp(prefix="epg_bridge_"), "review.json")
    write_review_queue(candidates, gaps, queue_path, uncatalogued=uncatalogued)
    print("\n  " + style.bold("3. Review queue written for the human gate"))
    print(f"     {style.dim(queue_path)}")
    print(
        f"     {style.dim('every candidate starts approved=false; ' + str(len(uncatalogued)) + ' uncatalogued finding(s) recorded')}"
    )
    act["events"].append(
        {
            "stage": "review_queue",
            "path": queue_path,
            "candidate_count": len(candidates),
            "uncatalogued_count": len(uncatalogued),
        }
    )

    # Human gate (simulated): a person edits the queue and flips the rows they trust.
    with open(queue_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    for entry in payload["candidates"]:
        entry["approved"] = True
    with open(queue_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    approved = load_approved(queue_path)
    print("\n  " + style.bold("4. Reviewer approves; rules apply to the guard"))
    act["events"].append(
        {
            "stage": "approved",
            "items": [{"tool": c.tool, "factory_key": c.factory_key, "params": c.params} for c in approved],
        }
    )

    new_guard = ToolCallGuard(apply(approved, VULNERABLE_GUARD))
    checks = [
        (
            "dev-bob",
            ToolCall("git_push", {"remote": "origin", "branch": "main"}),
            "push to a remote dev-bob does not own",
        ),
        (
            "dev-alice",
            ToolCall("run_shell", {"command": "sleep 600", "timeout_seconds": 600}),
            "shell under the old 3600s ceiling",
        ),
        (
            "dev-alice",
            ToolCall("install_package", {"name": "leftpad-evil", "version": "1.0"}),
            "install a now-denylisted package",
        ),
        (
            "dev-alice",
            ToolCall("write_file", {"path": "src/app.py", "content": "..."}),
            "write_file (newly-policied tool)",
        ),
    ]
    print("\n  " + style.bold("5. Before vs. after the new rules"))
    before_after = []
    for pid, call, note in checks:
        principal = PRINCIPALS.get(pid, Principal(pid))
        before = VULNERABLE_GUARD.authorize(call, principal)
        after = new_guard.authorize(call, principal)
        b = style.cyan("ALLOW") if before.allowed else style.green("BLOCK")
        a = style.cyan("ALLOW") if after.allowed else style.green("BLOCK")
        print(f"     - {note}")
        print(f"         before: {b} {style.dim('· ' + before.reason)}")
        print(f"         after:  {a} {style.dim('· ' + after.reason)}")
        before_after.append(
            {
                "note": note,
                "before": {"allowed": before.allowed, "reason": before.reason},
                "after": {"allowed": after.allowed, "reason": after.reason},
            }
        )
    act["events"].append({"stage": "before_after", "items": before_after})

    # The bridge to Act 4: the uncatalogued finding is real pressure for a new layer.
    print("\n  " + style.yellow("↪ The uncatalogued finding has nowhere to go in a per-call policy:"))
    print(style.dim("    " + format_factory_prompt(cluster_uncatalogued(findings)).replace("\n", "\n    ")))


# --- Act 4: the next layer (session-aware egress) --------------------------
def _req(host: str, **extra) -> ToolCall:
    return ToolCall("http_request", {"url": f"https://{host}/", **extra})


def _gauges(mon: EgressMonitor, session: str, principal: Principal) -> dict:
    state = mon._sessions.get((session, principal.id))
    if state is None:
        return {"requests": 0, "distinct_hosts": 0, "cumulative_bytes": 0}
    return {"requests": state.count, "distinct_hosts": len(state.hosts), "cumulative_bytes": state.cumulative_bytes}


def _egress_scenario(style, act, name, limit, mon, session, principal, steps, clock=None, times=None):
    print(f"\n  {style.bold(name)} {style.dim('(' + limit + ')')}")
    ev = {"scenario": name, "limit": limit, "steps": []}
    for idx, (call, note) in enumerate(steps):
        if clock is not None and times is not None:
            clock.t = float(times[idx])
        decision = authorize_with_egress(HARDENED_GUARD, mon, session, call, principal)
        flag = style.cyan("ALLOW") if decision.allowed else style.green("BLOCK")
        print(f"    {flag}  http_request({fmt_args(call.args)})  {style.dim('· ' + note)}")
        print(f"          {style.dim(decision.reason)}")
        ev["steps"].append(
            {
                "args": call.args,
                "note": note,
                "allowed": decision.allowed,
                "reason": decision.reason,
                "gauges": _gauges(mon, session, principal),
            }
        )
    act["events"].append(ev)


def act_egress(style: Style, trace: Trace) -> None:
    act = trace.add_act(
        "act4",
        4,
        "The next layer",
        "A novel technique the per-call catalog can't express: any one call looks fine; the session is the tell.",
        "egress",
    )
    banner(style, 4, "The next layer", "Session-aware egress backstop — watches the sequence, not the single call.")
    alice = PRINCIPALS["dev-alice"]

    mon = EgressMonitor(
        EgressLimits(max_distinct_hosts=3, max_requests=100, max_cumulative_bytes=10**9, max_requests_per_window=100)
    )
    _egress_scenario(
        style,
        act,
        "Distinct-host fan-out",
        "max 3 distinct hosts / session",
        mon,
        "sess-1",
        alice,
        [
            (_req("a.example.com"), "host #1"),
            (_req("b.example.com"), "host #2"),
            (_req("c.example.com"), "host #3"),
            (_req("d.example.com"), "host #4 → blocked"),
            (_req("a.example.com"), "already-seen host → still allowed"),
        ],
    )

    mon = EgressMonitor(
        EgressLimits(max_cumulative_bytes=5000, max_requests=100, max_distinct_hosts=100, max_requests_per_window=100)
    )
    _egress_scenario(
        style,
        act,
        "Cumulative outbound volume",
        "max 5000B / session",
        mon,
        "sess-2",
        alice,
        [
            (_req("sink.example.com", body="x" * 2000), "+2000B (total 2000)"),
            (_req("sink.example.com", body="x" * 2000), "+2000B (total 4000)"),
            (_req("sink.example.com", body="x" * 2000), "+2000B (total 6000) → blocked"),
        ],
    )

    _egress_scenario(
        style,
        act,
        "Layer ordering (defense in depth)",
        "per-call guard runs before the monitor",
        EgressMonitor(),
        "sess-3",
        alice,
        [
            (_req("169.254.169.254"), "cloud metadata → blocked by the per-call guard, monitor never consulted"),
            (_req("api.example.com"), "external host → guard allows, monitor records"),
        ],
    )

    class _Clock:
        t = 0.0

        def __call__(self):
            return self.t

    clk = _Clock()
    mon = EgressMonitor(
        EgressLimits(
            max_requests_per_window=3,
            window_seconds=10,
            max_requests=100,
            max_distinct_hosts=100,
            max_cumulative_bytes=10**9,
        ),
        clock=clk,
    )
    _egress_scenario(
        style,
        act,
        "Burst rate",
        "max 3 requests / 10s",
        mon,
        "sess-4",
        alice,
        [
            (_req("h0.example.com"), "t=0"),
            (_req("h1.example.com"), "t=1"),
            (_req("h2.example.com"), "t=2"),
            (_req("h3.example.com"), "t=3 → blocked (4 in window)"),
            (_req("h4.example.com"), "t=20 → window cleared, allowed"),
        ],
        clock=clk,
        times=[0, 1, 2, 3, 20],
    )


ACTS = [(1, act_gap), (2, act_guardrail), (3, act_pipeline), (4, act_egress)]


def _intro(style: Style) -> None:
    print(style.bold("NeMo Guardrails · Agent Tool-Call Authorization") + style.dim("  —  EPG Demo Day"))
    print(
        style.dim("legend: ")
        + style.red("RAN")
        + style.dim(" executed unguarded · ")
        + style.green("BLOCK")
        + style.dim(" guardrail acted · ")
        + style.cyan("ALLOW")
        + style.dim(" allowed through")
    )


def _outro(style: Style) -> None:
    print("\n" + style.cyan("━" * 75))
    print(
        style.bold("Takeaway: ")
        + "authorize the action, not just the text — and keep the policy current from the field, with a human in the loop."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Narrated EPG Demo Day demo for the agent tool-call guardrail.")
    parser.add_argument("--emit-trace", metavar="PATH", help="write a replayable JSON event log to PATH")
    parser.add_argument(
        "--act",
        type=int,
        action="append",
        choices=[1, 2, 3, 4],
        dest="acts",
        help="run only the given act(s); repeatable",
    )
    parser.add_argument("--pause", action="store_true", help="wait for Enter between acts (presenter pacing)")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color")
    args = parser.parse_args(argv)

    style = Style(enabled=(not args.no_color) and sys.stdout.isatty())
    trace = Trace()
    selected = args.acts or [1, 2, 3, 4]
    runlist = [(n, fn) for (n, fn) in ACTS if n in selected]

    _intro(style)
    for i, (_n, fn) in enumerate(runlist):
        fn(style, trace)
        if args.pause and i < len(runlist) - 1 and sys.stdin.isatty():
            input(style.dim("\n  [Enter ▸] "))
    _outro(style)

    if args.emit_trace:
        trace.dump(args.emit_trace)
        print(style.dim(f"\n  trace written → {args.emit_trace} ({len(trace.acts)} act(s))"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

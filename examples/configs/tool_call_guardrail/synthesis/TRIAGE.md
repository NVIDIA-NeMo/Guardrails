# Uncatalogued-finding triage log

When the scanner surfaces a technique the catalog cannot express, the finding is
recorded under `uncatalogued` in the review queue (never auto-applied) and shows
up in the `format_factory_prompt` report as *"tool X: N uncatalogued finding(s)
— consider a new rule factory."* See the README's *scanner → synthesis* section
for the full path.

This file is the human side of that loop: a log of decisions about those
findings, so a recurring signal is not re-litigated on every scan. **Before
designing a new factory for a tool that appears under uncatalogued pressure,
check whether there is already an entry here.**

A decision is one of:

- **catalogue** — worth a new rule factory; once built, record the class it maps to.
- **defer to another layer** — real, but not a per-call authorization concern;
  record where it belongs instead.
- **out of scope** — not a threat this guardrail should address.

---

## context-exfiltration (tool-output / context exfiltration)

- **Finding id:** `context-exfiltration-novel-novel`
- **Source:** https://example.org/research/2026-context-exfiltration
- **Attack class:** `novel` · **Affected tool:** `http_request`
- **Decision:** **defer to another layer** (egress / output rail)
- **Date:** 2026-06-29 · **Reviewer:** Sven Chilton

### Technique

An agent is coaxed into relaying earlier tool outputs and accumulated
conversation context to an attacker endpoint over a series of `http_request`
calls. Each individual request is benign; the aggregate effect leaks sensitive
context over time.

### Why it is not catalogued as a rule factory

This guardrail authorizes a single proposed call: a `Rule` is
`Callable[[ToolCall, Principal], Optional[str]]` and `authorize()` evaluates it
against one `(tool, args, principal)` triple. The exfiltration signal is not
present in any single call — it is a property of the *sequence* of calls and the
*sensitivity of the accumulated state*. No argument-level control (URL denylist,
pattern, ownership, bound) can see it, so the `novel` → uncatalogued path is
behaving correctly here: it flagged a technique that belongs elsewhere.

### Where it belongs instead

The session-aware layer that already sees the whole conversation:

- A **NeMo Guardrails output/egress rail** (or the orchestrator) that inspects
  outbound tool calls across the dialogue and can reason over sequence and the
  sensitivity of what is being sent.
- Data-provenance / taint tracking (sensitive read → external send) if that
  capability exists upstream — the signal the source paper actually points at,
  and a different discipline from per-call authorization.

### Optional add-on (not planned)

A deliberately-scoped **stateful-egress** rule *type* could give the guard a
coarse heuristic backstop — e.g. outbound-request rate, cumulative outbound
volume, or distinct-external-host cardinality per principal/session. This is a
rate-limiting proxy, **not** exfiltration detection: a patient attacker stays
under any static threshold, and a legitimate data-sync agent trips it. It also
requires breaking the stateless rule contract — a new signature carrying call
history (e.g. `(ToolCall, Principal, CallHistory) -> Optional[str]`), per-session
state with a defined lifecycle, and threading that state through `authorize()` —
which dilutes the thin, portable, side-effect-free core that is this guardrail's
main value. Pursue only if uncatalogued pressure on `http_request` grows enough
to justify the contract change; record it here if so.

### Revisit when

`cluster_uncatalogued` shows `http_request` (or another egress tool) accruing
several distinct uncatalogued findings — evidence the coarse backstop is worth
the cost.

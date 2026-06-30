# EPG Demo Day — talk track & slide beats

**Date:** Fri, July 10, 2026 · **Demo:** agent tool-call authorization guardrail
**One command on stage:** `python demo_epg.py --pause`
**Fallback if anything is off:** drop `--pause`, or run a single act with `--act N`. The demo is stdlib-only and deterministic — no LLM, no network, no install — so it cannot flake on connectivity.

> The story in one line: **guard the _action_ an agent takes, not just the text it emits — and keep the policy current from the field, with a human in the loop.**

---

## Run sheet

| Beat | Command | ~time |
| --- | --- | --- |
| Open (slide) | — | 0:30 |
| Act 1 — the gap | starts here (Enter advances) | 0:45 |
| Act 2 — the guardrail | continues on Enter | 1:00 |
| Act 3 — keeping it current | continues on Enter | 1:15 |
| Act 4 — the next layer | continues on Enter | 1:00 |
| Close (slide) | — | 0:30 |

Total ≈ 5 min. For a 3-min slot, cut Act 3's before/after detail and the Act 2 ground-truth proof (keep the 7/7 tally line).

On stage it's a single `python demo_epg.py --pause` run — press Enter to advance between acts. The per-act `--act N` commands are only for rehearsing one act in isolation.

Pre-generate the JSON the dashboard would replay (if you show the web app later):
`python demo_epg.py --emit-trace trace.json`

---

## Opening slide (before the terminal)

**Title:** *Agentic guardrails: authorize the action, not the text.*

**Say:** "Today's guardrails mostly filter the words going into and out of a model. But an agent does damage through the *actions* it takes — the tools it calls. This is a guardrail that sits between an agent's *decision* to call a tool and the tool's *execution*, and authorizes the `(tool, arguments, principal)` triple first. And the hard part isn't writing the first rule — it's keeping the rules current as attackers get more creative. Hold onto that question, because it's where this ends up. Everything you're about to see is real, working code — the tools are deliberately mocked so I can show you a metadata-exfil attack without actually running one."

*(Two jobs for the open: plant "who keeps the rules current?" — the close pays it off with the open contribution flywheel (anyone mines the literature, contributes guardrails back) — and land the credibility pre-empt (real code, mocked tools) early.)*

---

## Act 1 — The gap

**On screen:** seven attacks, every one prints `RAN`.

**Say:** "Here's an autonomous agent with no authorization layer. Path traversal, a runaway shell, a push to a repo it doesn't own, a malicious package, an SSRF at the cloud-metadata endpoint, writes outside its workspace. Every call it *decided* to make, it *executed*. Seven for seven. The tools are stand-ins here — but in production, each of those is real damage."

**Beat to hit:** the agent's intent is never the problem; the missing gate between intent and execution is.

---

## Act 2 — The guardrail

**On screen:** the same seven calls now `BLOCKED`, two legitimate calls `ALLOWED`, then a `7/7 match ground truth` proof.

**Say:** "Same calls, now each one is authorized against a policy before dispatch. The seven attacks are blocked — each with a specific reason, not a generic 'denied'. And the two legitimate calls still go through: this isn't lockdown, it's authorization. It's default-deny: a tool with no policy is refused. And the engine is provably correct — here it is matched against a ground-truth table, no model in the loop, deterministic."

**Beat to hit:** specific reasons + legit work still flows + provable. This is policy, not a kill switch.

---

## Act 3 — Keeping it current

**On screen:** scanner findings → synthesized candidates → review queue → human approves → before/after.

**Say:** "Where did those rules come from? A field-scanning agent reads papers and advisories about new agent-exploitation techniques and proposes rules. But it's untrusted: a finding only carries an attack *class* and parameters — never code — and it becomes an enforceable rule only by passing a fixed catalog of vetted rule factories *and* a human approval gate. Watch the before/after: rules the scanner surfaced, a human approved, now blocking calls that sailed through a minute ago. And critically — anything the catalog can't yet express does **not** get auto-applied; it's flagged for a human."

**Beat to hit:** one-directional trust — unprivileged producer, human gate, fail-closed. This is how it stays current *safely*.

---

## Act 4 — The next layer

**On screen:** the `novel` finding from Act 3 → egress monitor catching fan-out, volume, and burst.

**Say:** "Remember that one finding the catalog *couldn't* express — context exfiltration. That's the honest part of this story: a per-call policy can't see it, because every single request looks fine. The *session* is the tell. So that finding motivated a new layer — a session-aware egress monitor that watches the sequence: too many distinct destinations, too much cumulative outbound volume, too high a burst rate. And the per-call guard still runs first — it blocks the metadata endpoint before the monitor is ever consulted. Defense in depth."

**Beat to hit:** the system found its *own* blind spot and grew a layer to cover it. That's the most memorable beat — let it breathe.

---

## Closing slide

**Say:** "Two ideas. One: guard the action, not just the text. Two: keep the policy current from the field, with a human in the loop — and when a technique outgrows per-call rules, add a layer rather than pretend the old one covers it."

**Ask / next step:** "My ask: this shouldn't live on my branch. It's an open example in the Guardrails repo — anyone can point it at the latest AI-safety research, have it generate guardrails for the attacks it recognizes, and open a PR to harden the shared example for everyone. Every rule is human-reviewed twice: once at the tool's approval gate, once at PR review. The flywheel I want is simple — the field's newest attacks become everyone's defaults. Try it on your own agents — and when it surfaces something, open an issue or a PR against the example. Want a hand landing your first one? Grab me after."

> *Delivery note:* lead into it from Act 3 — "you just watched that pipeline propose human-approved rules; this is the same loop, open to anyone." The strength of this ask is **open + human-reviewed**: it's a community flywheel, not your inbox. Route findings to the *repo* (issue/PR), and offer yourself as the on-ramp — the guide, not the gatekeeper.

---

## Anticipated Q&A

- **"Is any of this real or is it a mockup?"** — All real code; run `demo.py` for the deterministic correctness proof. The *tools* are mocked so attacks are safe to show; the policy engine, scanner, synthesis, and egress monitor are the actual implementation.
- **"What's the perf cost per call?"** — Per-call authorization is pure-function rule evaluation, no model call. The egress monitor is in-memory counters; a real deployment backs it with a shared store keyed by conversation id.
- **"Does the scanner auto-update the live policy?"** — No, by design. Nothing applies without a human flipping `approved`, and a finding can only ever instantiate a *pre-vetted* rule factory.
- **"How does this relate to sandboxing / NemoClaw?"** — Complementary. This is *prevention* (the action never runs); a sandbox is *containment* (the action runs, isolated). Different layers; you'd want both.
- **"LLM in the loop?"** — The production scanner uses an LLM extractor to read prose; this demo uses a deterministic keyword extractor so it runs offline. The guardrail itself needs no model.
- **"Is this locked to NeMo Guardrails?"** — It's Guardrails-native today, and the authorization core has no Guardrails dependency by design, so it can generalize later. Guardrails is where it should prove out first — and as an open example, anyone can extend it there.

---

## If you take it to a web app later (deferred)

The `--emit-trace` JSON is the contract. A dashboard replays it: Act 1/2 as a red/green attack feed, Act 3 as the pipeline with approve toggles, Act 4 as live gauges (the trace already carries per-step `gauges`: requests / distinct_hosts / cumulative_bytes). Build the UI to **replay the trace**, not call Python live — keeps it visual, honest, and stage-safe, with this CLI as the guaranteed fallback. Build fresh rather than remixing; the data model is your own.

# EPG Demo Day — talk track & slide beats

**Date:** Fri, July 10, 2026 · **Demo:** *From Research and Red-Teaming to Tool-Call Guardrails* (agent tool-call authorization)
**Primary artifact (most of the video):** the `tool_call_guardrail_walkthrough.ipynb` notebook — Kernel → **Restart & Run All**, then step down through the cells (colored verdict tables + egress bar-gauges). Pre-run it once before recording so every output is populated and nothing recomputes on camera.
**Deterministic fallback:** `python demo_epg.py --pause` — the same four acts as narrated terminal output, stdlib-only (no LLM, no network, no install), so it cannot flake. Use it if the kernel misbehaves on the day.

> The story in one line: **guard the _action_ an agent takes, not just the text it emits — and keep the policy current from the field, with a human in the loop.**

---

## Run sheet

| Beat | On screen (notebook) | ~time |
| --- | --- | --- |
| Open (slide) | — | 0:30 |
| Act 1 — the gap | the Act 1 `RAN` table | 0:40 |
| Act 2 — the guardrail | the guarded table + the full `7/7` proof table | 0:50 |
| Act 3 — keeping it current (two sources + how it's built) | "how it's built" diagram → Source A/B tables → candidates + triage → before/after | 1:35 |
| Act 4 — the next layer | the four egress gauge tables | 1:00 |
| **Assistant clip** (pre-recorded, ~25s) | how you'd actually invoke it via a coding assistant | 0:25 |
| Close (slide) | — | 0:30 |

Total ≈ 5:20. You set the pace by how long you dwell on each cell — the notebook carries the **full** content (row-by-row proof, all four egress scenarios), so to hit a hard 5:00 just **scroll past** the Act 2 proof table and the 4th egress scenario (burst) instead of narrating them. For a 3-min slot, also cut Act 3's before/after and drop the assistant clip.

Record the notebook top-to-bottom after a **Restart & Run All** (so nothing recomputes on camera). The assistant clip is a separate pre-recorded take (see its section below) — splice it in right before the Close.

Fallback / future web app: `python demo_epg.py --emit-trace trace.json` still emits the replayable JSON contract for a dashboard, and `demo_epg.py --pause` is the stage-safe fallback if the kernel misbehaves.

---

## Opening slide (before the notebook)

**Slide:** the title slide — rendered as `epg_demo_slides.pptx` (slide 1; the pptx is the source of truth for what's on screen). Title: **Authorize the Action, Not Just the Text**. Subtitle: *Research and red-teaming → tool-call guardrails*.

**Say (~30s):** "Today's guardrails filter the words going in and out of a model. However, an agent does damage through its *actions* — the tools it calls. The Guardrail demoed here sits between the agent's *decision* to call a tool and the tool's *execution*, authorizing the `(tool, arguments, principal)` triple. This demo also illustrates how humans and automation collaborate to keep the guardrail rules current as attackers get more creative. Everything here is real code; the tools are mocked, so the metadata-exfil attack is safe to show."

*(Two jobs for the open: state the thesis — humans + automation keep the rules current — which the close then extends into the open-contribution flywheel (anyone mines the literature/audits, contributes guardrails back); and land the credibility pre-empt (real code, mocked tools) early.)*

---

## Act 1 — The gap

**On screen:** the Act 1 table — seven attack rows, every **verdict** cell reads `RAN` (red).

**Say:** "Here's an autonomous agent with no authorization layer. A simulated attacker talks it into seven dangerous tool calls — a path-traversal file read, a runaway shell command, a push to a repo it doesn't own, a malicious package install, an SSRF at the cloud-metadata endpoint, and writes outside its workspace. Every call it *decided* to make, it *executed* — seven for seven. The tools are stand-ins, but in production, abusing each one is real damage."

**Beat to hit:** the agent's intent is never the problem; the missing gate between intent and execution is.

---

## Act 2 — The guardrail

**On screen:** the guarded table — the same seven calls now `BLOCKED` (green) each with a specific reason, two legitimate calls `ALLOWED` (blue) — then the full row-by-row **`7/7 match ground truth`** proof table.

**Say:** "Same calls, now each one is authorized against a guardrail policy before dispatch. The seven attacks are blocked — each with a specific reason, not a generic 'denied'. And the two legitimate calls still go through: this isn't lockdown, it's authorization. It's default-deny: a tool with no policy is refused. And the engine is provably correct — here it is matched against a ground-truth table, no model in the loop, deterministic."

**Beat to hit:** specific reasons + legit work still flows + provable. This is policy, not a kill switch.

---

## Act 3 — Keeping it current  *(the heart of the talk — two sources, one pipeline)*

**On screen:** the "how it's built" markdown diagram → the **Source A** research-findings table → the **Source B** garak red-team table → the candidates + triage tables → the before/after table.

**Say (structure first — this is the "how it's built" beat):** "This is where the guard stays current, and I want to show you how it's *built*, not just what it does. **One pipeline, two producers, a human gate.** A literature scanner and a garak security audit both emit the **same `Finding` type** — the producer is pluggable, the trust boundary is one-way: evidence in, human-approved rules out, nothing auto-applies. And the policy core has no framework dependency."

**Say (the two sources — this is the title):** "**Source A, Research** — a field scanner reads papers and advisories and proposes findings. **Source B, Red-teaming** — a garak audit that *attacked* the agent and rediscovered the exact `run_shell` and SSRF exploits you saw in Act 1, from the outside. Both land as findings. Catalogued ones map to vetted rule factories; everything else — including every audit hit — is flagged for a human, never auto-applied. Watch the before/after: rules a human approved now block calls that sailed through a minute ago."

*(Framing note — not spoken, for live delivery/Q&A: "a field scanner reads papers and advisories" is two decoupled steps. **Acquire** (`scanner/acquire.py`) is what scrapes arXiv and RSS security feeds and writes each new paper/advisory into the corpus as a normalized `*.md` — a fetch + watermark, no LLM. **Extract** (`scanner/scan.py`) is where the LLM reads that prose into findings. So Source A intel is *scraped in by the scanner*, not dropped in from outside — the advisory in `./advisories` is the output of that scrape, which is why the assistant-clip prompt now opens "An arXiv scrape yielded a new advisory…". The demo just pre-places one and runs the deterministic keyword extractor so nothing hits the network on camera.)*

**Beat to hit:** two sources, one trust boundary — *that's the title*. Unprivileged producers, human gate, fail-closed. Research **and** red-teaming keep it current, safely.

---

## Act 4 — The next layer

**On screen:** four egress gauge tables — distinct-host fan-out, cumulative volume, layer-ordering, and burst rate — the green `Styler.bar` filling toward each session limit, and the call that would cross it blocked.

**Say:** "Remember that one finding the catalog *couldn't* express — context exfiltration. That's the honest part of this story: a per-call policy can't see it, because every single request looks fine. The *session* is the tell. So that finding motivated a new layer — a session-aware egress monitor that watches the sequence: too many distinct destinations, too much cumulative outbound volume, too high a burst rate. And the per-call guard still runs first — it blocks the metadata endpoint before the monitor is ever consulted. Defense in depth."

**Beat to hit:** the system found its *own* blind spot and grew a layer to cover it. That's the most memorable beat — let it breathe.

---

## Assistant clip — how you'd actually invoke it (~25s, pre-recorded)

*Splice this in right before the Close. Pre-record it so nothing runs live on camera; speed-ramp the middle and hold on the review queue. The notebook shows the pipeline step by step; this shows the real adoption path.*

**On screen (a coding assistant open in a repo):**
1. **(0–5s)** User types: *"An arXiv scrape yielded a new advisory on agent tool abuse in the `./advisories` directory. Scan it and propose tool-call guardrails for our agent; don't apply anything."*
2. **(5–13s, speed-ramp 2–4×)** The assistant runs the real scanner via the venv — `~/venv-guardrails-docs/bin/python scanner/scan.py --extractor llm --docs advisories/ --out findings.json`, then the same venv Python on `synthesize.py findings.json --out review_queue.json` — and replies: *"3 techniques found — 2 map to vetted rule factories, 1 is novel. Nothing applied; 3 proposals written to `review_queue.json`, all `approved: false`."* *(live LLM extractor at temp 0 via the NVIDIA gateway; `~/venv-guardrails-docs` is the only env with `openai`. ~10s for 3 calls — this is the speed-ramp stretch.)*
3. **(13–20s, hold longest)** Cut to `review_queue.json`: rows with `"approved": false`; the cursor flips one or two to `true`. **This is the shot that carries the trust story.**
4. **(20–25s)** The assistant applies the approved rules; a before/after line pops — *`git_push` to an unowned remote: ALLOW → **BLOCK***. Cut.

**Voiceover (one continuous line):** "In practice, you don't open a notebook — you ask your coding assistant. Point it at a fresh advisory, or a red-team audit, and it proposes tool-call guardrails. But it can't apply a thing on its own: every proposal lands in a review queue, off by default. You approve the ones you trust — and only then do they go live. Same human gate you just saw, now in your normal dev loop."

**Beat to hit:** the real adoption path is your dev loop, and it hits the *same* human gate — the notebook was just so we could watch it. Hand straight into the flywheel ask.

**Optics:** record with whatever assistant you'll actually show; consider a generic "your coding assistant" framing (or NemoClaw) so the star stays the guardrail system, not the tool — and make sure the review-queue/approve shot is unmistakable, or the clip reads as "AI auto-generates guardrails," which undercuts the whole trust story.

---

## Closing slide

**Slide:** the closing slide — rendered as `epg_demo_slides.pptx` (slide 2). Title: **Two Ideas, One Ask** — the two ideas, the green "one ask" flywheel line, and the *try it · open an issue/PR · grab me after* CTA. The **Say** below is the voiceover over it; the slide carries the detail so the read stays tight.

**Say (~30s):** "Two ideas to leave you with: guard the action, not just the text — and keep the policy current from both research and red-teaming, with a human in the loop. And one ask: this is an open example. Point it at the newest attacks, let it propose guardrails, PR them back — every rule human-reviewed twice. The field's newest attacks become everyone's defaults. Try it on your agents; grab me or ping me if you want a hand."

*(The recorded close is the ~30s read above; the slide's CTA carries the rest — issue/PR, "human-reviewed twice" = approval gate + PR review — so you can expand the ask live if there's time.)*

> *Delivery note:* lead into it straight from the assistant clip — "you just saw that same human gate in a normal dev loop; this is the same loop, open to anyone." The strength of this ask is **open + human-reviewed**: it's a community flywheel, not your inbox. Route findings to the *repo* (issue/PR), and offer yourself as the on-ramp — the guide, not the gatekeeper.

---

## Anticipated Q&A

- **"Is any of this real or is it a mockup?"** — All real code; the notebook runs the actual implementation top to bottom (run `demo.py` for the same deterministic correctness proof). The *tools* are mocked so attacks are safe to show; the policy engine, scanner, synthesis, and egress monitor are real.
- **"Why show a notebook if people would use a coding assistant?"** — The notebook is just so we can watch the pipeline step by step, deterministically. The clip shows the real path: you invoke it through your coding assistant in your normal dev loop — and it routes every proposal through the *same* human gate you saw in the notebook. Notebook = clarity; assistant = adoption.
- **"What's the perf cost per call?"** — Per-call authorization is pure-function rule evaluation, no model call. The egress monitor is in-memory counters; a real deployment backs it with a shared store keyed by conversation id.
- **"Does the scanner auto-update the live policy?"** — No, by design. Nothing applies without a human flipping `approved`, and a finding can only ever instantiate a *pre-vetted* rule factory.
- **"Where do the red-team (Source B) findings come from?"** — A garak run via NeMo Auditor. Its hitlog flows through a small adapter (`scanner/garak_hitlog.py`) into the *same* `Finding` type the literature scanner emits — one finding per (probe, tool). The demo bundles a captured report so it stays offline; the same human gate applies, so an audit hit is evidence, never an auto-rule.
- **"Where do the Source A (research) advisories come from — does the scanner scrape them?"**  — Yes. `scanner/acquire.py` pages arXiv (e.g. `cat:cs.CR` agent-attack queries) and RSS security feeds, normalizing each new document into the corpus with a watermark ledger so re-runs only fetch what's genuinely new. `scanner/scan.py` then runs the LLM extractor over that prose to produce findings. Two decoupled steps: acquisition is a fetch (no LLM); extraction is the LLM step. The demo skips the live scrape and pre-places one advisory in `./advisories` so it runs offline and deterministically.
- **"Why does the audit finding say OWASP llm07/llm08, not llm06?"** — That's a garak mistag we found and filed upstream (garak #1919); we join on the version-independent `payload:agentic:*` tag too, so the finding still surfaces the right candidate guardrail classes.
- **"How does this relate to sandboxing / NemoClaw?"** — Complementary. This is *prevention* (the action never runs); a sandbox is *containment* (the action runs, isolated). Different layers; you'd want both.
- **"LLM in the loop?"** — In the *extraction* step, yes — reading prose into findings is the judgement-heavy seam an LLM does. The **notebook (Acts 1–4)** runs the deterministic keyword extractor so it stays offline; the **assistant clip** runs the real LLM extractor live (gpt-4o-mini at temp 0, via the NVIDIA gateway) — same `Extractor` protocol, swapped backend. Either way the **guardrail itself needs no model**: enforcement is pure-function rule evaluation, and every LLM-proposed rule is clamped to the taxonomy and still has to clear the human review gate.
- **"Is this locked to NeMo Guardrails?"** — It's Guardrails-native today, and the authorization core has no Guardrails dependency by design, so it can generalize later. Guardrails is where it should prove out first — and as an open example, anyone can extend it there.

---

## If you take it to a web app later (deferred)

The `--emit-trace` JSON is the contract. A dashboard replays it: Act 1/2 as a red/green attack feed, Act 3 as the pipeline with approve toggles, Act 4 as live gauges (the trace already carries per-step `gauges`: requests / distinct_hosts / cumulative_bytes). Build the UI to **replay the trace**, not call Python live — keeps it visual, honest, and stage-safe, with this CLI as the guaranteed fallback. Build fresh rather than remixing; the data model is your own.

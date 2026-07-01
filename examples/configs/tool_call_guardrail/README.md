# Agent tool-call guardrail (proof of principle)

A guardrail that sits between an agent's *decision* to call a tool and the
tool's *execution*. Instead of filtering input/output text, it authorizes the
proposed **(tool, arguments, principal)** triple and only dispatches the call if
policy allows it — the gap most relevant to safeguarding agents, which can do
real damage through the actions they take, not just the text they emit.

This is an intentionally small proof of principle. It is **NeMo Guardrails-native**
today, but the authorization logic is deliberately quarantined in a
Guardrails-free core so it can be lifted out into a runtime-agnostic library
later (the stated end goal).

The example also includes a **field-scanning pipeline** (`scanner/`, `synthesis/`)
that turns documents about agent-exploitation techniques into *human-approved*
policies for this guard — see [Populating policies from the
field](#populating-policies-from-the-field-scanner--synthesis) below. The guard
works standalone; the pipeline is how its policies are kept current.

## Design: a thin runtime over a portable core

| Layer | Files | Depends on Guardrails? |
| --- | --- | --- |
| **Portable policy core** | `policy.py` | No |
| **Session-aware egress backstop** (optional, prototype — off by default) | `egress.py` | No |
| **Example config** (policies, principals, mock tools) | `example_policies.py`, `tools.py` | No |
| **Guardrails wiring** | `config.py`, `config.yml`, `flows.co` | Yes |
| **Offline demo** | `demo.py` | No |

`policy.py` defines `Principal`, `ToolCall`, `ToolPolicy`, and `ToolCallGuard`.
The guard is **default-deny**: a tool with no registered policy is refused. Each
policy names the roles allowed to call a tool and a list of pure-function
argument `Rule`s (e.g. "principal must own this git remote", "shell timeout must
not exceed a ceiling"). `config.py` only wraps this in a `safe_tool_call` action and
registers it; it contains no authorization logic of its own. Lifting `policy.py`
(plus the rule helpers) into a standalone package is the planned path to the
runtime-agnostic library — nothing in it imports Guardrails.

## Run the offline proof (no LLM, no install)

```bash
python demo.py
```

This exercises the policy core against a ground-truth set of allowed/blocked
tool-call attempts and prints each verdict plus a pass/fail tally. It needs only
the standard library, so the guardrail's correctness is checkable
deterministically, independent of any model.

## Run the Guardrails-native path (needs an install + LLM)

Requires `nemoguardrails` installed and a model the `config.yml` can reach (set
the relevant API key for your provider). The `safe_tool_call` action is what a
flow — or a tool-calling agent — invokes in place of a raw tool:

```python
from nemoguardrails import LLMRails, RailsConfig

rails = LLMRails(RailsConfig.from_path("examples/configs/tool_call_guardrail"))
print(rails.generate(messages=[{"role": "user", "content": "read the source file"}]))
```

`flows.co` contains canned flows (an allowed workspace read and write, a blocked
shell call, a blocked write outside the principal's workspace, and a blocked
metadata-egress request) so the wiring runs deterministically; in a real agent the
LLM supplies the tool name and arguments instead. The runtime serves
`HARDENED_GUARD` — `VULNERABLE_GUARD` with the human-approved output of the
scanner→synthesis pipeline folded in. `demo.py` and `demo_bridge.py` run against
`VULNERABLE_GUARD` to show the "before" gaps the scanner discovers.

### Optional: session-aware egress backstop

`safe_tool_call` always runs the stateless per-call guard above. It can
*additionally* route each call through a session-aware **egress monitor**
(`egress.py`) that watches the *sequence* of outbound calls within a conversation
and vetoes aggregate behavior no single-call rule can see — too many distinct
destination hosts, too much cumulative outbound volume, or too high a request
burst. The per-call guard runs first either way (a metadata-egress URL is blocked
before the monitor is ever consulted).

This layer is a **prototype and is off by default**. Enable it by setting
`TCG_EGRESS=1` in the environment before loading the config:

```bash
export TCG_EGRESS=1   # opt into the session-aware egress layer (default: off)
```

See `EGRESS_PROTOTYPE.md` for the design and `demo_egress.py` for an offline
demonstration of the four aggregate signals.

## Populating policies from the field: scanner → synthesis

A field-scanning agent reads about new agent-exploitation techniques; this
pipeline turns what it finds into vetted, **human-approved** policies for the
guard above. It is a one-directional trust flow — the producer is unprivileged,
and a finding becomes an enforceable rule only after passing a fixed catalog and
a human gate:

```
documents → Finding → RuleCandidate → review queue → applied ToolPolicy
            (untrusted) (vetted factory) (human gate)  (merged into the guard)
```

| Stage | Files | Depends on Guardrails? |
| --- | --- | --- |
| **Acquisition** (fetch a corpus) | `scanner/sources.py`, `scanner/acquire.py` | No |
| **Scanner** (untrusted producer) | `scanner/scan.py`, `scanner/llm_extractor.py` | No |
| **Synthesis** (the trust boundary) | `synthesis/catalog.py`, `synthesis/proposals.py`, `synthesis/findings.py` | No |
| **Human gate** | `synthesis/review.py` | No |
| **Offline pipeline demo** | `demo_bridge.py` | No |

The safety properties that make this sound:

- **No stage generates executable code.** A `Finding` carries an `attack_class`
  and parameters, never code. The *only* place it becomes a `Rule` is
  `synthesis/catalog.py`, which maps the class to a **vetted factory** from a
  fixed `RULE_FACTORIES` table and feeds it the parameters. The worst a poisoned
  source can do is propose parameters to a factory that already exists.
- **Unknown techniques fail closed, but are surfaced for triage.** A finding whose
  `attack_class` is not in the catalog (e.g. `novel`) produces no candidate — it is
  never silently auto-acted upon. It is not silently discarded either: it is recorded
  in the review queue under `uncatalogued` (each `"triaged": false`) so a human sees
  a technique the catalog cannot yet express. Acting on one means designing a new
  rule factory (a deliberate code change); `load_approved` reads only `candidates`,
  so an uncatalogued entry can never become a rule on its own. This closes the
  field-learning loop without widening the trust boundary. To turn scattered
  `novel` hits into direction, `proposals.cluster_uncatalogued` aggregates them by
  affected tool and `format_factory_prompt` renders a ranked *"N uncatalogued
  findings on tool X — consider a new rule factory"* report (shown by the pipeline
  demo). It only reports — a person still writes any new factory. Decisions about
  these findings (catalogue, defer to another layer, or out of scope) are logged
  in [`synthesis/TRIAGE.md`](synthesis/TRIAGE.md) so a recurring signal is not
  re-litigated each scan.
- **Nothing is applied without a human.** `review.py` writes every candidate to a
  queue with `"approved": false`; only entries a person flips to `true` (and that
  still validate against the catalog) are loaded. Applying an approved rule to a
  previously-unpoliced tool creates a **fail-closed** policy (no allowed roles),
  so an auto-proposed rule can never, by itself, open access.

### Run the end-to-end pipeline demo (no LLM, no network, no install)

```bash
python demo_bridge.py
```

Runs the whole chain over the bundled `scanner/sample_docs/`: scan → synthesize
candidates (dropping unknown classes) → write a review queue → approve → apply →
print before/after authorization for a couple of tool calls. Standard library
only, so the bridge is checkable deterministically before any real scanner exists.

### Scan documents into findings

```bash
# offline keyword extractor (default) — deterministic, stdlib only
python scanner/scan.py --docs path/to/docs --out findings.json

# LLM extractor — needs `openai`; point at the gateway or a local server
python scanner/scan.py --extractor llm \
  --base-url http://localhost:11434/v1 --api-key-env OLLAMA_KEY \
  --model llama3.1:8b --docs path/to/docs --out findings.json
```

The keyword extractor grounds by literal tool-name mention and matches a few
keywords — enough to prove the plumbing on the sample docs. The **LLM extractor**
is the real path for prose: it classifies into the closed taxonomy and grounds by
*meaning* against tool descriptions, treats the document as untrusted (clamping
its output back onto the allowed taxonomy and registry), and chunks long papers.
It is also given each tool's **argument schema** (`TOOL_SCHEMAS`) and the
recognized **principal attributes** (`PRINCIPAL_ATTRS`), so a proposed `arg_name`
or `attr_name` is grounded against real names — a hallucinated name is dropped,
which makes the candidate fail closed at the synthesis gate rather than becoming a
wrong-but-valid rule. If the backend is unreachable for an entire document the run
**fails fast with a clear error** rather than silently reporting nothing.

> The example tool registry, argument schemas, and principal attributes model a
> coding/dev agent (`read_file`, `write_file`, `run_shell`, `http_request`,
> `git_push`, `install_package`) defined in `example_policies.py`. For real use,
> replace them with the surface of the agent you are protecting — the LLM
> extractor grounds findings against those tool descriptions and argument names.

### Acquire a corpus from real sources

`scan.py` reads documents from a folder; `acquire.py` is how that folder gets
populated from the field. It fetches from the arXiv API and RSS/Atom feeds,
normalizes each entry into the same markdown shape the scanner reads, and records
what it has seen in a JSON ledger so re-runs only add genuinely new documents:

```bash
python scanner/acquire.py \
  --arxiv 'cat:cs.CR AND abs:"LLM agent"' --arxiv-full-text \
  --feed https://example.org/security/feed.xml \
  --out-dir corpus/
python scanner/scan.py --extractor llm --docs corpus/ --out findings.json
```

Standard library only (no new dependencies). A source that fails is skipped with a
warning rather than aborting the run; `--dry-run` reports what would be fetched.

**`--arxiv-full-text`** (recommended for real corpus runs): by default only the
abstract is fetched as the document body. Abstracts describe research *about*
vulnerabilities but rarely include the concrete `(tool, arg, control)` detail
the LLM extractor needs to produce a grounded finding — in practice they yield
zero findings. With `--arxiv-full-text` the fetcher retrieves each paper's
rendered HTML (`arxiv.org/html/<id>`), strips non-content elements, and uses the
full text as the body. If no HTML rendering exists for a paper the fetcher
silently falls back to the abstract. The canonical provenance URL (`/abs/`) is
unchanged — a reviewer can still trace any finding back to its source paper.

**`--full-text-max-chars N`** (default 40 000): cap on characters kept per
full-text document. References and appendices — which rarely carry a groundable
technique — fall off the end, keeping each document within a reasonable LLM
context window. Only relevant when `--arxiv-full-text` is set.

### Tests

The offline pipeline (scanner, synthesis, acquisition) has a unit suite that needs
no network, no LLM, and no Guardrails install:

```bash
poetry run pytest examples/configs/tool_call_guardrail/tests/
```

## Extending it

- **New tool:** add an implementation to `tools.py`, a `ToolPolicy` entry plus a
  `TOOL_SCHEMAS` argument list in `example_policies.py` (the schema is what lets
  the LLM extractor ground an `arg_name` against the tool's real arguments).
- **New constraint:** write a `Rule` — a function `(ToolCall, Principal) -> str | None`
  returning `None` to pass or a reason string to block — and add it to the tool's
  policy. `require_owns_arg` and `max_numeric_arg` in `policy.py` are examples.
- **New constraint the scanner may auto-propose:** add the rule factory to
  `RULE_FACTORIES` in `synthesis/catalog.py` and map an attack class to it in
  `CLASS_TO_FACTORY`. This is the *only* way to widen what a finding can become —
  a deliberate, reviewed act.
- **New document source:** implement the `Fetcher` protocol in
  `scanner/sources.py` (a `fetch()` that yields `SourceDoc`s); the arXiv and
  RSS/Atom fetchers are examples. Swap the LLM backend by passing a different
  `Chat` callable to `LLMExtractor` — any OpenAI-compatible endpoint works via
  `--base-url`.

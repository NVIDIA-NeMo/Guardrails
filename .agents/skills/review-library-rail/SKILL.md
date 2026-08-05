---
name: "review-library-rail"
description: "Review a pull request that adds or modifies a rail in nemoguardrails/library/ (built-in or vendor guardrail integration). Runs the mechanical gates first, then judgment dimensions tests cannot check (privacy honesty, retry semantics, fail-closed behavior, dialect coverage, docs accuracy), then a security pass. Use when reviewing rail-integration PRs, vendor integration contributions, or changes under nemoguardrails/library/. Trigger keywords - review rail PR, rail integration review, vendor integration PR, library rail review, guardrail contribution."
license: "Apache-2.0"
---

# Reviewing a Library Rail Contribution

This skill reviews PRs that add or change a rail under
`nemoguardrails/library/`. The contract being reviewed against is defined by
the `add-library-rail` skill; do not restate it, read it. Follow the
repository Review Mode rules (compare against the merge base with `develop`,
inspect tests as well as implementation, treat every finding as advisory
until verified against the real code path).

The guiding fact, learned from evaluation: contributors and reviewers both
comply with what is named and miss what is omitted. Work through every
numbered check; do not stop at the first few findings.

## Step 0: Scope and duplicates

- Confirm the PR is a rail contribution: files under
  `nemoguardrails/library/<name>/`, usually with tests and a docs page.
- Check for duplicate or in-flight integrations for the same vendor:
  `gh pr list --state open --search "<vendor>"` and the same for issues. If
  another PR covers the vendor, surface that before reviewing further.
- Is this vendor integration plausibly wanted? If there is no linked triaged
  issue or maintainer signal, flag for maintainer judgment; do not review it
  into shape first.

## Step 1: Run the mechanical gates (do not re-review what they check)

```bash
make test TEST="tests/rails/llm/test_builtin_rail_manifests.py tests/rails/llm/test_builtin_rail_conformance.py tests/rails/llm/test_library_flow_files.py"
make test TEST="tests/rails/llm/test_config.py tests/test_runtime_flow_gate_equivalence.py"
make test TEST=tests/http/test_library_boundary.py
make test TEST=<the PR's test files>
make test TEST=tests/recorded/rails/library ARGS="--block-network -q"
git diff <merge-base> -- pyproject.toml uv.lock   # usually empty; see below
```

A red gate is a finding by itself; report it and keep going. A green gate
means that dimension is DONE; spend review attention only on what follows.

Most rail PRs leave `pyproject.toml` untouched, since a rail's optional
vendor package is declared in the manifest rather than in packaging. A
diff there is not automatically wrong: rail-scoped extras exist (`sdd`,
`gcp`, `jailbreak`, `multilingual`) and `RailRequirements.extras` is a
first-class field, so a PR adding one must justify the extra and declare it
(exemplar: `nemoguardrails/library/sensitive_data_detection/rail.py`). An
undeclared or unjustified dependency change is the finding.

## Step 1b: New-rail completeness (presence, not quality)

Gates verify what exists; they cannot flag what is absent. For a PR that
adds a new rail, check each of these is PRESENT before judging quality:

- `rail.py` with a `RAIL` manifest. A rail without one falls back to legacy
  loading and silently bypasses every catalog-keyed gate, so its absence
  makes all green gates above meaningless for this rail.
- Both-dialect unit tests: block path under Colang 1 AND under
  `colang_version: "2.x"`. A v2 flow file with no v2 test is unexecuted
  code; inbound PRs have shipped v2 files that parse but cannot run.
- A flow-level test of the action raising (vendor down, fail-closed).
- The recorded outcome triad (allow, block, provider-error) under
  `tests/recorded/rails/library/` with config dir and `configs.py`
  constant; recorded coverage is mandatory per
  `nemoguardrails/library/README.md`.
- Docs page and example config.

Each missing item is a finding; request it rather than inferring it is
covered elsewhere.

## Step 2: Judgment dimensions (what tests cannot check)

Each item names the check, where to look, and what "wrong" looks like.

1. **Privacy honesty.** Compare `RailPrivacy` in `rail.py` against what the
   request layer actually transmits and what the vendor's own documentation
   says about retention. `sends_user_text`/`sends_bot_text` must match the
   surfaces' bindings, including `sends_retrieved_chunks` for a retrieval
   surface; `remote_services` must be non-empty for any HTTP
   rail; `data_retention` must be stated if the vendor states one. A
   defaults-only `RailPrivacy()` on a vendor rail is almost always wrong.
2. **Requirements honesty.** Every env var the code reads
   (`os.environ`/`os.getenv`) appears in `RailRequirements.env_vars`; every
   lazily imported package appears in `optional_dependencies` (and in
   `extras` when the package already ships in a nemoguardrails extra); the
   docs `pip install` line matches the declaration and carries the version
   bound, which `RailRequirements` cannot express.
3. **Retry semantics.** For HTTP rails: a rail-owned `RetryPolicy` in one
   named module-level place, either a constant or a single config-derived
   builder, never inline at the call site; POST retries only via explicit
   `retryable_methods={"POST"}` carrying a comment that names one of the
   three permitted resend-safety mechanisms (documented idempotency, an
   idempotency key header, or a stateless scan endpoint), and no comment is
   a finding; no hand-rolled retry
   loop around `http_call`. Then verify any retry/backoff claims in docs or
   comments against `nemoguardrails/http/retry.py`: backoff is full-jitter
   exponential with a lower bound of zero, `Retry-After` is honored only
   within `max_retry_after` and is otherwise discarded rather than capped
   unless `clamp_retry_after=True`, and vendor override headers are ignored
   unless `honor_retry_override_header` is set. Contributors document this
   from guesswork.
4. **Fail-closed behavior, tested not asserted.** Vendor errors, timeouts,
   and malformed payloads must surface as typed errors or blocks, never a
   silent allow. There must be a flow-level test of the action raising; a
   summary or comment claiming fail-closed without a test is a finding.
5. **Dialect coverage.** The flow-files gate checks structure; behavior
   needs tests. Both dialects' block path must be exercised end to end in
   the UNIT suite (`colang_version: "2.x"` variant present?), and a per-rail
   Colang 2 RECORDED test is itself a finding, since the recorded suite is
   dialect-single with a single shared v2 smoke pin. If any surface declares
   a `transform_target`, the transform path also needs a Colang 2 behavior
   test; the flow-gate harness is Colang 1 only, so a v2 transform can be
   entirely unexecuted while the block-path tests pass. Every
   `enable_rails_exceptions` branch must be either tested or absent. Check
   `flows.co` and `flows.v1.co` against the four equivalence items in
   add-library-rail Step 3 (same flow names, same action and bindings per
   flow, same branch structure, same asserted output); divergent semantics
   between dialects is a finding even when both parse.
6. **Outcome discipline.** Actions return `RailOutcome` with neutral
   evidence in metadata; refusal prose, exception types, or localization in
   the outcome is a finding. Flows must only read metadata keys the action
   actually sets (grep the action for each `metadata[` subscript used in a
   flow). Then cross-check the config model: every field in `rail_config.py`
   must have a test at a non-default value, and decision-changing fields
   need both sides of the boundary. Read the comparison operator against the
   documented semantics; an inverted or off-by-one threshold passes every
   gate.
7. **Telemetry content-freeness.** No logging of checked text, request or
   response bodies, URLs with query strings, API keys, or tokens; exception
   messages must not embed payloads. Check `log.` calls and every f-string
   in raised errors.
8. **Test placement and hygiene.** Deterministic tests in `tests/` with
   `RecordingHTTPClient` (from `nemoguardrails.testing`) injected via
   `register_action_param`; recorded suite entries follow the
   `recorded-tests` skill (vcr-structured, or non-vcr with a defensible
   reason). Nothing enforces that split, so read every suite test in the
   diff that has no `vcr` mark and challenge it against the skill's
   criteria. Secrets via `monkeypatch`; any real-looking key in a test or
   cassette is a blocking finding.
9. **Docs accuracy.** Every statement in the catalog page must match the
   code: config options, defaults, env vars, install line, the vendor's data
   retention period (matching `RailPrivacy.data_retention`), and
   limitations. Verify each "known limitation" is real; a false limitation
   is as bad as a missing one, and an uncited or unreproducible one is a
   finding. `docs_url` in the manifest is the repo-relative `.mdx`
   path of the page the PR adds; the manifest gate asserts that path
   resolves.
10. **Fail-open semantics.** If the rail supports fail-open at all: it must
    default to fail-closed; the switch must live in the rail's config model
    where a config reviewer sees it, not in an env var (env vars are for
    secrets, and fail-open flips the rail's security posture); a fail-open
    pass must be distinguishable from a genuine vendor clear, carrying an
    explicit marker in the outcome metadata (as `_fail_open_outcome` in
    `nemoguardrails/library/f5/actions.py` does) so it is never mistaken for
    a real clear in logs and traces, and it must be logged. A
    fail-open path also needs its own test alongside the fail-closed one.
    Cross-check the docs against the code: if the page claims the switch
    covers availability only, confirm the fail-open branch actually excludes
    4xx. Failing open on 401 or 403 while documenting otherwise leaves a
    revoked key silently unguarded, and is a finding.
11. **Cassette provenance.** A cassette is a claim by its submitter, and an
    agent or contributor without service access can fabricate one that
    looks recorded. The sanitizer fixed-point gate
    (`tests/recorded/test_cassette_provenance.py`) catches naive
    fabrication (realistic ids, timestamps, cookies that the recorder
    would have scrubbed), but a sentinel-perfect forgery passes it, so for
    new cassettes from external contributors also ask HOW they were
    recorded. If the answer is unverifiable and the org holds no key to
    re-record, the honest options are maintainer re-recording before
    merge or demoting the tests to the explicit `fake_cassette` regime;
    do not merge unverifiable cassettes as recorded truth.
12. **Conformance drift (a green suite can hide this).** The conformance
    gates pass if a contributor made them pass the WRONG way, so diff the
    generic infrastructure, not just the rail. Reject any change to
    `test_builtin_rail_manifests.py`, `test_builtin_rail_conformance.py`, or
    `test_library_flow_files.py` unless the PR's purpose is the gate itself.
    Reject a new entry in `LEGACY_UNMANIFESTED_PACKAGES` or
    `NON_PORTABLE_DECLARED_FLOWS` unless it carries an explicit, reviewed
    design rationale (a new rail should satisfy the contract, not join the
    exceptions). For any change to `schemas/rails_config.snapshot.json`,
    confirm it was regenerated for a deliberate config-schema change and
    read the diff as a public-schema change, not a mechanical refresh.
13. **Error-path coverage (unit, exhaustive).** Dimension 4 checks that
    errors fail closed; this checks that EVERY error branch is actually
    exercised. Enumerate the action's failure branches (timeout, connection
    error, each handled status class, 429 retried and retry-exhausted,
    malformed payload, missing credential, missing optional dependency, each
    crossed with fail-open/closed where supported) and require a unit test
    per branch with a mocked transport. A branch with handling code but no
    test is a finding. These belong in the unit layer, not recorded: a
    synthetic error forced into `tests/recorded/` violates the placement
    rule, so do not accept "it's covered by the recorded suite" for anything
    but the recordable real error.
14. **Client ownership and error typing.** Neither is gate-checked; the HTTP
    boundary test is an import-only AST check, so read the code. The action
    must never call `close` or `aclose` on an injected `http_client`; only
    the `factory=` path passed to `http_call` may own a client, and that
    factory must return the fully composed closable client, since wrapping
    an injected client must not transfer ownership
    (`nemoguardrails/http/request.py`). Vendor failures must surface as
    rail-specific error types (the `errs.py` pattern, exemplar
    `nemoguardrails/library/clavata/errs.py`), not as raw transport or
    vendor SDK exceptions escaping the action.

## Step 3: Security pass

- **Credential handling**: key read from env only, never from config values
  that end up in logs or snapshots; JWT/signature construction does not log
  the secret; no credentials committed in tests, cassettes, or examples.
- **Outbound URL control**: if the config exposes `base_url` or an endpoint
  override, note that guardrails configs are sometimes attacker-adjacent;
  the URL must only ever be used for the vendor call (no template
  injection into paths), and redirects stay disabled (transport default).
- **Input handling**: user/bot text goes into the request body as data,
  never interpolated into URLs, headers, or shell commands; response
  parsing treats the payload as untrusted (no eval, no attribute access on
  unvalidated structures without try/except).
- **Deserialization**: only `response.json()`; anything like pickle or yaml
  load on vendor bytes is blocking.

## Step 4: Process conformance

- DCO sign-off on every commit; conventional commit title; no changelog
  edits; AI_POLICY disclosure if applicable.
- Docs build if docs changed: `make docs-fern`.
- Every reviewer finding verified against the actual code path before it is
  reported (Review Mode rule); drop anything you cannot reproduce or point
  to a line for.

## Verdict format

Report findings ranked by severity, each with file:line evidence and the
dimension number it came from. State explicitly which gates ran and their
results, and which dimensions came up clean; a review that only lists
problems hides its own coverage. End with one of: approve, approve with
nits, request changes (list the blocking subset).

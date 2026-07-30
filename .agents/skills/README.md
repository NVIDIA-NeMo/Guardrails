# Agent Skills

Optional agent skills for clients that support skill discovery (see the
repository `AGENTS.md`). Each skill is a `SKILL.md` with frontmatter
(`name`, `description` with trigger keywords) and a checklist-style body.

## Skills

- `guardrails-developer-guide`, `guardrails-developer-create-guardrails`:
  product-usage skills for people building guardrails configurations.
- `recorded-tests`: decides when a change requires a recorded/VCR test and
  whether a test belongs in `tests/recorded/` or `tests/`.
- `add-library-rail`: step-by-step contribution guide for a new rail under
  `nemoguardrails/library/`.
- `review-library-rail`: reviewer counterpart for rail-contribution PRs;
  runs the mechanical gates first, then the judgment dimensions tests
  cannot check.

The three rail/testing skills assume the manifest, RailOutcome,
`nemoguardrails.http`, and `RailRequirements` contracts; they describe this
codebase state, not older releases.

## Design rationale for the rail and testing skills

**Skills are checklists, not principles.** An evaluation run (a fresh agent
session implementing a fictional vendor rail against `add-library-rail`)
showed that agents comply with what a skill names and miss what it omits,
close to one-for-one: every defect in the produced rail mapped to a rule
the skill did not state, and every stated rule, including the non-obvious
ones, was followed. Consequence: these skills enumerate concrete checks
with file paths and commands, and when a gap is found the fix is a named
rule in the skill, not better prose.

**Division of labor with enforcement.** Anything mechanically checkable is
a test, not skill text: manifest completeness
(`tests/rails/llm/test_builtin_rail_manifests.py`), cross-artifact
conformance (`tests/rails/llm/test_builtin_rail_conformance.py`), the HTTP
transport boundary
(`tests/http/test_library_boundary.py`), cassette provenance
(`tests/recorded/test_cassette_provenance.py`), and flow-file
validity across dialects (`tests/rails/llm/test_library_flow_files.py`).
Skills own placement decisions and judgment calls; the review skill's first
step is to run the gates and then spend attention only on what they cannot
prove.

**Which test layer owns Colang dialect coverage.** Library rails ship flow
files in both Colang dialects, and historically only Colang 1 was tested,
which let structurally broken Colang 2 files ship silently. The split:

- The flow-files gate owns structure: both dialect files parse, define the
  manifest-declared flows, and invoke dispatcher-resolvable actions. A
  declared flow whose Colang 2 definition is parameterized is exempt from
  Colang 1 presence, because Colang 1 has no parameterized flows; the
  exemption is derived from the parse result rather than an allowlist.
- Unit tests own dialect behavior: the block path in BOTH dialects via
  `TestChat` (Colang 2 via `colang_version: "2.x"`), plus every
  `enable_rails_exceptions` branch.
- Recorded tests are dialect-single. For library rails the provider
  traffic is made by the action, so both dialects record identical wire
  bytes; what differs is deterministic flow routing, which by the recorded
  suite's own placement criterion ("could the provider change this test's
  expectation?") belongs in unit tests. Duplicating cassettes per dialect
  doubles refresh cost for zero drift-detection value. Two exceptions: a
  single Colang 2 smoke pin in the library recorded suite is worth its one
  cassette, and the `public_api` generation surfaces issue genuinely
  different LLM traffic under the Colang 2 runtime, so recorded v2
  coverage there is provider-shaped and legitimate.

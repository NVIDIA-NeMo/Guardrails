# AGENTS.md

`CONTRIBUTING.md` is canonical for public contribution workflow: issues,
assignment, pull requests, refactors, changelogs, validation, commits, and DCO.
`AI_POLICY.md` is canonical for public AI-assisted contribution policy.

When working inside `nemoguardrails/`, also follow `nemoguardrails/AGENTS.md` for
runtime, public-API, and provider-integration rules.

## Quick Rules

- Agent-specific rule: do not submit issues, PRs, or draft PRs through browser
  automation, the GitHub API, `gh`, or similar tooling. Draft text for a human to
  review and submit, following the repo's issue/PR templates so it can be
  submitted as-is.
- Do not push branches or prepare public-submission-ready PR materials unless
  the linked issue is triaged and assigned to the human contributor.
- Do not implement refactors unless a maintainer has approved the plan and
  assigned the work.
- Never edit `CHANGELOG.md` or `CHANGELOG-Colang.md` manually.
- Do not commit secrets, credentials, or sensitive provider data, and do not
  fabricate results, approvals, or citations. See `AI_POLICY.md` Safety and
  Privacy for the canonical list.
- Do not add generated media, large generated assets, or synthetic datasets
  without clear provenance and maintainer alignment.
- Unit tests must not call live LLM or provider services.
- Do not add license headers manually. Pre-commit handles license insertion.
- Do not add comments unless explicitly requested; keep existing comments,
  docstrings, and license headers unless your change makes them inaccurate.
- Use Poetry for Python commands: `poetry run python ...`,
  `poetry run pytest ...`, `poetry run pre-commit ...`.

## Repository Map

- Main package: `nemoguardrails/`
- Tests: `tests/` and `benchmark/tests` (the `testpaths` in `pytest.ini`)
- Schemas and validation snapshots: `schemas/`
- Default development branch: `develop`

## Setup

- Install development dependencies:

  ```bash
  poetry install --with dev
  ```

- Install documentation dependencies when working on docs:

  ```bash
  poetry install --with dev,docs
  ```

- Do not add dependencies to `pyproject.toml` or update `poetry.lock` unless the
  task requires it. For temporary local investigation, use:

  ```bash
  poetry run pip install <package-name>
  ```

- When a dependency change is required, keep it in the narrowest appropriate
  dependency group or optional extra, add clear compatibility bounds, and avoid
  moving optional integration dependencies into the default install path.

## Validation

Canonical command reference: `CONTRIBUTING.md` Validation. The table below adds
agent-operational diagnosis commands; see `CONTRIBUTING.md` for shared rows such
as tox and package coverage.

| Task | Command |
| --- | --- |
| Run the test suite | `make test` (pytest-xdist parallel; unsets live-provider keys so unit tests cannot reach live services; runs all `pytest.ini` testpaths) |
| Focused test | `make test TEST=path/to/test_file.py::test_name` (extra flags via `ARGS="-k ... -q"`) |
| Serial, deterministic run | `make test WORKERS=1` (no parallelism, still unsets live keys) |
| Coverage | `make test-coverage` |
| Pre-commit hooks | `poetry run pre-commit run --all-files` |
| Docs check | `make docs-fern` |
| Ruff diagnosis | `poetry run ruff check path/to/file.py` |
| Ruff formatting diagnosis | `poetry run ruff format path/to/file.py` |
| Pyright diagnosis | `poetry run pyright` |

| Change type | Minimum validation |
| --- | --- |
| Docs or repository metadata only | `poetry run pre-commit run --files <changed files>`; build docs when rendering, links, examples, or docs configuration may be affected |
| Runtime bug fix | Focused regression test plus pre-commit on changed files; broaden when shared behavior is touched |
| Public API, config, or Colang behavior | Focused tests plus related docs/examples; add broader package tests when compatibility risk is meaningful |
| Server, streaming, tracing, actions, or generation | Targeted tests for the changed path and fallback/unsupported path |
| Packaging, dependencies, or lockfiles | Relevant install/package checks plus pre-commit; keep dependency diffs separate from unrelated changes |

- For PR-ready code changes, pre-commit is the authoritative lint, format,
  license-header, and type-checking path.
- Standalone Ruff, Ruff format, and Pyright runs are local diagnosis only; run
  pre-commit on changed files before handoff and report if it is skipped.
- `make test` runs every `pytest.ini` testpath, so it includes `benchmark/tests`,
  not just `tests/`; scope with `TEST=`. The default suite needs no network:
  `tests/conftest.py` swaps the default FastEmbed model for a deterministic
  provider, so only `real_embeddings`-marked tests use the real model.
- Diagnose isolation flakiness by comparing `make test` with `make test
  WORKERS=1` (same env-safety, no parallelism); `serial`/`slow` markers in
  `pytest.ini` are advisory and not enforced by the parallel runner.
- `make test-serial` and bare `poetry run pytest` do NOT unset live-provider
  keys; prefer `make test` / `make test WORKERS=1` so unit tests cannot reach
  live services.

## Contribution Workflow

- Follow `CONTRIBUTING.md` for issue, assignment, PR title, refactor, changelog,
  validation, commit, and DCO policy.
- Follow `AI_POLICY.md` for disclosure, human accountability, safety, and
  privacy requirements.
- For non-trivial features, API changes, refactors, or behavior changes without
  clear maintainer direction (a linked issue that is triaged, assigned to you,
  and has an agreed approach recorded in the thread), stop at a proposal or
  implementation plan (an issue comment, or a throwaway `PLAN.md` PR maintainers
  can review) rather than implementing.
- Before preparing PR-shaped work, check for duplicate or in-flight effort with
  read-only `gh` (distinct from the no-`gh`-submission rule above):
  `gh issue view <issue> --comments`, `gh pr list --state open --search "<issue>
  in:body"`, and `gh pr list --state open --search "<area keywords>"`. If an open
  PR already covers the change, do not prepare a duplicate; if your approach
  differs materially, surface that difference in the issue draft for a maintainer.
- If work is exploratory, draft an issue comment with the branch and relevant
  files instead of opening a PR.
- Use the Conventional Commit-style titles described in `CONTRIBUTING.md`.
- Do not prefix PR titles or commit messages with agent markers, and do not add
  AI tools or agents as commit co-authors (no `Co-Authored-By` trailers for AI).

## Review Readiness

- Follow `CONTRIBUTING.md` for review-readiness policy (CodeRabbit, Greptile,
  human comments, readiness labels): address or reply to every open review comment
  before requesting maintainer review, do not resolve reviewers' own threads, and
  do not self-apply the readiness label.

## Code Changes

- For maintainer-approved refactors, add or update characterization tests before
  changing subtle behavior; when the existing suite does not cover the refactored
  code, keep the equivalence check you used to prove behavior is unchanged.

## Documentation And Generated Files

- Update docs when changing user-visible behavior, public APIs, configuration
  syntax, examples, or installation requirements.
- For optional integrations, document whether the integration is optional, which
  extras or packages are required, which API keys or environment variables are
  expected, and whether the integration uses the default OpenAI-compatible
  framework path or the LangChain framework.
- When documenting model or provider examples, state the relevant model type,
  routing mode, supported modes, and known limitations rather than assuming the
  example generalizes to every backend.
- Use current generally-available model IDs in docs/examples (verify against the
  provider's docs), and do not change shipped default model parameters as a
  documentation update.
- Do not hand-edit generated files or lockfiles unless the task explicitly
  requires regenerating them with project tooling.
- Put release-note context in issue or PR draft text instead of changelog files.
- For notebook documentation, follow `CONTRIBUTING.md`. Do not run
  `build_notebook_docs.py` unless explicitly asked; it currently runs broad git
  staging and pre-commit commands. Use a clean worktree if it must be run.

## Review Mode

- When reviewing a branch, compare against the merge base with `develop` and
  inspect tests as well as implementation.
- Before handing off, run pre-commit first (per Validation, so line locations are
  stable), then a structured code review (for example `codex review` or
  `/code-review`) with a security pass for auth, input-handling, deserialization,
  or external-call changes. Treat findings as advisory: verify each against the
  real code path and loop until clean.

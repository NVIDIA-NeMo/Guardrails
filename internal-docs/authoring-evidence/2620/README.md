# PR #2620 Style Corrections

## Scope and Candidate

- Task: review and correct the documentation style of [PR #2620](https://github.com/NVIDIA-NeMo/Guardrails/pull/2620).
- Starting documentation revision: `39980e52e175aeec7a266ca44d793b753260c01d`.
- Workflow branch: `docrelay/workflow-04d413e7`.
- Reviewed the complete four-file PR delta against merge base `d36e2fcb4` with the PR's develop base `7f9ea4d1009bc5bd87eb4b0c94169e7f975c5684`. The local `develop` ref is older; its larger diff is not the task scope.
- PR #2620 remains unchanged. Read-only PR searches identified it as the matching work, not a new competing implementation.
- No subagent interface was available. Writer performed the drafting and author checks directly; independent evaluation is still pending.

## Page Decisions and Coverage

| Path | Audience and Purpose | Changes |
| --- | --- | --- |
| `docs/about/release-notes.mdx` | Developers; release reference | Added meaningful section introductions. Split independent features and configuration fixes. Ordered migration actions without changing release categories, claims, anchors, or links. |
| `docs/README.mdx` | Contributors; build, preview, and publishing how-to | Separated setup actions, numbered each preview alternative, and expanded snapshot creation and revision into atomic steps. Retained snapshot immutability, review, DCO, generated-file, and tag-before-branch safeguards. |
| `fern/docs.yml` | Site configuration | Inspected; unchanged. All version entries and redirects remain intact. |
| `fern/fern.config.json` | CLI pin | Inspected; unchanged at `5.223.1`. |

This is a focused style assignment, not a documentation restructuring. The canonical About release page remains in place. The contributor README is not in `docs/index.yml`; `CONTRIBUTING.md` explicitly links to that path for publishing guidance. Retaining it preserves the repository's contributor entry point. No page moves, navigation additions, or new public sections are needed. Task evidence stays outside the published guide in `internal-docs/`.

All original 0.25.0 coverage remains: Gateway, hooks, per-tool checks, language detection, Zscaler, Python support, event-loop patching, streaming tools, jailbreak handling, deprecations, streaming errors, injection handling, and configuration validation. All 12 previous-release links remain unchanged.

## Inspected Technical Support

Implementation sources below belong to `NVIDIA-NeMo/Guardrails` at the starting revision, not a final release audit. No dependency implementation or external repository changes were needed for these editorial changes.

- `pyproject.toml`: Python `>=3.11,<3.14` and optional `nest-asyncio` extra.
- `nemoguardrails/patch_asyncio.py:apply`: optional patch dependency and asynchronous alternative.
- `nemoguardrails/deprecation.py`: `DEPRECATED_SINCE`, `REMOVAL_RELEASE`, and Colang 2.x replacement limitation.
- `nemoguardrails/guardrails/guardrails.py:Guardrails.__init__`: supported configurations use IORails; unsupported configurations can fall back, except per-tool configurations that must not silently lose checks.
- `nemoguardrails/guardrails/iorails.py:stream_async`: text events and completed `ToolCallChunk` events.
- `nemoguardrails/rails/llm/config.py`: canonical and deprecated tool-rail fields, migration validator, model binding traversal, and unknown-key suggestions.
- `nemoguardrails/rails/llm/options.py`: rail types and `tool_violations`; `nemoguardrails/server/api.py` and `server/schemas/openai.py`: check response propagation and terminal stream error handling.
- `nemoguardrails/library/jailbreak_detection/model_based/input_length.py`: 2,048-token validation, including special tokens; `library/jailbreak_detection/actions.py`: oversize blocks and invalid-verdict errors.
- `nemoguardrails/library/injection_detection/actions.py`: omission followed by matching again, with fail-closed handling.
- `nemoguardrails/library/language_detector/actions.py`: allow/block lists and localized refusal metadata.
- `nemoguardrails/server/hooks/tool_policy.py`: name and argument policies.
- Cross-checked `docs/migration/0.25.mdx`, `docs/reference/deprecations.mdx`, and Gateway guard-contract documentation. The task does not assert that the open release-preparation PR proves final released behavior.
- `Makefile`, `package.json`, `scripts/watch-fern-preview.mjs`, and `.github/workflows/docs-build.yaml`: actual build, preview, pinned CLI, and workflow behavior.

### Publishing Qualification

The README previously promised `/not-latest` behavior. The checked-in workflow has tag triggers for both `docs-v*` and `v*`, but no public tag-publishing job or `/not-latest` handling. The revised README retains the convention and adds a visible warning plus maintainer-confirmation step. It no longer promises protection that this checkout cannot establish. External deployment automation was not inspected or executed.

### Immutable Snapshot

Read-only remote tag resolution and a local fetch established that `fern-docs-snapshot-v0.25.0` peels to `a1017fef730b330d7948b50e2af2354555363c68`. Inspected that commit's `fern/docs.yml` and release-notes tree. The snapshot has its own uncorrected release-note prose and pins SDK source to `6d852a7993559947079102ecb1a375112ba51ba3`.

These edits affect Latest, not the immutable v0.25.0 snapshot. No new snapshot is required to deliver this scoped branch correction. Applying the same prose corrections to the historical version later requires a separately reviewed snapshot revision such as `-r2`; never move the current tag. Historical rendering and final release status are not certified by this task.

## Validation

Context: November 18, 2026; task worktree; Node.js `v22.23.5`; CPython `3.12.3`. uv was initially unavailable. Installed uv `0.13.9` under `/tmp/docrelay-tools` without changing the repository or global configuration. Python commands used `PATH=/tmp/docrelay-tools:$PATH` and `uv run --locked`. uv warned that its version differs from the project's build-system range; dependency manifests and lockfiles remain unchanged.

| Command or Check | Exit and Observed Result | Evidence |
| --- | --- | --- |
| `npm ci` | 0; one package installed, zero audit vulnerabilities | Tool execution transcript |
| `git diff --check` | 0; no whitespace errors | Final execution transcript |
| `uv run --locked pre-commit run --files docs/README.mdx docs/about/release-notes.mdx` | 3; hook initialization blocked by GitHub HTTP 503, repeated on retry | `pre-commit.log`, `pre-commit-retry.log`, `pre-commit-final.log`, associated `.exit` files |
| `uv run --locked ty check` | 0; all checks passed; diagnostic fallback, not a substitute for pre-commit | `ty.log`, `ty.exit` |
| `make docs-fern` | 0; generated local API/SDK references; zero errors and one warning | `docs-fern-final.log`, `docs-fern-final.exit` |
| `node scripts/run-fern-with-ref-sdk.mjs check --warnings` | 0; warning is authenticated published-redirect comparison skipped because `FERN_TOKEN` is not set | `fern-warnings.log`, `fern-warnings.exit` |
| In-memory Node assertions using `yaml` and `node:assert/strict` | 0; preserved link targets and anchors; unchanged Fern configs/navigation; version example matches every selector entry; exactly one 0.25.0 section | `preservation.log` |
| Pinned `fern-api` local `docs dev` | Unavailable: bootstrapped but failed watching files with `EMFILE`; no usable local preview URL | `preview.log` |

The final pre-commit attempt also included this evidence README and exited 3 at the same fetch. Pre-commit remains unavailable because its upstream hook repository returns HTTP 503, not because an authoring check found a defect. Retry it when GitHub access recovers. Source inspection found no code changes requiring runtime tests or a security-specific code review. Writer manually reviewed the final diff after checks for meaning preservation, heading introductions, list nesting, atomic procedures, and long sentences. This is author review, not an independent verdict.

Unverified: browser rendering, authenticated redirect comparison, historical-version rendering, and live publishing. No credentials, publishing tags, remote branches, or PR content were changed. Generated references remain ignored and uncommitted. Publication must wait for Evaluator approval of the exact candidate commit.

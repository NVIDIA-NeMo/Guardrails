---
name: "guardrails-contributor-dev-setup"
description: "Sets up and verifies a local Guardrails contributor environment. Use when cloning the repo, installing development dependencies, choosing optional extras, running focused tests, installing pre-commit hooks, or diagnosing missing Poetry/pre-commit/dev tools. Trigger keywords - dev setup, contributor setup, install dev dependencies, poetry install, local environment, run tests, pre-commit, tox, make test."
license: "Apache-2.0"
---

# Contributor Development Setup

Use this skill when a contributor needs to set up or verify a local development environment for the Guardrails repository.
`CONTRIBUTING.md` is canonical for the public contribution workflow.
Root `AGENTS.md` is canonical for agent-specific repository rules and validation guidance.

## Prerequisites

Verify the contributor has:

- Git.
- Python 3.10, 3.11, 3.12, or 3.13.
- Poetry `>=1.8,<2.0`.
- Compiler and development tools needed to build Annoy on their platform.

Do not install system packages without explaining the change and getting user approval.

## Clone And Install

For a fresh clone:

```bash
git clone https://github.com/NVIDIA-NeMo/Guardrails.git nemoguardrails
cd nemoguardrails
poetry install --with dev
```

For documentation work, install the Python docs dependency group:

```bash
poetry install --with dev,docs
```

For Fern-specific docs workflow, use the `guardrails-contributor-docs` skill.

Optional extras include `sdd`, `eval`, `gcp`, `tracing`, `jailbreak`, `multilingual`, `server`, `chat-ui`, and `all`.
Install only the extras needed for the task.

Example:

```bash
poetry install --with dev -E server -E tracing
```

## Temporary Local Tools

For local investigation tools, use the Poetry environment without modifying project dependencies:

```bash
poetry run pip install <package-name>
```

Do not commit environment-only dependency changes.

## Run Commands Through Poetry

Use Poetry for Python commands:

```bash
poetry run python ...
poetry run pytest ...
poetry run pre-commit ...
```

If `poetry` is unavailable but the repository `.venv` already has the needed tool, report that fallback clearly.
For example, `.venv/bin/pre-commit` can run local hooks, but this is a fallback for the current machine, not the canonical command.

## Validation Commands

Start with the smallest meaningful check, then broaden when the change touches shared runtime behavior, public APIs, packaging, server behavior, tracing, or docs.

| Task | Command |
| --- | --- |
| Focused tests | `make test TEST=path/to/test_file.py::test_name` |
| Full test suite | `make test` |
| Serial deterministic test run | `make test WORKERS=1` |
| Supported Python versions | `poetry run tox` |
| Pre-commit hooks | `poetry run pre-commit run --all-files` |
| Changed-file pre-commit | `poetry run pre-commit run --files <changed files>` |
| Docs check | `make docs-fern` |
| Coverage | `make test-coverage` |
| Ruff diagnosis | `poetry run ruff check path/to/file.py` |
| Ruff formatting diagnosis | `poetry run ruff format path/to/file.py` |
| Pyright diagnosis | `poetry run pyright` |

`make test` unsets live-provider keys so unit tests cannot reach live services.
Prefer `make test` or `make test WORKERS=1` over bare `poetry run pytest` for unit-test safety.

## Pre-commit Hooks

Install local pre-commit hooks if the contributor wants checks before every commit:

```bash
poetry run pre-commit install
```

The hooks run Ruff, Ruff format, license-header insertion, and Pyright.
For PR-ready code changes, pre-commit is the authoritative lint, format, license-header, and type-checking path.

## Troubleshooting

If Poetry is missing:

- Explain that Poetry `>=1.8,<2.0` is required.
- Do not silently switch the repository to another package manager.
- If a local `.venv` fallback exists, use it only to inspect or validate current work and report the fallback.

If tests try to call live providers:

- Stop and switch to the project's safe `make test` targets.
- Unit tests must mock LLM and provider services.

If docs validation fails during SDK generation with a network fetch error:

- Rerun with appropriate network access before treating it as a docs-content failure.
- Report whether the rerun passed.

## Related Skills

- Use `guardrails-docs` for product-usage questions.
- Use `guardrails-contributor-docs` for Fern docs editing, preview, links, and publishing workflow.
- Use `guardrails-contributor-create-pr` when drafting PR text and verification summaries for a human to submit.

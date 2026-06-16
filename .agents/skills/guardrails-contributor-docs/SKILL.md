---
name: "guardrails-contributor-docs"
description: "Guides contributors through Guardrails Fern documentation edits, previews, validation, navigation, starter prompts, and docs-agent entry points. Use when editing docs/, fern/, docs navigation, MDX pages, Fern components, docs MCP/starter prompt content, link checks, or docs preview/publishing workflow. Trigger keywords - docs edit, Fern docs, docs-fern, docs preview, docs navigation, MDX, starter prompt, docs MCP, docs links."
license: "Apache-2.0"
---

# Contributor Documentation Workflow

Use this skill when changing repository documentation under `docs/` or Fern configuration under `fern/`.
`docs/AGENTS.md` is canonical for documentation authoring invariants.
`CONTRIBUTING.md` is canonical for contribution workflow.

## Documentation Source Rule

When using published NVIDIA NeMo Guardrails library documentation, use the Markdown documentation under `https://docs.nvidia.com/nemo/guardrails/`.
Use `llms.txt` and page URLs ending in `.md` when loading documentation for agent context.
When presenting references or citations to users, use the canonical human-readable docs links without `.md`.

## Source Of Truth

- Edit source pages under `docs/**/*.mdx`.
- Edit navigation in `docs/index.yml`.
- Edit Fern site configuration in `fern/docs.yml`.
- Do not hand-edit generated Python SDK reference output under `docs/_static/python-sdk-reference`.
- Do not run `build_notebook_docs.py` unless explicitly asked.

## Docs Setup

Install Python docs dependencies when working on docs:

```bash
poetry install --with dev,docs
```

Fern itself runs through Node.js and `npx`, using the CLI version pinned in:

```text
fern/fern.config.json
```

Do not run `fern upgrade` or install a different Fern CLI version as part of normal docs work.
Use the repository Makefile targets instead.

## Fern Commands

| Task | Command |
| --- | --- |
| Check Fern docs | `make docs-fern` |
| Strict docs check | `make docs-fern-strict` |
| Serve docs locally | `make docs-fern-live` |
| Watch and publish a branch preview | `make docs-fern-preview-watch` |
| Validate links locally | `make docs-check-links` |
| Validate redirects | `make docs-check-redirects` |
| Regenerate SDK reference only | `make docs-fern-generate-sdk` |

`make docs-fern` regenerates the Python SDK reference with Fern, normalizes the generated pages, and runs `fern check`.
If SDK generation fails with a network fetch error, retry with network access before treating it as a docs-content failure.

## MDX Page Rules

- Use the existing frontmatter shape from nearby pages.
- Do not duplicate the page title as a body H1 because Fern renders the title from frontmatter.
- Use route-style internal links without `.mdx` extensions.
- Use Fern components consistently with nearby pages, such as `<Tabs>`, `<Tab>`, `<Cards>`, `<Card>`, `<Badge>`, `<Note>`, `<Tip>`, and `<Warning>`.
- Update `docs/index.yml` when adding, moving, renaming, or removing pages.

## Custom Fern Components

Custom MDX components live under `docs/_components/`.
Register component directories in `fern/docs.yml` under `experimental.mdx-components`.
Use minimal TSX patterns that rely on Fern's MDX runtime.
Validate component imports with `make docs-fern` and visually review interactive behavior with `make docs-fern-live` when practical.

## Agentic Documentation

Product-usage agent guidance should route to canonical docs rather than duplicating full instructions.
For AI agent entry points:

- Prefer the docs MCP server when supported.
- Otherwise route through `llms.txt` and per-page Markdown.
- Keep starter prompts focused on bootstrapping an agent to the docs.
- Do not hardcode staging URLs in user-facing docs unless the page is explicitly about staging.
- Document version-alignment behavior when telling agents how to use docs.

## Product Naming

Refer to this package as "the NVIDIA NeMo Guardrails library".
Follow `docs/.cursor/rules/product-names/RULE.mdc`.

## Publishing And Preview

Use `make docs-fern-preview-watch` when a branch preview is needed.
Publishing to staging or public instances is maintainer-controlled.
Do not publish docs unless the user explicitly requests it and repository policy allows it.

## Validation Before Handoff

Run the smallest meaningful checks:

- `poetry run pre-commit run --files <changed files>` for docs-only changed files when practical.
- `make docs-fern` when rendering, navigation, Fern components, examples, links, SDK reference generation, or docs configuration may be affected.
- `make docs-check-links` when link changes are broad or risky.

Report any skipped validation and residual risk clearly.

## Related Skills

- Use `guardrails-developer-guide` for product-usage questions.
- Use `guardrails-contributor-dev-setup` for local environment setup.
- Use `guardrails-contributor-create-pr` when drafting PR text and verification summaries for a human to submit.

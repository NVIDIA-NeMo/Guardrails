---
name: "guardrails-skills-guide"
description: "Start here. Lists the NVIDIA NeMo Guardrails library agent skills and explains when to use each one. Use when discovering Guardrails agent skills, choosing a skill, or orienting an agent in this repository. Trigger keywords - guardrails skills, agent skills, what can I do, help, guide, index, overview, start here."
license: "Apache-2.0"
---

# Guardrails Skills Guide

The NVIDIA NeMo Guardrails library agent skills are organized around two paths:

- Product users who installed the Python package and need help using the library.
- Contributors who cloned this repository and need help following project workflow and implementation rules.

For product usage, prefer the canonical documentation instead of duplicating instructions in skills.
The `guardrails-docs` skill routes the agent to the docs MCP server, `llms.txt`, and clean Markdown pages.

## Skill Catalog

| Skill | Audience | Summary |
| --- | --- | --- |
| `guardrails-docs` | Users and developers building with the library | Routes product-usage questions to the canonical documentation through MCP, `llms.txt`, per-page Markdown, or local docs fallback. |
| `guardrails-contributor-dev-setup` | Repository contributors | Sets up and verifies the local development environment, dependencies, hooks, and validation commands. |
| `guardrails-contributor-docs` | Documentation contributors | Guides Fern docs edits, navigation, previews, validation, custom components, and docs-agent entry points. |
| `guardrails-contributor-create-pr` | Repository contributors | Prepares PR-ready text and creates PRs only after explicit user permission and the repository issue gate. |

## Choose a Skill

Use `guardrails-docs` for questions such as:

- How do I install the NVIDIA NeMo Guardrails library?
- Which guardrail should I use?
- How do I configure input, output, retrieval, dialog, or execution rails?
- How do I write Colang flows?
- How do I use Python APIs, LangChain, LangGraph, the server, evaluation, tracing, metrics, or deployment docs?

Use `guardrails-contributor-create-pr` when preparing contribution text for a human to submit.
The skill may create PRs only after the user explicitly asks, the linked issue is triaged and assigned, and the user gives final confirmation.
Agents must not open issues through automation.

Use `guardrails-contributor-dev-setup` when setting up a clone, installing dependencies, running local validation, or diagnosing missing development tools.

Use `guardrails-contributor-docs` when editing docs, Fern configuration, docs navigation, custom MDX components, docs MCP guidance, or starter prompts.

When changing package runtime code, follow `nemoguardrails/AGENTS.md` for public API, provider, integration, and test invariants.

## Related Repository Instructions

- Root repository rules live in `AGENTS.md`.
- Package-specific runtime, public API, and provider-integration invariants live in `nemoguardrails/AGENTS.md`.
- Documentation rules live in `docs/AGENTS.md`.
- Public contribution workflow is canonical in `CONTRIBUTING.md`.
- AI-assisted contribution policy is canonical in `AI_POLICY.md`.

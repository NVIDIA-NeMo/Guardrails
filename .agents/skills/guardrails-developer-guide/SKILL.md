---
name: "guardrails-developer-guide"
description: "Routes NVIDIA NeMo Guardrails library product-usage questions to the canonical documentation. Use when users ask how to install, configure, integrate, evaluate, observe, deploy, troubleshoot, or use the NVIDIA NeMo Guardrails library. Trigger keywords - install guardrails, configure rails, guardrail catalog, Colang, Python API, LangChain, LangGraph, server, evaluate guardrails, tracing, metrics, Docker, troubleshooting."
license: "Apache-2.0"
---

# Guardrails Developer Guide

Use this skill for product-usage questions about the NVIDIA NeMo Guardrails library.
Do not restate full product documentation in this skill.
Route the agent to the canonical docs and summarize the relevant guidance for the user's task.

## Documentation Source Rule

Always use the Markdown documentation under `https://docs.nvidia.com/nemo/guardrails/`.
Use `llms.txt` and page URLs ending in `.md` when loading documentation for agent context.
When presenting references or citations to users, use the canonical human-readable docs links without `.md`.

## Retrieval Order

1. Prefer the docs MCP server when the client supports MCP.
   Use the NVIDIA NeMo Guardrails library docs MCP server documented on the published docs site.
2. If MCP is not available, fetch the docs index:

   ```text
   https://docs.nvidia.com/nemo/guardrails/llms.txt
   ```

3. Use the index to locate the relevant page, then fetch the clean Markdown form of that page by using the page URL with `.md`.
4. If the user is working in a cloned repository and remote docs are unavailable, fall back to local `docs/**/*.mdx`.
5. If the user has the package installed, align docs to the installed `nemoguardrails` version when versioned docs are available.
   If the version cannot be determined, ask whether to use the latest docs.

## Do Not Hardcode Staging

Use production docs as the canonical source.
Use staging URLs only when the user explicitly asks to inspect staging or when validating migration behavior.

## Intent Routing

Use this table to find the right docs area quickly.

| User intent | Docs area |
| --- | --- |
| Install or verify environment | Get Started → Installation |
| Add harmful-content, jailbreak, topic, PII, self-check, fact-check, or agentic security rails | Configure Guardrails → Guardrail Catalog |
| Configure `config.yml`, models, prompts, tracing, streaming, or exceptions | Configure Guardrails → YAML schema and configuration reference |
| Write or debug Colang flows | Configure Guardrails → Colang |
| Use Python APIs | Run Guardrailed Inference → Python API |
| Run the Guardrails API server or actions server | Run Guardrailed Inference → Guardrails API Server |
| Integrate with LangChain, LangGraph, RunnableRails, or tools | Integration with Third-Party Libraries |
| Evaluate guardrails or run vulnerability scanning | Evaluation |
| Configure tracing, metrics, or logging | Observability |
| Deploy with Docker or NeMo microservice | More Deployment Options |
| Troubleshoot errors | Troubleshooting |
| Understand telemetry and privacy | Resources → Telemetry and Privacy |

## Security And Credential Handling

- Never ask users to paste real API keys, tokens, passwords, or provider credentials into chat.
- Use placeholders such as `<NVIDIA_API_KEY>`, `<OPENAI_API_KEY>`, or `<YOUR_ENDPOINT>` in examples.
- Explain where users should set secrets locally, such as shell environment variables, secret managers, local config, or provider dashboards.
- Do not print, store, or echo secrets in generated commands or summaries.

## Response Style

- Start with the user's immediate task and the relevant doc source.
- Give the smallest working path first.
- Add production hardening, optional extras, or alternative integrations only when they are relevant.
- When examples use live providers, remind contributors that tests must mock LLM and provider calls.

## Related Skills

- Use `guardrails-developer-create-guardrails` when creating or modifying a guardrails configuration.
When editing this repository, follow `AGENTS.md` and any subtree `AGENTS.md` files that apply.

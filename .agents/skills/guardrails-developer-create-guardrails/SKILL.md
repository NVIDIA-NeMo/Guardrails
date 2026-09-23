---
name: "guardrails-developer-create-guardrails"
description: "Helps developers create a NeMo Guardrails configuration for an LLM application. Use when users want to build, scaffold, configure, test, or iterate on input, output, retrieval, dialog, execution, Colang, or catalog-based guardrails. Trigger keywords - create guardrails, build guardrails, scaffold config, write rails, create config.yml, add input rails, add output rails, Colang flow, guardrails config, test guardrails."
license: "Apache-2.0"
---

# Create Guardrails

Use this skill when a developer wants help creating a guardrails configuration, not just reading documentation.
The goal is to produce a small, working configuration first, then iterate based on the user's risk, model, app, and test cases.

Use `guardrails-developer-guide` to look up canonical docs when needed.
Do not duplicate full docs in this skill.

## Documentation Source Rule

When using NVIDIA NeMo Guardrails library documentation, use the Markdown documentation under `https://docs.nvidia.com/nemo/guardrails/`.
Use `llms.txt` and page URLs ending in `.md` when loading documentation for agent context.
When presenting references or citations to users, use the canonical human-readable docs links without `.md`.

## First Questions

Ask only what you need to choose a starting path:

1. What kind of application are you guarding?
2. Which model/provider or framework are you using?
3. Which risk do you want to handle first?
4. Do you want a quick catalog-based guardrail, a Colang flow, or a Python integration?

If the user is unsure, recommend starting with the smallest working input/output rail and one concrete test prompt.

## Choose The Starting Pattern

| User goal | Starting pattern |
| --- | --- |
| Block harmful content | Content safety input/output rails |
| Restrict topics | Topic control or topical rails |
| Detect jailbreaks | Jailbreak protection or heuristics |
| Mask or detect sensitive data | PII detection rails |
| Reduce hallucinations in RAG | Retrieval/output fact-checking rails |
| Control conversation flow | Colang dialog flows |
| Guard tool calls or actions | Execution rails and action validation |
| Integrate with LangChain or LangGraph | RunnableRails, middleware, or documented integration path |

Route to the relevant docs page through `guardrails-developer-guide` before filling in details that depend on the current docs.

## Create A Minimal Config

Prefer a standard config folder layout:

```text
config/
  config.yml
  prompts.yml
  rails.co
  actions.py
```

Only create files that are needed:

- Use `config.yml` for models, rails, streaming, tracing, and configuration.
- Use `prompts.yml` when the selected rail needs custom prompt templates.
- Use `.co` files when the solution needs Colang flows.
- Use `actions.py` only when Python actions are required.

When editing an existing app, preserve the user's project layout and avoid moving unrelated files.

## Build Iteratively

1. Start with one guardrail objective.
2. Write the smallest config that exercises that objective.
3. Add two or three test prompts:
   - a request that should pass,
   - a request that should be blocked or modified,
   - an edge case if the user has one.
4. Run the config through the documented Python API, CLI chat, or server path that matches the user's setup.
5. Inspect the result and adjust the rail, prompt, flow, or model configuration.

Do not silently introduce live provider calls.
Ask before running commands that require network access, credentials, paid APIs, Docker, or long-running services.

## Testing And Verification

For product users, verify with the smallest runnable example:

- `nemoguardrails chat --config <config-path>` when using the CLI.
- A short Python script with `RailsConfig.from_path(...)` and `LLMRails(...)` when embedding in an app.
- The documented server endpoints when using the Guardrails API server.

For repository contributors, unit tests must not call live LLM or provider services.
Use repository test doubles and mocks according to `nemoguardrails/AGENTS.md`.

## Security And Credentials

- Never ask users to paste real API keys, tokens, or provider credentials into chat.
- Use placeholders such as `<NVIDIA_API_KEY>`, `<OPENAI_API_KEY>`, and `<YOUR_ENDPOINT>`.
- Explain where secrets should be set locally.
- Do not write secrets into committed config examples.

## Output Format

When helping create guardrails, return:

1. The chosen starting pattern and why.
2. The files to create or edit.
3. The proposed config or code snippets.
4. The verification command or script.
5. The test prompts and expected behavior.
6. Follow-up improvements after the first working version.

## Related Skills

- Use `guardrails-developer-guide` for documentation lookup and product-usage questions.
When editing this repository, follow `AGENTS.md` and any subtree `AGENTS.md` files that apply.

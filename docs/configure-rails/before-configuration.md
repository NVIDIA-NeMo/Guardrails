---
title: Before You Begin
description: Prerequisites and decisions to make before configuring the NeMo Guardrails toolkit.
---

# Before You Begin Configuring Rails

Before configuring your guardrails, ensure you have the following components ready.

## Required: LLM Backend

You need a main LLM hosted and accessible via API. This LLM handles:

- Generating responses to user queries

**Options:**

| Provider | Requirements |
|----------|--------------|
| NVIDIA NIM | Deploy NIM and note the API endpoint |
| OpenAI | Obtain API key |
| Azure OpenAI | Configure Azure endpoint and API key |
| Other providers | Refer to [Supported LLMs](../supported-llms.md) |

**What you need:**

- [ ] LLM API endpoint URL
- [ ] Authentication credentials (API key or token)

## Recommended: Safety Models (NemoGuard NIMs)

For production deployments, deploy dedicated safety models to offload guardrail checks from the main LLM:

| NemoGuard Model | Purpose |
|-----------------|---------|
| Content Safety | Detect harmful or inappropriate content |
| Jailbreak Detection | Block adversarial prompt attacks |
| Topic Control | Keep conversations on-topic |

**What you need:**

- [ ] NemoGuard NIM endpoint URLs
- [ ] KV cache enabled for better performance (recommended)

:::{tip}
If you use NVIDIA NIM for LLMs and LLM-based NemoGuard NIMs, KV cache helps reduce latency for sequential guardrail checks. To learn more about KV cache, see the [KV Cache Reuse](https://docs.nvidia.com/nim/large-language-models/latest/kv-cache-reuse.html) guide in the NVIDIA NIM documentation.
:::

## Optional: Knowledge Base Documents

If using RAG (Retrieval-Augmented Generation) for grounded responses:

- [ ] Prepare documents in markdown format (`.md` files)
- [ ] Organize documents in a `kb/` folder

## Optional: Advanced Components

For advanced use cases such as implementing your own custom scripts or guardrails, prepare the following as needed:

| Component | Purpose | Format |
|-----------|---------|--------|
| **Custom Actions** | External API calls, validation logic | Python functions in `actions.py` |
| **Custom Initialization** | Register custom LLM/embedding providers | Python code in `config.py` |
| **Custom Prompts** | Override default guardrails prompts | YAML in `config.yml` |

## Checklist Summary

**Before starting configuration:**

- [ ] Main LLM endpoint and credentials ready
- [ ] (Recommended) NemoGuard NIM endpoints deployed
- [ ] (Optional) Knowledge base documents prepared
- [ ] (Optional) Custom action requirements identified

## Next Steps

Once you have these components ready, proceed to:

- [Configuration Overview](index.md) - Create your configuration files
- [Core Configuration](yaml-schema/index.md) - Configure `config.yml`

If you need tutorials to understand how to use the NeMo Guardrails toolkit, revisit the [Get Started](../getting-started/index.md) section.

---
title: Before You Begin
description: Prerequisites and decisions to make before configuring the NeMo Guardrails Library.
---

# Before You Begin Configuring Rails

This Configure Rails chapter thoroughly describes how to prepare guardrails configuration files.
This page covers the prerequisites and decisions to make before you begin working on guardrails configurations.

## Checklist Summary

Use the following checklist to ensure that you have all the necessary components ready before you begin configuring guardrails.

- [ ] (Required) Main LLM endpoint and credentials ready. Refer to [](#hosted-llm-for-the-main-llm) for more details.
- [ ] (Recommended) NemoGuard NIM endpoints deployed. Refer to [](#nemoguard-nim-microservices) for more details.
- [ ] (Optional) Knowledge base documents prepared. Refer to [](#knowledge-base-documents) for more details.
- [ ] (Optional) Custom action requirements identified. Refer to [](#advanced-components) for more details.

Each item in the checklist is described in detail in the following sections.

## Hosted LLM for the Main LLM

You need a main LLM hosted and accessible via API. This LLM handles the conversation by generating responses to user queries.

**Options:**

| Provider | Requirements |
|----------|--------------|
| NVIDIA NIM | Deploy NIM and note the API endpoint |
| OpenAI | Obtain API key |
| Azure OpenAI | Configure Azure endpoint and API key |
| Other providers | Refer to [Supported LLMs](../supported-llms.md) |

**Checklist of what you need:**

- [ ] LLM API endpoint URL, either locally, on NVIDIA API Catalog, or on the third-party providers
- [ ] Authentication credentials (API key or token)

## Nemotron NIM Microservices

Deploy dedicated safety models to offload guardrail checks from the main LLM:

| Nemotron Model | Purpose |
|-----------------|---------|
| Content Safety | Detect harmful or inappropriate content |
| Jailbreak Detect | Block adversarial prompt attacks |
| Topic Control | Keep conversations on-topic |

**Checklist of what you need:**

- [ ] Nemotron NIM endpoint URLs, either locally or on NVIDIA API Catalog
- [ ] KV cache enabled for better performance (recommended)

:::{tip}
If you use NVIDIA NIM for LLMs and LLM-based Nemotron NIMs, KV cache helps reduce latency for sequential guardrail checks. To learn more about KV cache, see the [KV Cache Reuse](https://docs.nvidia.com/nim/large-language-models/latest/kv-cache-reuse.html) guide in the NVIDIA NIM documentation.
:::

## Knowledge Base Documents

If using RAG (Retrieval-Augmented Generation) for grounded responses (i.e. Retrieval rails):

- [ ] Prepare documents in markdown format (`.md` files)
- [ ] Organize documents in a `kb/` folder

## Advanced Components

For advanced use cases such as implementing your own custom scripts or guardrails, prepare the following as needed:

| Component | Purpose | Format |
|-----------|---------|--------|
| **Custom Actions** | External API calls, validation logic | Python functions in `actions.py` |
| **Custom Initialization** | Register custom LLM/embedding providers | Python code in `config.py` |
| **Custom Prompts** | Override default guardrails prompts | YAML in `config.yml` |

## Next Steps

Once you have these components ready, proceed to the next section [Configuration Overview](index.md) to start organizing your guardrails configuration files.

If you need tutorials to understand how to use the NeMo Guardrails Library, revisit the [Get Started](../getting-started/index.md) section.

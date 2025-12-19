---
title:
  page: "Supported LLMs"
  nav: "Supported LLMs"
description: "Connect to NVIDIA NIM, OpenAI, Azure, Anthropic, HuggingFace, and LangChain providers."
topics: ["AI"]
tags: ["LLM", "NVIDIA NIM", "OpenAI", "LangChain", "Embeddings"]
content:
  type: "Reference"
  difficulty: "Beginner"
  audience: ["Developer", "Machine Learning Engineer"]
---

# Supported LLMs

The NeMo Guardrails library supports a wide range of LLM providers and their models.

## LLM Providers

The NeMo Guardrails library supports the following LLM providers:

### NVIDIA NIM

The NeMo Guardrails library supports NVIDIA NIM microservices for local deployment and NVIDIA API Catalog for hosted models.

- **Locally-deployed LLM NIM Microservices**: LLMs deployed on your own infrastructure.
- **NVIDIA API Catalog**: Hosted LLMs on [build.nvidia.com](https://build.nvidia.com/models).
- **Specialized NIM Microservices**: Nemo Content Safety, NeMo Topic Control, and NeMo Jailbreak Detect.

### External LLM Providers

The NeMo Guardrails library supports the following external LLM providers:

- OpenAI
- Azure OpenAI
- Anthropic
- Cohere
- Google Vertex AI

### Self-Hosted

The NeMo Guardrails library supports the following self-hosted LLM providers:

- HuggingFace Hub
- HuggingFace Endpoints
- vLLM
- Generic

### Providers from LangChain Community

The NeMo Guardrails library supports any LLM provider from the LangChain Community. Refer to [All integration providers](https://docs.langchain.com/oss/python/integrations/providers/all_providers) in the LangChain documentation.

## Embedding Providers

The NeMo Guardrails library supports the following embedding providers:

- NVIDIA NIM
- FastEmbed
- OpenAI

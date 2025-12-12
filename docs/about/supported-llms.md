---
title: Supported LLMs and Providers
description: Browse the LLMs and their providers supported by the NeMo Guardrails Library.
---

# Supported LLMs

The NeMo Guardrails Library supports a wide range of LLM providers and their models.

## LLM Providers

The NeMo Guardrails Library supports the following LLM providers:

### NVIDIA NIM

The NeMo Guardrails Library supports NVIDIA NIM microservices for local deployment and NVIDIA API Catalog for hosted models.

- **Locally-deployed LLM NIM Microservices**: LLMs deployed on your own infrastructure.
- **NVIDIA API Catalog**: Hosted LLMs on [build.nvidia.com](https://build.nvidia.com/models).
- **Specialized NIM Microservices**: Nemo Content Safety, NeMo Topic Control, and NeMo Jailbreak Detect.

### External LLM Providers

The NeMo Guardrails Library supports the following external LLM providers:

- OpenAI
- Azure OpenAI
- Anthropic
- Cohere
- Google Vertex AI

### Self-Hosted

The NeMo Guardrails Library supports the following self-hosted LLM providers:

- HuggingFace Hub
- HuggingFace Endpoints
- vLLM
- Generic

### Providers from LangChain Community

The NeMo Guardrails Library supports any LLM provider from the LangChain Community. Refer to [All integration providers](https://docs.langchain.com/oss/python/integrations/providers/all_providers) in the LangChain documentation.

## Embedding Providers

The NeMo Guardrails Library supports the following embedding providers:

- NVIDIA NIM
- FastEmbed
- OpenAI

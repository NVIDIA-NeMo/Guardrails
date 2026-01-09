---
title:
  page: "Supported LLMs"
  nav: "Supported LLMs"
description: "Connect to NVIDIA NIM, OpenAI, Azure, Anthropic, HuggingFace, and LangChain providers."
keywords: ["llm providers", "nvidia nim", "openai", "langchain", "embedding providers"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "pretrained_models", "nlp"]
content:
  type: reference
  difficulty: technical_beginner
  audience: [engineer, data_scientist]
---

# Supported LLMs

The NeMo Guardrails library supports a wide range of LLM providers and models. This includes base models, instruct-tuned, and reasoning models. These models can be served locally on the same machine as NeMo Guardrails, or at a remote endpoint accessible from Guardrails over a network. This flexible approach allows Guardrails to be used for a range of applications: from edge deployments on resource-constrained devices, to horizontally-scalable backend clusters.

## LLM Types

Integrating NeMo Guardrails improves safety and security of an Application LLM, which is responsible for generating responses to the end-user. NeMo Guardrails can also use the same Application LLM to run guardrails, simplifying deployments and reducing friction to on-ramp. Two examples of this are self-check rails and dialog rails. Self-check rails use the Application LLM to decide whether a user request or LLM response is safe. Dialog rails use the Application LLM to guide the user through a pre-defined conversational flow.

NeMo Guardrails can also call models for a specific guardrail on behalf of the client. Having guardrail-specific models allows the use of smaller fine-tuned models, which are specialized on the guardrails task. For example the NVIDIA Nemoguard collection of models includes [content-safety](https://build.nvidia.com/nvidia/llama-3_1-nemotron-safety-guard-8b-v3), [topic-control](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-topic-control), and [jailbreak-detect](https://build.nvidia.com/nvidia/nemoguard-jailbreak-detect) models. These models can be accessed on [build.nvidia.com](https://build.nvidia.com/) for rapid prototyping, or on [NGC Catalog](https://catalog.ngc.nvidia.com/) for deployment with NIM Docker containers.

## Application LLM Providers

The NeMo Guardrails library supports major LLM providers, including:

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

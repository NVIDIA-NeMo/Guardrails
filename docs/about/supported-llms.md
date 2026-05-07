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

## Routing matrix

Starting with 0.22, NeMo Guardrails routes every model through one of two LLM frameworks:

- **DefaultFramework** speaks OpenAI's wire protocol directly over `httpx`. It is the primary path for any provider whose endpoint is OpenAI-compatible, and works out of the box with `pip install nemoguardrails`.
- **LangChain** is the fallback for providers whose API genuinely is not OpenAI-compatible. It requires installing LangChain plus the matching `langchain-*` provider package.

The table below maps each engine to the framework that handles it. Engines listed under `DefaultFramework` need no extra install. Engines listed under `LangChain` require `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain` and the corresponding provider package.

| Engine | Framework | Streaming | Tool calls | Reasoning models | Notes |
|---|---|---|---|---|---|
| `openai` | DefaultFramework | yes | yes | yes | OpenAI public API or any OpenAI-compatible endpoint via `parameters.base_url`. |
| `nim` | DefaultFramework | yes | yes | yes | Default base URL `https://integrate.api.nvidia.com/v1`. |
| `nvidia_ai_endpoints` | DefaultFramework | yes | yes | yes | Alias for `nim`. |
| `ollama` | DefaultFramework | yes | yes | n/a | Default base URL `http://localhost:11434/v1`. |
| vLLM, TGI, OpenRouter, Together.ai, Fireworks.ai, Groq, DeepSeek, llama.cpp, ... | DefaultFramework | yes | yes | yes (where supported) | Use `engine: openai` plus `parameters.base_url` (and `parameters.api_key`). The legacy `engine: vllm_openai` is LangChain-only and is not recommended for new configs. |
| `anthropic` | LangChain | yes | yes | via wrapper | Requires `pip install langchain langchain-anthropic`. |
| `cohere` | LangChain | yes | yes | n/a | Requires `pip install langchain langchain-cohere`. |
| `google_genai`, `vertexai` | LangChain | yes | yes | n/a | Requires the matching `langchain-google-*` package. |
| `azure` (also `azure_openai`, `azure_ai`) | LangChain | yes | yes | yes | Azure OpenAI's deployment-name URL pattern plus `api-version` query string is not handled by DefaultFramework's OpenAI-compatible client. Requires `langchain-openai`. |
| `huggingface_endpoint` | LangChain | varies | varies | varies | Default text-generation schema. If your endpoint exposes `/v1/chat/completions`, prefer `engine: openai` plus `parameters.base_url` instead. |
| `huggingface_pipeline`, `huggingface_hub`, `trt_llm`, `self_hosted` | LangChain | varies | varies | varies | In-process pipelines and LangChain wrappers without a native HTTP path. |
| `vllm_openai`, `deepseek`, other legacy LangChain wrappers | LangChain | yes | yes | yes | Legacy LangChain provider engines (the `deepseek` row above is the same DeepSeek hosted endpoint, reachable as `engine: openai` plus `parameters.base_url` under DefaultFramework). They continue to work under the LangChain framework; for new configs, prefer `engine: openai` plus `parameters.base_url` (DefaultFramework) when the wire is OpenAI-compatible. |
| Other LangChain providers | LangChain | varies | varies | varies | Any community provider exposed through LangChain's chat-model integrations. |

There is no automatic fallback from `DefaultFramework` to `LangChain`. To use a LangChain-only engine, install LangChain and the provider package, then set `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain`. Engine names in `config.yml` stay bare (`engine: anthropic`, `engine: cohere`, ...) — there is no `langchain/<provider>` prefix syntax. For details and examples, see [Upgrading to 0.22: LLM Framework Transition](../upgrade/0.22-framework-transition.md).

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

### Providers from LangChain

The NeMo Guardrails library supports LLM providers from the LangChain Community, including both text completion and chat completion providers. Refer to [Chat model integrations](https://python.langchain.com/docs/integrations/chat/) in the LangChain documentation. You can also use the [`nemoguardrails find-providers`](find-providers-command) CLI command to discover available providers.

## Embedding Providers

The NeMo Guardrails library supports the following embedding providers:

- NVIDIA NIM
- NVIDIA AI Endpoints
- FastEmbed
- OpenAI
- Azure OpenAI
- Cohere
- SentenceTransformers
- Google

For more information on configuring embedding providers, refer to [Embedding Search Providers](../configure-rails/other-configurations/embedding-search-providers.md).

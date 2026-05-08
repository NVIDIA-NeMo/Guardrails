---
title:
  page: "Supported LLMs"
  nav: "Supported LLMs"
description: "Connect to NVIDIA NIM, OpenAI, Azure, Anthropic, Hugging Face, and LangChain providers."
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

## Inference Providers

```{raw} html
<button type="button" class="table-expand-button" data-table-title="Routing Matrix">
  <span aria-hidden="true" class="table-expand-button__icon">&#x26F6;</span>
  Expand table
</button>
```

| Engine | Framework Routing\* | Streaming | Tool calls | Reasoning models | Notes |
| --- | --- | --- | --- | --- | --- |
| `anthropic` | LangChainFramework | yes | yes | wrapper-dependent | Requires `pip install langchain langchain-anthropic`. |
| `azure`, `azure_openai` | LangChainFramework | yes | yes | yes | Azure OpenAI's deployment-name URL pattern plus `api-version` query string is not handled by DefaultFramework's OpenAI-compatible client. Requires `langchain-openai`. |
| `cohere` | LangChainFramework | yes | yes | n/a | Requires `pip install langchain langchain-cohere`. |
| `google_genai` | LangChainFramework | yes | yes | n/a | Requires `pip install langchain langchain-google-genai`. |
| `huggingface_endpoint` | LangChainFramework | varies | varies | varies | Default text-generation schema. If your endpoint exposes `/v1/chat/completions`, prefer `engine: openai` with `parameters.base_url` instead. |
| `huggingface_pipeline`, `huggingface_hub`, `trt_llm`, `self_hosted` | LangChainFramework | varies | varies | varies | In-process pipelines and LangChain wrappers without a native HTTP path. |
| `nim` | DefaultFramework | yes | yes | yes | Default base URL `https://integrate.api.nvidia.com/v1`. |
| `nvidia_ai_endpoints` | DefaultFramework | yes | yes | yes | Alias for `nim`. |
| `ollama` | DefaultFramework | yes | yes | yes (where supported) | Default base URL `http://localhost:11434/v1`. |
| `openai` | DefaultFramework | yes | yes | yes | OpenAI public API or any OpenAI-compatible endpoint using `parameters.base_url`. For vLLM, TGI, OpenRouter, Together.ai, Fireworks.ai, Groq, DeepSeek, llama.cpp, NVIDIA Nemotron, and similar providers, use `engine: openai` with `parameters.base_url` and `parameters.api_key`. |
| `vertexai` | LangChainFramework | yes | yes | n/a | Requires `pip install langchain langchain-google-vertexai`. |
| `vllm_openai`, `deepseek` | LangChainFramework | yes | yes | yes | Legacy LangChain provider engines. They continue to work under the LangChain framework. For new configurations, use `engine: openai` with `parameters.base_url` when the wire protocol is OpenAI-compatible. |
| `<provider_name>` | LangChainFramework | varies | varies | varies | Any community provider exposed through LangChain's chat-model integrations. Use the bare provider name as the engine name. |

\* Starting with version 0.22, the NVIDIA NeMo Guardrails library routes every model through one of the following LLM frameworks:

- `DefaultFramework` uses OpenAI's wire protocol directly over `httpx`. It is the primary path for providers whose endpoints are OpenAI-compatible.
- `LangChainFramework` supports providers whose APIs are not OpenAI-compatible.

For framework selection rules and migration examples, refer to [LLM Framework Routing](../configure-rails/yaml-schema/llm-framework-routing.md).

## LangChain-Backed Providers

The NeMo Guardrails library supports LLM providers from the LangChain Community, including both text completion and chat completion providers. Refer to [Chat model integrations](https://python.langchain.com/docs/integrations/chat/) in the LangChain documentation. You can also use the [`nemoguardrails find-providers`](find-providers-command) CLI command to discover available providers.

## Embedding Model Providers

The NeMo Guardrails library uses embedding models for vector similarity search in dialog rails, `embeddings_only` intent matching, and knowledge base retrieval. The following table lists the supported embedding model providers and their corresponding engine names.

| Provider | Engine | Notes |
| --- | --- | --- |
| NVIDIA NIM | `nim` | NVIDIA NIM microservices |
| NVIDIA AI Endpoints | `nvidia_ai_endpoints` | Alias for `nim` |
| FastEmbed | `fastembed` | FastEmbed embedding model provider |
| OpenAI | `openai` | OpenAI embedding model provider |
| Azure OpenAI | `azure` | Azure OpenAI embedding model provider |
| Cohere | `cohere` | Cohere embedding model provider |
| SentenceTransformers | `sentence_transformers` | SentenceTransformers embedding model provider |
| Google | `google` | Google embedding model provider |

For more information on configuring embedding providers, refer to [Embedding Search Providers](../configure-rails/other-configurations/embedding-search-providers.md).

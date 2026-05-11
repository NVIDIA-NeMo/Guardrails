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

Each engine is served by a framework that manages the underlying HTTP or SDK calls. NeMo Guardrails ships with a built-in framework that talks to OpenAI-compatible endpoints over `httpx` with no LangChain dependency. For engines whose API is not OpenAI-compatible, opt into the LangChain framework by setting `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain` and installing the matching `langchain-<provider>` package. To add a custom framework, implement the `LLMFramework` protocol from `nemoguardrails.types`.

```{raw} html
<button type="button" class="table-expand-button" data-table-title="Inference Providers">
  <span aria-hidden="true" class="table-expand-button__icon">&#x26F6;</span>
  Expand table
</button>
```

| Engine | Framework | Streaming | Tool calls | Reasoning models | Notes |
| --- | --- | --- | --- | --- | --- |
| `anthropic` | LangChain (opt-in) | yes | yes | wrapper-dependent | Requires `pip install langchain langchain-anthropic`. |
| `azure`, `azure_openai` | LangChain (opt-in) | yes | yes | yes | Azure OpenAI is OpenAI-compatible at the wire level. The LangChain path (`langchain-openai`) is the convenient default because it handles the deployment-name URL pattern and `api-version` query string for you. Azure is also reachable through the built-in client by setting `parameters.base_url` to the deployment URL and passing `api-version` via `default_query` and `api-key` via `default_headers`. |
| `cohere` | LangChain (opt-in) | yes | yes | n/a | Requires `pip install langchain langchain-cohere`. |
| `google_genai` | LangChain (opt-in) | yes | yes | n/a | Requires `pip install langchain langchain-google-genai`. |
| `huggingface_endpoint` | LangChain (opt-in) | varies | varies | varies | Default text-generation schema. If your endpoint exposes `/v1/chat/completions`, prefer `engine: openai` with `parameters.base_url` instead. |
| `huggingface_pipeline`, `huggingface_hub`, `trt_llm`, `self_hosted` | LangChain (opt-in) | varies | varies | varies | In-process pipelines and LangChain wrappers without a native HTTP path. |
| `nim` | Built-in | yes | yes | yes | Default base URL `https://integrate.api.nvidia.com/v1`. |
| `nvidia_ai_endpoints` | Built-in | yes | yes | yes | Alias for `nim`. |
| `ollama` | Built-in | yes | yes | yes (where supported) | Default base URL `http://localhost:11434/v1`. |
| `openai` | Built-in | yes | yes | yes | OpenAI public API or any OpenAI-compatible endpoint using `parameters.base_url`. For vLLM, TGI, OpenRouter, Together.ai, Fireworks.ai, Groq, DeepSeek, llama.cpp, NVIDIA Nemotron, and similar providers, use `engine: openai` with `parameters.base_url` and `parameters.api_key`. |
| `vertexai` | LangChain (opt-in) | yes | yes | n/a | Requires `pip install langchain langchain-google-vertexai`. |
| `vllm_openai`, `deepseek` | LangChain (opt-in) | yes | yes | yes | Legacy LangChain provider engines. They continue to work when you opt into LangChain. For new configurations, use `engine: openai` with `parameters.base_url` when the wire protocol is OpenAI-compatible. |
| `<provider_name>` | LangChain (opt-in) | varies | varies | varies | Any community provider exposed through LangChain's chat-model integrations. Use the bare provider name as the engine name. |

For migration recipes between the built-in path and the LangChain path, see [Migrating to 0.22](../migration/0.22.md).

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

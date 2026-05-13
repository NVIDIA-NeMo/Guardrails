---
title:
  page: Model Configuration for NeMo Guardrails
  nav: Models
description: Configure LLM engines, embedding models, and task-specific models in config.yml.
topics:
- Configuration
- AI Safety
tags:
- Models
- LLM
- Embeddings
- OpenAI
- NIM
- YAML
content:
  type: reference
  difficulty: technical_intermediate
  audience:
  - engineer
  - AI Engineer
---

# Model Configuration

In this section, learn how to configure the models used in your guardrails configuration. For a complete reference of all configuration options, refer to the [](../configuration-reference.md).

## NVIDIA NIM Configuration

The NeMo Guardrails library provides seamless integration with NVIDIA NIM microservices:

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama-3.1-8b-instruct
```

This provides access to:

- **Locally-deployed NIMs**: Run models on your own infrastructure with optimized inference.
- **NVIDIA API Catalog**: Access hosted models on [build.nvidia.com](https://build.nvidia.com/models).
- **Specialized NIMs**: NemoGuard Content Safety, Topic Control, and Jailbreak Detect.

### Local NIM Deployment

For locally-deployed NIMs, specify the base URL:

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama-3.1-8b-instruct
    parameters:
      base_url: http://localhost:8000/v1
```

---

## Task-Specific Models

Configure different models for specific tasks:

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama-3.1-8b-instruct

  - type: self_check_input
    engine: nim
    model: meta/llama3-8b-instruct

  - type: self_check_output
    engine: nim
    model: meta/llama-3.1-70b-instruct

  - type: generate_user_intent
    engine: nim
    model: meta/llama-3.1-8b-instruct
```

---

## Configuration Examples

### OpenAI

The following example shows how to configure the OpenAI model as the main application LLM:

```yaml
models:
  - type: main
    engine: openai
    model: gpt-4o
```

### Azure OpenAI

The following example shows how to configure the Azure OpenAI model as the main application LLM using the Azure OpenAI API:

```yaml
models:
  - type: main
    engine: azure
    model: gpt-4
    parameters:
      azure_deployment: my-gpt4-deployment
      azure_endpoint: https://my-resource.openai.azure.com
```

```{note}
Azure OpenAI is OpenAI-compatible at the wire level, but the LangChain path is the convenient default because `langchain-openai` handles the deployment-name URL pattern and `api-version` query string for you. Set `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain` and install `langchain-openai`. Azure is also reachable through the built-in client with manual plumbing; see [Migrating to 0.22](../../migration/0.22.md#azure-openai).
```

### Anthropic

The following example shows how to configure the Anthropic model as the main application LLM:

```yaml
models:
  - type: main
    engine: anthropic
    model: claude-3-5-sonnet-20241022
```

```{note}
Anthropic's API isn't OpenAI-compatible, so this engine is opt-in: set `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain` and install `langchain-anthropic`. For background, see [Migrating to 0.22](../../migration/0.22.md#using-langchain).
```

### vLLM (OpenAI-Compatible)

vLLM exposes an OpenAI-compatible API, so the recommended configuration uses `engine: openai` pointed at the vLLM endpoint. The built-in client handles it with no LangChain dependency.

```yaml
models:
  - type: main
    engine: openai
    model: meta-llama/Llama-3.1-8B-Instruct
    parameters:
      base_url: http://localhost:5000/v1
      api_key: EMPTY
```

The following example shows how to configure Llama Guard as a guardrail model using the same pattern:

```yaml
models:
  - type: llama_guard
    engine: openai
    model: meta-llama/LlamaGuard-7b
    parameters:
      base_url: http://localhost:5000/v1
      api_key: EMPTY
```

When self-hosted vLLM does not enforce authentication, set `parameters.api_key` to any non-empty placeholder such as `EMPTY`. If your deployment requires a real token, replace `parameters.api_key` with the literal token, or omit it and set `api_key_env_var` at the **top level** of the model entry (not inside `parameters:`):

```yaml
- type: main
  engine: openai
  model: meta-llama/Llama-3.1-8B-Instruct
  api_key_env_var: MY_VLLM_API_KEY
  parameters:
    base_url: http://localhost:5000/v1
```

```{note}
The referenced environment variable must be set before `RailsConfig.from_content` or `RailsConfig.from_path` is called. Otherwise, config loading fails with `Model API Key environment variable 'X' not set.`. This is a Pydantic validator on the model schema; the check is eager, not lazy.
```

```{note}
The legacy `engine: vllm_openai` with `parameters.openai_api_base` form is only needed when running under `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain`. For new configurations, prefer the form above.
```

### Other OpenAI-compatible endpoints

The same `engine: openai` plus `parameters.base_url` pattern works for any provider whose wire protocol is OpenAI-compatible, including OpenRouter, Together.ai, Fireworks.ai, Groq, DeepSeek's hosted API at `https://api.deepseek.com/v1`, TGI deployments that expose `/v1/chat/completions`, and `llama.cpp` server with `--api`. Provide `parameters.base_url` and either `parameters.api_key` or a top-level `api_key_env_var`.

### Google Vertex AI

The following example shows how to configure the Google Vertex AI model as the main application LLM:

```yaml
models:
  - type: main
    engine: vertexai
    model: gemini-1.0-pro
```

```{note}
Vertex AI's API isn't OpenAI-compatible, so this engine is opt-in: set `NEMOGUARDRAILS_LLM_FRAMEWORK=langchain` and install `langchain-google-vertexai`. For background, see [Migrating to 0.22](../../migration/0.22.md#using-langchain).
```

### Complete Example

The following example shows how to configure the main application LLM, embeddings model, and a dedicated NemoGuard model for input and output checking:

```yaml
models:
  # Main application LLM
  - type: main
    engine: nim
    model: meta/llama-3.1-70b-instruct
    parameters:
      temperature: 0.7
      max_tokens: 2000

  # Embeddings for knowledge base
  - type: embeddings
    engine: FastEmbed
    model: all-MiniLM-L6-v2

  # Dedicated model for input checking
  - type: self_check_input
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-content-safety

  # Dedicated model for output checking
  - type: self_check_output
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-content-safety
```

---

## Model Parameters

Pass additional parameters to the underlying LLM client. For engines served by the built-in client (any OpenAI-compatible endpoint), parameters are forwarded to the OpenAI-compatible HTTP request (for example, `temperature`, `max_tokens`, `base_url`, `api_key`, `default_query`, `default_headers`). For LangChain engines, parameters follow the conventions of the underlying LangChain class.

```yaml
models:
  - type: main
    engine: openai
    model: gpt-4
    parameters:
      temperature: 0.7
      max_tokens: 1000
      top_p: 0.9
```

Common parameters vary by provider. For built-in engines, see the OpenAI-compatible client options. For LangChain engines, refer to the corresponding LangChain provider documentation.

---
title:
  page: "Model Configuration for NeMo Guardrails"
  nav: "Models"
description: "Configure LLM engines, embedding models, and task-specific models in config.yml."
topics: ["Configuration", "AI Safety"]
tags: ["Models", "LLM", "Embeddings", "OpenAI", "NIM", "YAML"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
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
- **Specialized NIMs**: Nemotron Content Safety, Topic Control, and Jailbreak Detect.

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

### Anthropic

The following example shows how to configure the Anthropic model as the main application LLM:

```yaml
models:
  - type: main
    engine: anthropic
    model: claude-3-5-sonnet-20241022
```

### vLLM (OpenAI-Compatible)

The following example shows how to configure the vLLM model as the main application LLM using the vLLM OpenAI API:

```yaml
models:
  - type: main
    engine: vllm_openai
    parameters:
      openai_api_base: http://localhost:5000/v1
      model_name: meta-llama/Llama-3.1-8B-Instruct
```

The following example shows how to configure Llama Guard as a guardrail model using the vLLM OpenAI API:

```yaml
models:
  - type: llama_guard
    engine: vllm_openai
    parameters:
      openai_api_base: http://localhost:5000/v1
      model_name: meta-llama/LlamaGuard-7b
```

### Google Vertex AI

The following example shows how to configure the Google Vertex AI model as the main application LLM:

```yaml
models:
  - type: main
    engine: vertexai
    model: gemini-1.0-pro
```

### Complete Example

The following example shows how to configure the main application LLM, embeddings model, and a dedicated Nemotron model for input and output checking:

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

Pass additional parameters to the underlying LangChain class:

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

Common parameters vary by provider. Refer to the LangChain documentation for provider-specific options.

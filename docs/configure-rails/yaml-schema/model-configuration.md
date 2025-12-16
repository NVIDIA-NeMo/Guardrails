---
title: Model Configuration
description: Configure LLM providers, embedding models, and task-specific models in the config.yml file.
---

# Model Configuration

This section describes how to configure LLM models and embedding models in the `config.yml` file.

## The `models` Key

The `models` key defines the LLM providers and models used by the NeMo Guardrails Library.

```yaml
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct
```

| Attribute | Description |
|-----------|-------------|
| `type` | The model type (`main`, `embeddings`, or task-specific types) |
| `engine` | The LLM provider (for example, `openai`, `nim`, `anthropic`) |
| `model` | The model name (for example, `gpt-3.5-turbo-instruct`, `meta/llama-3.1-8b-instruct`) |
| `parameters` | Optional parameters to pass to the LangChain class that is used by the LLM provider. For example, when engine is set to `openai`, the library loads the ChatOpenAI class. The ChatOpenAI class supports `temperature`, `max_tokens`, and other class-specific arguments. |

---

## LLM Engines

### Core Engines

| Engine | Description |
|--------|-------------|
| `openai` | OpenAI models |
| `nim` | NVIDIA NIM microservices |
| `nvidia_ai_endpoints` | Alias for `nim` engine |
| `azure` | Azure OpenAI models |
| `anthropic` | Anthropic Claude models |
| `cohere` | Cohere models |
| `vertexai` | Google Vertex AI |

### Self-Hosted Engines

| Engine | Description |
|--------|-------------|
| `huggingface_hub` | HuggingFace Hub models |
| `huggingface_endpoint` | HuggingFace Inference Endpoints |
| `vllm_openai` | vLLM with OpenAI-compatible API |
| `trt_llm` | TensorRT-LLM |
| `self_hosted` | Generic self-hosted models |

### Auto-Discovered LangChain Providers

The library automatically discovers all LLM providers from LangChain Community at runtime. This includes 50+ additional providers. Use the provider name as the `engine` value in your configuration.

To help you explore and select the right LLM provider, the library CLI provides the [`find-providers`](find-providers-command) command to discover available LLM providers:

```bash
nemoguardrails find-providers [--list]
```

---

## Embedding Engines

| Engine | Description |
|--------|-------------|
| `FastEmbed` | FastEmbed (default) |
| `openai` | OpenAI embeddings |
| `nim` | NVIDIA NIM embeddings |

### Embeddings Configuration

```yaml
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct

  - type: embeddings
    engine: FastEmbed
    model: all-MiniLM-L6-v2
```

---

## NVIDIA NIM Configuration

The NeMo Guardrails Library provides seamless integration with NVIDIA NIM microservices:

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

### Available Task Types

| Task Type | Description |
|-----------|-------------|
| `main` | Primary application LLM |
| `embeddings` | Embedding generation |
| `self_check_input` | Input validation checks |
| `self_check_output` | Output validation checks |
| `generate_user_intent` | Canonical user intent generation |
| `generate_next_steps` | Next step prediction |
| `generate_bot_message` | Bot response generation |
| `fact_checking` | Fact verification |

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

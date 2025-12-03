# Model Configuration

This section describes how to configure LLM models and embedding models in the `config.yml` file.

## The `models` Key

The `models` key defines the LLM providers and models used by the NeMo Guardrails toolkit.

### Main LLM Model

Configure the primary LLM model using the `main` type:

```yaml
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct
```

| Attribute | Description |
|-----------|-------------|
| `type` | Set to `main` to indicate the application LLM |
| `engine` | The LLM provider (for example, `openai`, `nim`, `anthropic`) |
| `model` | The model name (for example, `gpt-3.5-turbo-instruct`, `meta/llama-3.1-8b-instruct`) |
| `parameters` | Optional parameters to pass to the LangChain class |

### Supported Engines

| Engine | Description |
|--------|-------------|
| `openai` | OpenAI models (GPT-3.5, GPT-4, GPT-4o) |
| `nim` | NVIDIA NIM microservices (local or hosted) |
| `nvidia_ai_endpoints` | Alias for `nim` engine |
| `anthropic` | Anthropic Claude models |
| `azure` | Azure OpenAI models |
| `cohere` | Cohere models |
| `huggingface_endpoint` | HuggingFace Inference Endpoints |
| `huggingface_hub` | HuggingFace Hub models |
| `self_hosted` | Self-hosted models |

### Model Parameters

Pass additional parameters to the underlying LangChain class:

```yaml
models:
  - type: main
    engine: openai
    model: gpt-4
    parameters:
      temperature: 0.7
      max_tokens: 1000
```

## NVIDIA NIM Configuration

Configure NVIDIA NIM microservices for optimized inference:

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama-3.1-8b-instruct
```

For locally-deployed NIMs, specify the base URL:

```yaml
models:
  - type: main
    engine: nim
    model: meta/llama-3.1-8b-instruct
    parameters:
      base_url: http://localhost:8000/v1
```

## Embeddings Model

Configure the embedding model for knowledge base retrieval and similarity search:

```yaml
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct

  - type: embeddings
    engine: FastEmbed
    model: all-MiniLM-L6-v2
```

### Supported Embedding Providers

| Provider | Engine Name | Default Model |
|----------|-------------|---------------|
| FastEmbed (default) | `FastEmbed` | `all-MiniLM-L6-v2` |
| OpenAI | `openai` | `text-embedding-ada-002` |
| NVIDIA NIM | `nim` | Various |

### OpenAI Embeddings Example

```yaml
models:
  - type: embeddings
    engine: openai
    model: text-embedding-ada-002
```

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
| `self_check_input` | Input validation checks |
| `self_check_output` | Output validation checks |
| `generate_user_intent` | Canonical user intent generation |
| `generate_next_steps` | Next step prediction |
| `generate_bot_message` | Bot response generation |
| `fact_checking` | Fact verification |
| `embeddings` | Embedding generation |

## Example Configuration

Complete model configuration example:

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

## Related Topics

- [LLM Configuration](../../user-guides/configuration-guide/llm-configuration) - Detailed LLM provider options
- [LLM Support](../../user-guides/llm-support) - Supported models and evaluation results

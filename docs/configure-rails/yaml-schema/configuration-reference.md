---
title:
  page: "Complete Configuration Reference for config.yml"
  nav: "Configuration Reference"
description: "Complete reference for all config.yml options including models, rails, prompts, and advanced settings."
topics: ["Configuration", "Reference"]
tags: ["config.yml", "Models", "Rails", "YAML", "Reference"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  Source of truth: nemoguardrails/rails/llm/config.py
-->

# Complete Configuration Reference

This reference documents all configuration options for `config.yml`, derived from the authoritative Pydantic schema in [`nemoguardrails/rails/llm/config.py`](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/nemoguardrails/rails/llm/config.py).

## Configuration Structure

```yaml
models:           # LLM and embedding model configurations
  - type: main
    engine: openai
    model: gpt-4

rails:            # Guardrail configurations
  input:
    flows: []
  output:
    flows: []
  config: {}

prompts:          # Task-specific prompts
  - task: self_check_input
    content: "..."

instructions:     # System instructions
  - type: general
    content: "..."
```

---

## Models Configuration

The `models` key defines LLM providers and models used by NeMo Guardrails.

### Model Schema

```yaml
models:
  - type: main                    # Required: Model type
    engine: openai                # Required: LLM provider
    model: gpt-4                  # Required: Model name
    mode: chat                    # Optional: "chat" or "text" (default: "chat")
    api_key_env_var: OPENAI_KEY   # Optional: Environment variable for API key
    parameters:                   # Optional: Provider-specific parameters
      temperature: 0.7
      max_tokens: 1000
    cache:                        # Optional: Caching configuration
      enabled: false
      maxsize: 50000
```

### Model Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `type` | string | ✓ | Model type: `main`, `embeddings`, or task-specific |
| `engine` | string | ✓ | LLM provider (see [Engines](#engines)) |
| `model` | string | ✓ | Model name (can also be in `parameters.model_name`) |
| `mode` | string | | Completion mode: `chat` or `text` (default: `chat`) |
| `api_key_env_var` | string | | Environment variable containing API key |
| `parameters` | object | | Provider-specific parameters passed to LangChain |
| `cache` | object | | Cache configuration for this model |

### Model Types

| Type | Description |
|------|-------------|
| `main` | Primary application LLM |
| `embeddings` | Embedding generation model |
| `self_check_input` | Input validation model |
| `self_check_output` | Output validation model |
| `content_safety` | Content safety model |
| `topic_control` | Topic control model |
| `generate_user_intent` | Canonical user intent generation |
| `generate_next_steps` | Next step prediction |
| `generate_bot_message` | Bot response generation |
| `fact_checking` | Fact verification model |
| `llama_guard` | LlamaGuard content moderation |

### Engines

#### Core Engines

| Engine | Description |
|--------|-------------|
| `openai` | OpenAI models |
| `nim` | NVIDIA NIM microservices |
| `nvidia_ai_endpoints` | Alias for `nim` |
| `azure` | Azure OpenAI models |
| `anthropic` | Anthropic Claude models |
| `cohere` | Cohere models |
| `vertexai` | Google Vertex AI |

#### Self-Hosted Engines

| Engine | Description |
|--------|-------------|
| `huggingface_hub` | HuggingFace Hub models |
| `huggingface_endpoint` | HuggingFace Inference Endpoints |
| `vllm_openai` | vLLM with OpenAI-compatible API |
| `trt_llm` | TensorRT-LLM |
| `self_hosted` | Generic self-hosted models |

#### Embedding Engines

| Engine | Description |
|--------|-------------|
| `FastEmbed` | FastEmbed (default) |
| `openai` | OpenAI embeddings |
| `nim` | NVIDIA NIM embeddings |

### Model Cache Configuration

```yaml
models:
  - type: content_safety
    engine: nim
    model: nvidia/llama-3.1-nemotron-safety-guard-8b-v3
    cache:
      enabled: true
      maxsize: 50000
      stats:
        enabled: false
        log_interval: null
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable caching for this model |
| `maxsize` | integer | `50000` | Maximum cache entries |
| `stats.enabled` | boolean | `false` | Enable cache statistics tracking |
| `stats.log_interval` | float | `null` | Seconds between stats logging |

---

## Rails Configuration

The `rails` key configures guardrails that control LLM behavior.

### Rails Schema

```yaml
rails:
  input:
    parallel: false
    flows:
      - self check input
      - check jailbreak

  output:
    parallel: false
    flows:
      - self check output
    streaming:
      enabled: false
      chunk_size: 200
      context_size: 50
      stream_first: true

  retrieval:
    flows:
      - check retrieval sensitive data

  dialog:
    single_call:
      enabled: false
      fallback_to_multiple_calls: true
    user_messages:
      embeddings_only: false

  actions:
    instant_actions: []

  tool_output:
    flows: []
    parallel: false

  tool_input:
    flows: []
    parallel: false

  config:
    # Rail-specific configurations
```

### Input Rails

Process user messages before they reach the LLM.

```yaml
rails:
  input:
    parallel: false      # Execute flows in parallel
    flows:
      - self check input
      - check jailbreak
      - mask sensitive data on input
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `parallel` | boolean | `false` | Execute input rails in parallel |
| `flows` | list | `[]` | Names of flows that implement input rails |

#### Built-in Input Flows

| Flow | Description |
|------|-------------|
| `self check input` | LLM-based policy compliance check |
| `check jailbreak` | Jailbreak detection heuristics |
| `jailbreak detection model` | NIM-based jailbreak detection |
| `mask sensitive data on input` | Mask PII in user input |
| `detect sensitive data on input` | Detect and block PII |
| `llama guard check input` | LlamaGuard content moderation |
| `content safety check input` | NVIDIA content safety model |
| `topic safety check input` | Topic control model |

### Output Rails

Process LLM responses before returning to users.

```yaml
rails:
  output:
    parallel: false
    flows:
      - self check output
      - self check facts
    streaming:
      enabled: false
      chunk_size: 200
      context_size: 50
      stream_first: true
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `parallel` | boolean | `false` | Execute output rails in parallel |
| `flows` | list | `[]` | Names of flows that implement output rails |
| `streaming` | object | | Streaming output configuration |

#### Output Streaming Configuration

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable streaming mode |
| `chunk_size` | integer | `200` | Tokens per processing chunk |
| `context_size` | integer | `50` | Tokens carried from previous chunk |
| `stream_first` | boolean | `true` | Stream before applying output rails |

#### Built-in Output Flows

| Flow | Description |
|------|-------------|
| `self check output` | LLM-based policy compliance check |
| `self check facts` | Fact verification |
| `self check hallucination` | Hallucination detection |
| `mask sensitive data on output` | Mask PII in output |
| `llama guard check output` | LlamaGuard content moderation |
| `content safety check output` | NVIDIA content safety model |

### Retrieval Rails

Process chunks retrieved from knowledge base.

```yaml
rails:
  retrieval:
    flows:
      - check retrieval sensitive data
```

### Dialog Rails

Control conversation flow after user intent is determined.

```yaml
rails:
  dialog:
    single_call:
      enabled: false
      fallback_to_multiple_calls: true
    user_messages:
      embeddings_only: false
      embeddings_only_similarity_threshold: null
      embeddings_only_fallback_intent: null
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `single_call.enabled` | boolean | `false` | Use single LLM call for intent + response |
| `single_call.fallback_to_multiple_calls` | boolean | `true` | Fall back if single call fails |
| `user_messages.embeddings_only` | boolean | `false` | Use only embeddings for intent matching |

### Action Rails

Control custom action and tool invocations.

```yaml
rails:
  actions:
    instant_actions:
      - action_name_1
      - action_name_2
```

### Tool Rails

Control tool input/output processing.

```yaml
rails:
  tool_output:
    flows:
      - validate tool parameters
    parallel: false

  tool_input:
    flows:
      - filter tool results
    parallel: false
```

---

## Rails Config Section

The `rails.config` section contains configuration for specific built-in rails.

### Jailbreak Detection

```yaml
rails:
  config:
    jailbreak_detection:
      # Heuristics-based detection
      server_endpoint: null
      length_per_perplexity_threshold: 89.79
      prefix_suffix_perplexity_threshold: 1845.65

      # NIM-based detection
      nim_base_url: "http://localhost:8000/v1/"
      nim_server_endpoint: "classify"
      api_key_env_var: "JAILBREAK_KEY"
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `server_endpoint` | string | `null` | Heuristics model endpoint |
| `length_per_perplexity_threshold` | float | `89.79` | Length/perplexity threshold |
| `prefix_suffix_perplexity_threshold` | float | `1845.65` | Prefix/suffix perplexity threshold |
| `nim_base_url` | string | `null` | NIM base URL (e.g., `http://localhost:8000/v1`) |
| `nim_server_endpoint` | string | `"classify"` | NIM endpoint path |
| `api_key_env_var` | string | `null` | Environment variable for API key |
| `api_key` | string | `null` | API key (not recommended) |

### Sensitive Data Detection (Presidio)

```yaml
rails:
  config:
    sensitive_data_detection:
      recognizers: []
      input:
        entities:
          - PERSON
          - EMAIL_ADDRESS
          - PHONE_NUMBER
          - CREDIT_CARD
        mask_token: "*"
        score_threshold: 0.2
      output:
        entities:
          - PERSON
          - EMAIL_ADDRESS
      retrieval:
        entities: []
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `recognizers` | list | `[]` | Custom Presidio recognizers |
| `input/output/retrieval.entities` | list | `[]` | Entity types to detect |
| `input/output/retrieval.mask_token` | string | `"*"` | Token for masking |
| `input/output/retrieval.score_threshold` | float | `0.2` | Detection confidence threshold |

### Injection Detection

```yaml
rails:
  config:
    injection_detection:
      injections:
        - sqli
        - template
        - code
        - xss
      action: reject    # "reject" or "omit"
      yara_path: ""
      yara_rules: {}
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `injections` | list | `[]` | Injection types: `sqli`, `template`, `code`, `xss` |
| `action` | string | `"reject"` | Action: `reject` or `omit` |
| `yara_path` | string | `""` | Custom YARA rules path |
| `yara_rules` | object | `{}` | Inline YARA rules |

### Fact Checking

```yaml
rails:
  config:
    fact_checking:
      parameters:
        endpoint: "http://localhost:5000"
      fallback_to_self_check: false
```

### Content Safety

```yaml
rails:
  config:
    content_safety:
      multilingual:
        enabled: false
        refusal_messages:
          en: "Sorry, I cannot help with that."
          es: "Lo siento, no puedo ayudar con eso."
```

### Third-Party Integrations

#### AutoAlign

```yaml
rails:
  config:
    autoalign:
      parameters: {}
      input:
        guardrails_config: {}
      output:
        guardrails_config: {}
```

#### Patronus

```yaml
rails:
  config:
    patronus:
      input:
        evaluate_config:
          success_strategy: all_pass  # or any_pass
          params: {}
      output:
        evaluate_config:
          success_strategy: all_pass
          params: {}
```

#### Clavata

```yaml
rails:
  config:
    clavata:
      server_endpoint: "https://gateway.app.clavata.ai:8443"
      policies: {}
      label_match_logic: ANY  # or ALL
      input:
        policy: "policy_alias"
        labels: []
      output:
        policy: "policy_alias"
        labels: []
```

#### Pangea AI Guard

```yaml
rails:
  config:
    pangea:
      input:
        recipe: "recipe_key"
      output:
        recipe: "recipe_key"
```

#### Trend Micro

```yaml
rails:
  config:
    trend_micro:
      v1_url: "https://api.xdr.trendmicro.com/beta/aiSecurity/guard"
      api_key_env_var: "TREND_MICRO_API_KEY"
```

#### Cisco AI Defense

```yaml
rails:
  config:
    ai_defense:
      timeout: 30.0
      fail_open: false
```

#### Private AI

```yaml
rails:
  config:
    private_ai_detection:
      server_endpoint: "http://localhost:8080/process/text"
      input:
        entities: []
      output:
        entities: []
      retrieval:
        entities: []
```

#### Fiddler Guardrails

```yaml
rails:
  config:
    fiddler:
      fiddler_endpoint: "http://localhost:8080/process/text"
      safety_threshold: 0.1
      faithfulness_threshold: 0.05
```

#### Guardrails AI

```yaml
rails:
  config:
    guardrails_ai:
      input:
        validators:
          - name: toxic_language
            parameters:
              threshold: 0.5
            metadata: {}
      output:
        validators:
          - name: pii
            parameters: {}
```

---

## Prompts Configuration

Define prompts for LLM tasks.

```yaml
prompts:
  - task: self_check_input
    content: |
      Your task is to check if the user input is safe.
      User input: {{ user_input }}
      Answer [Yes/No]:
    output_parser: null
    max_length: 16000
    max_tokens: null
    mode: standard
    stop: null
    models: null    # Restrict to specific engines/models
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | string | ✓ | Task identifier |
| `content` | string | | Prompt template (mutually exclusive with `messages`) |
| `messages` | list | | Chat messages (mutually exclusive with `content`) |
| `output_parser` | string | `null` | Output parser name |
| `max_length` | integer | `16000` | Maximum prompt length (characters) |
| `max_tokens` | integer | `null` | Maximum response tokens |
| `mode` | string | `"standard"` | Prompting mode |
| `stop` | list | `null` | Stop tokens |
| `models` | list | `null` | Restrict to engines/models (e.g., `["openai", "nim/llama-3.1"]`) |

---

## Other Configuration Options

### Instructions

```yaml
instructions:
  - type: general
    content: |
      You are a helpful assistant.
```

### Sample Conversation

```yaml
sample_conversation: |
  user: Hello
  assistant: Hi! How can I help you?
```

### Knowledge Base

```yaml
knowledge_base:
  folder: kb
  embedding_search_provider:
    name: default
    parameters: {}
    cache:
      enabled: false
```

### Core Settings

```yaml
core:
  embedding_search_provider:
    name: default
    parameters: {}
```

### Tracing

```yaml
tracing:
  enabled: false
  adapters:
    - name: FileSystem
  span_format: opentelemetry
  enable_content_capture: false
```

### Streaming

```yaml
streaming:
  enabled: false
  stream_on_start: false
  stream_on_end: true
  first_chunk_suffix: ""
  last_chunk_suffix: ""
```

### Import Paths

```yaml
import_paths:
  - path/to/shared/config
```

---

## Complete Example

```yaml
models:
  # Main application LLM
  - type: main
    engine: nim
    model: meta/llama-3.1-70b-instruct
    parameters:
      temperature: 0.7

  # Content safety model
  - type: content_safety
    engine: nim
    parameters:
      base_url: "http://localhost:8000/v1"
      model_name: "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"

  # Embeddings
  - type: embeddings
    engine: FastEmbed
    model: all-MiniLM-L6-v2

rails:
  input:
    flows:
      - content safety check input $model=content_safety

  output:
    flows:
      - content safety check output $model=content_safety

  config:
    jailbreak_detection:
      nim_base_url: "http://localhost:8001/v1/"

prompts:
  - task: content_safety_check_input $model=content_safety
    content: |
      Check if this content is safe: {{ user_input }}
    output_parser: nemoguard_parse_prompt_safety
    max_tokens: 50

instructions:
  - type: general
    content: |
      You are a helpful, harmless, and honest assistant.

streaming:
  enabled: true
```

---

## Related Topics

- [Model Configuration](model-configuration.md)
- [Guardrails Configuration](guardrails-configuration/index.md)
- [Built-in Guardrails](guardrails-configuration/built-in-guardrails.md)
- [Prompt Configuration](prompt-configuration.md)
- [Streaming Configuration](streaming/index.md)

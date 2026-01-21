---
title:
  page: "Configuration YAML Schema Reference"
  nav: "YAML Schema Reference"
description: "Reference for all config.yml options including models, rails, prompts, and advanced settings."
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

# Configuration YAML Schema Reference

This reference documents all configuration options for `config.yml`, derived from the authoritative Pydantic schema in [`nemoguardrails/rails/llm/config.py`](https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/nemoguardrails/rails/llm/config.py).

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
| `models.type` | string | ✓ | Model type: `main`, `embeddings`, or task-specific |
| `models.engine` | string | ✓ | LLM provider (see [Engines](#engines)) |
| `models.model` | string | ✓ | Model name (can also be in `parameters.model_name`) |
| `models.mode` | string | | Completion mode: `chat` or `text` (default: `chat`) |
| `models.api_key_env_var` | string | | Environment variable containing API key |
| `models.parameters` | object | | Provider-specific parameters passed to LangChain |
| `models.cache` | object | | Cache configuration for this model |

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
| `models.cache.enabled` | boolean | `false` | Enable caching for this model |
| `models.cache.maxsize` | integer | `50000` | Maximum cache entries |
| `models.cache.stats.enabled` | boolean | `false` | Enable cache statistics tracking |
| `models.cache.stats.log_interval` | float | `null` | Seconds between stats logging |

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

### Rail Types

The following table summarizes the available rail types and their trigger points.

| Rail Type | Trigger Point | Purpose |
|----------|---------------|---------|
| **Input rails** | When user input is received | Validate, filter, or modify user input |
| **Retrieval rails** | After RAG retrieval completes | Process retrieved chunks |
| **Dialog rails** | After canonical form is computed | Control conversation flow |
| **Execution rails** | Before/after action execution | Control tool and action calls |
| **Output rails** | When LLM generates output | Validate, filter, or modify bot responses |

The following diagram shows the guardrails process described in the table above in detail.

```{image} ../_static/images/programmable_guardrails_flow.png
:alt: "Diagram showing the programmable guardrails flow"
:width: 800px
:align: center
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
| `rails.input.parallel` | boolean | `false` | Execute input rails in parallel |
| `rails.input.flows` | list | `[]` | Names of flows that implement input rails |

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
| `rails.output.parallel` | boolean | `false` | Execute output rails in parallel |
| `rails.output.flows` | list | `[]` | Names of flows that implement output rails |
| `rails.output.streaming` | object | | Streaming output configuration |

#### Output Streaming Configuration

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `rails.output.streaming.enabled` | boolean | `false` | Enable streaming mode |
| `rails.output.streaming.chunk_size` | integer | `200` | Tokens per processing chunk |
| `rails.output.streaming.context_size` | integer | `50` | Tokens carried from previous chunk |
| `rails.output.streaming.stream_first` | boolean | `true` | Stream before applying output rails |

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
| `rails.dialog.single_call.enabled` | boolean | `false` | Use single LLM call for intent + response |
| `rails.dialog.single_call.fallback_to_multiple_calls` | boolean | `true` | Fall back if single call fails |
| `rails.dialog.user_messages.embeddings_only` | boolean | `false` | Use only embeddings for intent matching |

### Execution Rails

Control tool and action invocations.

#### Action Rails

Control custom action and tool invocations.

```yaml
rails:
  actions:
    instant_actions:
      - action_name_1
      - action_name_2
```

#### Tool Rails

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

### Rails Config Section

The `rails.config` section contains configuration for specific built-in rails.

#### Jailbreak Detection

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
| `rails.config.jailbreak_detection.server_endpoint` | string | `null` | Heuristics model endpoint |
| `rails.config.jailbreak_detection.length_per_perplexity_threshold` | float | `89.79` | Length/perplexity threshold |
| `rails.config.jailbreak_detection.prefix_suffix_perplexity_threshold` | float | `1845.65` | Prefix/suffix perplexity threshold |
| `rails.config.jailbreak_detection.nim_base_url` | string | `null` | NIM base URL (e.g., `http://localhost:8000/v1`) |
| `rails.config.jailbreak_detection.nim_server_endpoint` | string | `"classify"` | NIM endpoint path |
| `rails.config.jailbreak_detection.api_key_env_var` | string | `null` | Environment variable for API key |
| `rails.config.jailbreak_detection.api_key` | string | `null` | API key (not recommended) |

#### Sensitive Data Detection (Presidio)

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
| `rails.config.sensitive_data_detection.recognizers` | list | `[]` | Custom Presidio recognizers |
| `rails.config.sensitive_data_detection.input/output/retrieval.entities` | list | `[]` | Entity types to detect |
| `rails.config.sensitive_data_detection.input/output/retrieval.mask_token` | string | `"*"` | Token for masking |
| `rails.config.sensitive_data_detection.input/output/retrieval.score_threshold` | float | `0.2` | Detection confidence threshold |

#### Injection Detection

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
| `rails.config.injection_detection.injections` | list | `[]` | Injection types: `sqli`, `template`, `code`, `xss` |
| `rails.config.injection_detection.action` | string | `"reject"` | Action: `reject` or `omit` |
| `rails.config.injection_detection.yara_path` | string | `""` | Custom YARA rules path |
| `rails.config.injection_detection.yara_rules` | object | `{}` | Inline YARA rules |

#### Fact Checking

```yaml
rails:
  config:
    fact_checking:
      parameters:
        endpoint: "http://localhost:5000"
      fallback_to_self_check: false
```

#### Content Safety

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

#### Third-Party Integrations

##### AutoAlign

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

For more information, refer to [AutoAlign Integration](../user-guides/community/auto-align.md).

##### Patronus

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

For more information, refer to [Patronus Evaluate API Integration](../user-guides/community/patronus-evaluate-api.md).

##### Clavata

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

For more information, refer to [Clavata Integration](../user-guides/community/clavata.md).

##### Pangea AI Guard

```yaml
rails:
  config:
    pangea:
      input:
        recipe: "recipe_key"
      output:
        recipe: "recipe_key"
```

For more information, refer to [Pangea AI Guard Integration](../user-guides/community/pangea.md).

##### Trend Micro

```yaml
rails:
  config:
    trend_micro:
      v1_url: "https://api.xdr.trendmicro.com/beta/aiSecurity/guard"
      api_key_env_var: "TREND_MICRO_API_KEY"
```

For more information, refer to [Trend Micro Integration](../user-guides/community/trend-micro.md).

##### Cisco AI Defense

```yaml
rails:
  config:
    ai_defense:
      timeout: 30.0
      fail_open: false
```

For more information, refer to [Cisco AI Defense Integration](../user-guides/community/ai-defense.md).

##### Private AI

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

For more information, refer to [Private AI Integration](../user-guides/community/privateai.md).

##### Fiddler Guardrails

```yaml
rails:
  config:
    fiddler:
      fiddler_endpoint: "http://localhost:8080/process/text"
      safety_threshold: 0.1
      faithfulness_threshold: 0.05
```

For more information, refer to [Fiddler Guardrails Integration](../user-guides/community/fiddler.md).

##### Guardrails AI

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

For more information, refer to [Guardrails AI Integration](../user-guides/community/guardrails-ai.md).

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
| `prompts.task` | string | ✓ | Task identifier |
| `prompts.content` | string | | Prompt template (mutually exclusive with `messages`) |
| `prompts.messages` | list | | Chat messages (mutually exclusive with `content`) |
| `prompts.output_parser` | string | `null` | Output parser name |
| `prompts.max_length` | integer | `16000` | Maximum prompt length (characters) |
| `prompts.max_tokens` | integer | `null` | Maximum response tokens |
| `prompts.mode` | string | `"standard"` | Prompting mode |
| `prompts.stop` | list | `null` | Stop tokens |
| `prompts.models` | list | `null` | Restrict to engines/models (e.g., `["openai", "nim/llama-3.1"]`) |

### Available Tasks

The following table lists all available tasks you can specify to `prompts.task`.

| Task | Description |
|------|-------------|
| `self_check_input` | Check if user input complies with policy |
| `self_check_output` | Check if bot output complies with policy |
| `self_check_facts` | Verify factual accuracy of responses |
| `self_check_hallucination` | Detect hallucinations in responses |
| `generate_user_intent` | Generate canonical user intent |
| `generate_next_steps` | Determine next conversation step |
| `generate_bot_message` | Generate bot response |
| `general` | General response generation (no dialog rails) |

### Available Prompt Message Types

The following table lists all available message types you can specify to `prompts.messages.type`.

| Type | Description |
|------|-------------|
| `system` | System-level instructions |
| `user` | User message content |
| `assistant` | Assistant/bot message content |
| `bot` | Alias for `assistant` |

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

The following YAML example demonstrates a complete `config.yml` file that wires together a main language model, a dedicated content safety model, and an embeddings model. It configures rails for input and output content safety checks, points to a local NIM service for jailbreak detection, defines a content safety prompt, provides general instructions for the assistant, and enables response streaming from both the main and content safety models.

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
    streaming:
      enabled: true

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

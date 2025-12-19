---
title: Guardrails Configuration
description: Configure input, output, dialog, retrieval, and execution rails in config.yml to control LLM behavior.
---

# Guardrails Configuration

This section describes how to configure guardrails in the `config.yml` file to control LLM behavior.

## The `rails` Key

The `rails` key defines which guardrails are active and their configuration options.
Rails are organized into five categories based on when they trigger during the guardrails process.

## Rail Categories

The following table summarizes the different rail categories and their trigger points.

| Category | Trigger Point | Purpose |
|----------|---------------|---------|
| **Input rails** | When user input is received | Validate, filter, or modify user input |
| **Retrieval rails** | After RAG retrieval completes | Process retrieved chunks |
| **Dialog rails** | After canonical form is computed | Control conversation flow |
| **Execution rails** | Before/after action execution | Control tool and action calls |
| **Output rails** | When LLM generates output | Validate, filter, or modify bot responses |

The following diagram shows the guardrails process described in the table above in detail.

```{image} ../../../_static/images/programmable_guardrails_flow.png
:alt: "Diagram showing the programmable guardrails flow"
:width: 800px
:align: center
```

## Basic Configuration

```yaml
rails:
  input:
    flows:
      - self check input
      - check jailbreak
      - mask sensitive data on input

  output:
    flows:
      - self check output
      - self check facts
      - check output sensitive data

  retrieval:
    flows:
      - check retrieval sensitive data
```

## Input Rails

Input rails process user messages before they reach the LLM:

```yaml
rails:
  input:
    flows:
      - self check input           # LLM-based input validation
      - check jailbreak            # Jailbreak detection
      - mask sensitive data on input  # PII masking
```

### Available Flows for Input Rails

| Flow | Description |
|------|-------------|
| `self check input` | LLM-based policy compliance check |
| `check jailbreak` | Detect jailbreak attempts |
| `mask sensitive data on input` | Mask PII in user input |
| `detect sensitive data on input` | Detect and block PII |
| `llama guard check input` | LlamaGuard content moderation |
| `content safety check input` | NVIDIA content safety model |

## Retrieval Rails

Retrieval rails process chunks retrieved from the knowledge base:

```yaml
rails:
  retrieval:
    flows:
      - check retrieval sensitive data
```

## Dialog Rails

Dialog rails control conversation flow after user intent is determined:

```yaml
rails:
  dialog:
    single_call:
      enabled: false
      fallback_to_multiple_calls: true

    user_messages:
      embeddings_only: false
```

### Dialog Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `single_call.enabled` | Use single LLM call for intent, next step, and message | `false` |
| `single_call.fallback_to_multiple_calls` | Fall back to multiple calls if single call fails | `true` |
| `user_messages.embeddings_only` | Use only embeddings for user intent matching | `false` |

## Execution Rails

Execution rails control custom action and tool invocations:

```yaml
rails:
  execution:
    flows:
      - check tool input
      - check tool output
```

## Output Rails

Output rails process LLM responses before returning to users:

```yaml
rails:
  output:
    flows:
      - self check output          # LLM-based output validation
      - self check facts           # Fact verification
      - self check hallucination   # Hallucination detection
      - mask sensitive data on output  # PII masking
```

### Available Flows for Output Rails

| Flow | Description |
|------|-------------|
| `self check output` | LLM-based policy compliance check |
| `self check facts` | Verify factual accuracy |
| `self check hallucination` | Detect hallucinations |
| `mask sensitive data on output` | Mask PII in output |
| `llama guard check output` | LlamaGuard content moderation |
| `content safety check output` | NVIDIA content safety model |

## Rail-Specific Configuration

Configure options for specific rails using the `config` key:

```yaml
rails:
  config:
    # Sensitive data detection settings
    sensitive_data_detection:
      input:
        entities:
          - PERSON
          - EMAIL_ADDRESS
          - PHONE_NUMBER
      output:
        entities:
          - PERSON
          - EMAIL_ADDRESS

    # Jailbreak detection settings
    jailbreak_detection:
      length_per_perplexity_threshold: 89.79
      prefix_suffix_perplexity_threshold: 1845.65

    # Fact-checking settings
    fact_checking:
      parameters:
        endpoint: "http://localhost:5000"
```

## Example Configuration

Complete guardrails configuration example:

```yaml
rails:
  # Input validation
  input:
    flows:
      - self check input
      - check jailbreak
      - mask sensitive data on input

  # Output validation
  output:
    flows:
      - self check output
      - self check facts

  # Retrieval processing
  retrieval:
    flows:
      - check retrieval sensitive data

  # Dialog behavior
  dialog:
    single_call:
      enabled: false

  # Rail-specific settings
  config:
    sensitive_data_detection:
      input:
        entities:
          - PERSON
          - EMAIL_ADDRESS
          - CREDIT_CARD
      output:
        entities:
          - PERSON
          - EMAIL_ADDRESS
```

## Related Topics

- [Built-in Guardrails](built-in-guardrails.md) - Complete list of built-in rails
- [Parallel Rails](parallel-rails.md) - How to invoke rails in parallel

```{toctree}
:hidden:
:maxdepth: 2

built-in-guardrails
Rails in Parallel <parallel-rails.md>
```

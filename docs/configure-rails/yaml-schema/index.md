---
title: Core Configuration
description: Complete reference for config.yml structure including models, guardrails, prompts, and tracing settings.
---

# Core Configuration

This section describes the `config.yml` file schema used to configure the NeMo Guardrails toolkit.
The `config.yml` file is the primary configuration file for defining LLM models, guardrails behavior, prompts, knowledge base settings, and tracing options.

## Overview

The following is a complete schema for a `config.yml` file:

```yaml
# LLM model configuration
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct

# Instructions for the LLM (similar to system prompts)
instructions:
  - type: general
    content: |
      You are a helpful AI assistant.

# Guardrails configuration
rails:
  input:
    flows:
      - self check input
  output:
    flows:
      - self check output

# Prompt customization
prompts:
  - task: self_check_input
    content: |
      Your task is to check if the user message complies with policy.

# Knowledge base settings
knowledge_base:
  embedding_search_provider:
    name: default

# Tracing and monitoring
tracing:
  enabled: true
  adapters:
    - name: FileSystem
      filepath: "./logs/traces.jsonl"
```

## Configuration Sections

The following sections provide detailed documentation for each configuration area:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Model Configuration
:link: model-configuration
:link-type: doc

Configure LLM providers, embedding models, and task-specific models in the config.yml file.
:::

:::{grid-item-card} Guardrails Configuration
:link: guardrails-configuration/index
:link-type: doc

Configure input, output, dialog, retrieval, and execution rails in config.yml to control LLM behavior.
:::

:::{grid-item-card} Prompt Configuration
:link: prompt-configuration
:link-type: doc

Customize prompts for LLM tasks including self-check input/output, fact checking, and intent generation.
:::

:::{grid-item-card} Tracing Configuration
:link: tracing-configuration
:link-type: doc

Configure tracing adapters (FileSystem, OpenTelemetry) to monitor and debug guardrails interactions.
:::

:::{grid-item-card} Streaming Configuration
:link: streaming/index
:link-type: doc

Configure streaming for LLM token generation and output rail processing in config.yml.
:::

::::

## File Organization

Configuration files are typically organized in a `config` folder:

```text
.
├── config
│   ├── config.yml        # Main configuration file
│   ├── prompts.yml       # Custom prompts (optional)
│   ├── rails/            # Colang flow definitions
│   │   ├── input.co
│   │   ├── output.co
│   │   └── ...
│   ├── kb/               # Knowledge base documents
│   │   ├── doc1.md
│   │   └── ...
│   ├── actions.py        # Custom actions (optional)
│   └── config.py         # Custom initialization (optional)
```

For detailed information about each configuration section, refer to the individual pages linked above.

```{toctree}
:hidden:
:maxdepth: 2

model-configuration
guardrails-configuration/index
prompt-configuration
tracing-configuration
streaming/index
```

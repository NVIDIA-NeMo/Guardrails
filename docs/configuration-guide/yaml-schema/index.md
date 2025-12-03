# Configuration YAML Schema

This section describes the `config.yml` file schema used to configure the NeMo Guardrails toolkit.
The `config.yml` file is the primary configuration file for defining LLM models, guardrails behavior, prompts, knowledge base settings, and tracing options.

## Overview

A typical `config.yml` file contains the following top-level keys:

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
:link: yaml-schema/model-configuration
:link-type: doc

Configure LLM providers, models, embeddings, and model-specific parameters.
:::

:::{grid-item-card} Guardrails Configuration
:link: yaml-schema/guardrails-configuration
:link-type: doc

Set up input, output, dialog, retrieval, and execution rails to control LLM behavior.
:::

:::{grid-item-card} Prompt Configuration
:link: yaml-schema/prompt-configuration
:link-type: doc

Customize prompts for various LLM tasks including self-check, intent generation, and more.
:::

:::{grid-item-card} Knowledge Base Configuration
:link: yaml-schema/knowledge-base-configuration
:link-type: doc

Configure document retrieval and RAG (Retrieval-Augmented Generation) settings.
:::

:::{grid-item-card} Tracing Configuration
:link: yaml-schema/tracing-configuration
:link-type: doc

Enable monitoring, logging, and observability for guardrails interactions.
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
guardrails-configuration
prompt-configuration
knowledge-base-configuration
tracing-configuration
```

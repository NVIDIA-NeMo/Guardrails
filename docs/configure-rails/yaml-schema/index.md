---
title:
  page: "Core Configurations for Guardrails"
  nav: "Core Configuration"
description: "Define models, guardrails, prompts, and tracing settings in the config.yml file."
topics: ["Configuration", "AI Safety"]
tags: ["YAML", "Configuration", "Models", "Prompts", "Tracing"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

# Core Configurations for Guardrails

This section describes the `config.yml` file schema used to configure the NeMo Guardrails library.
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

## Quick Reference

For a complete, consolidated reference of all configuration options, see the [Configuration Reference](configuration-reference.md).

## Configuration Sections

The following guides provide detailed documentation for each configuration section of the overall `config.yml` file:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Configuration Reference
:link: configuration-reference
:link-type: doc

Complete reference for all config.yml options including models, rails, prompts, and advanced settings.
:::

:::{grid-item-card} Models
:link: model-configuration
:link-type: doc

Configure LLM engines, embedding models, and task-specific models in config.yml.
:::

:::{grid-item-card} Guardrails Configuration
:link: guardrails-configuration/index
:link-type: doc

Configure input, output, dialog, retrieval, and execution rails in config.yml to control LLM behavior.
:::

:::{grid-item-card} Prompts
:link: prompt-configuration
:link-type: doc

Customize prompts for self-check, fact-checking, and intent generation tasks.
:::

:::{grid-item-card} Tracing
:link: tracing-configuration
:link-type: doc

Configure FileSystem and OpenTelemetry tracing adapters to monitor guardrails.
:::

:::{grid-item-card} Streaming
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
│   ├── rails/            # Colang flow definitions (optional)
│   │   ├── input.co
│   │   ├── output.co
│   │   └── ...
│   ├── kb/               # Knowledge base documents (optional)
│   │   ├── doc1.md
│   │   └── ...
│   ├── actions.py        # Custom actions (optional)
│   └── config.py         # Custom initialization (optional)
```

Once you have finished crafting your overall `config.yml` file, refer to the following guides for detailed information each of the optional components as needed:

- [Core Configuration](yaml-schema/index.md) - A complete guide to writing your `config.yml` file.
- [Colang Rails](colang/index.md) - `.co` flow files.
- [Custom Actions](actions/index.md) - `actions.py` for callable actions.
- [Custom Initialization](custom-initialization/index.md) - `config.py` for provider registration.
- [Knowledge Base Documents](other-configurations/knowledge-base.md) - `kb/` folder for RAG.

```{toctree}
:hidden:
:maxdepth: 2

Configuration Reference <configuration-reference>
Models <model-configuration>
Guardrails Configuration <guardrails-configuration/index>
Prompts <prompt-configuration>
Tracing <tracing-configuration>
Streaming <streaming/index>
```

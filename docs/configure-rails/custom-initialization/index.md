---
title:
  page: "Configuring Custom Initialization"
  nav: "Custom Initialization"
description: "Use config.py to register custom LLM providers, embedding providers, and shared resources at startup."
topics: ["Configuration", "Customization"]
tags: ["config.py", "Initialization", "LLM Providers", "Embedding Providers", "Python"]
content:
  type: "Guide"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

# Configuring Custom Initialization

The `config.py` file contains initialization code that runs **once at startup**, before the `LLMRails` instance is fully initialized. Use it to register custom providers and set up shared resources.

## When to Use config.py vs actions.py

| Use Case | File | Reason |
|----------|------|--------|
| Register custom LLM provider | `config.py` | Must happen before LLMRails initialization |
| Register custom embedding provider | `config.py` | Must happen before LLMRails initialization |
| Initialize database connection | `config.py` | Shared resource, initialized once |
| Validate user input | `actions.py` | Called during request processing |
| Call external API | `actions.py` | Called during request processing |
| Custom guardrail logic | `actions.py` | Called from Colang flows |

## Configuration Sections

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Init Function
:link: init-function
:link-type: doc

Define the init() function to initialize resources and register action parameters at startup.
:::

:::{grid-item-card} LLM Providers
:link: custom-llm-providers
:link-type: doc

Register custom text completion (BaseLLM) and chat models (BaseChatModel) for use with NeMo Guardrails.
:::

:::{grid-item-card} Embedding Providers
:link: custom-embedding-providers
:link-type: doc

Register custom embedding providers for vector similarity search in NeMo Guardrails.
:::

:::{grid-item-card} Custom Data
:link: custom-data
:link-type: doc

Pass and access custom data from config.yml in your initialization code and actions.
:::

::::

## Related Topics

- [Custom Actions](../actions/index.md) - Define callable actions in `actions.py`
- [Model Configuration](../yaml-schema/model-configuration.md) - Configure LLM models in `config.yml`

```{toctree}
:hidden:
:maxdepth: 2

Init Function <init-function>
LLM Providers <custom-llm-providers>
Embedding Providers <custom-embedding-providers>
Custom Data <custom-data>
```

---
title: Run Rails
description: Use the Python SDK and understand core classes like RailsConfig and LLMRails.
---

# Run Rails

This section covers how to use the NeMo Guardrails toolkit programmatically through the Python API. Learn about the core classes, generation methods, and advanced features for integrating guardrails into your applications.

## Core Classes

The NeMo Guardrails toolkit provides two core classes for running guardrails:

- **`RailsConfig`**: Loads and manages guardrails configuration from files or content.
- **`LLMRails`**: The main interface for generating responses with guardrails applied.

Upon initializing the core classes (`RailsConfig` and `LLMRails`) or starting the `nemoguardrails` CLI chat or server, the toolkit loads the configuration files you created in the previous chapter [Configure Rails](../configuration-guide/index.md).

## Quick Start

The following example shows the minimal code to load the prepared configuration files in the `config` directory and generate a response using the `LLMRails` class.

```python
from nemoguardrails import LLMRails, RailsConfig

# Load configuration from the config directory
config = RailsConfig.from_path("path/to/config")

# Create the LLMRails instance
rails = LLMRails(config)

# Generate a response
response = rails.generate(messages=[
    {"role": "user", "content": "Hello! How are you?"}
])
print(response["content"])
```

## Sections

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Core Classes
:link: core-classes
:link-type: doc

This guide covers the two fundamental classes in the NeMo Guardrails toolkit: `RailsConfig` for loading configurations and `LLMRails` for generating responses with guardrails.
:::

:::{grid-item-card} Generation Options
:link: generation-options
:link-type: doc

NeMo Guardrails exposes a set of **generation options** that give you fine-grained control over how the LLM generation is performed (for example, what rails are enabled, additional parameters that...
:::

:::{grid-item-card} Streaming
:link: streaming
:link-type: doc

If the application LLM supports streaming, you can configure NeMo Guardrails to stream tokens as well.
:::

:::{grid-item-card} Event-based API
:link: event-based-api
:link-type: doc

You can use a guardrails configuration through an event-based API using [`LLMRails.generate_events_async`](../api/nemoguardrails.rails.llm.llmrails.md#method-llmrailsgenerate_events_async) and...
:::

:::{grid-item-card} Tools Integration with NeMo Guardrails
:link: tools-integration
:link-type: doc

This guide provides comprehensive instructions for integrating and using tools within NeMo Guardrails via the LLMRails interface. It covers supported tools, configuration settings, practical...
:::

::::

## When to Use Each API

| API | Use Case |
|-----|----------|
| `generate()` / `generate_async()` | Standard chat interactions with messages |
| `stream_async()` | Real-time token streaming for responsive UIs |
| `generate_events()` / `generate_events_async()` | Low-level event control for custom integrations |

## Synchronous vs Asynchronous

The NeMo Guardrails toolkit provides both synchronous and asynchronous methods:

| Synchronous | Asynchronous | Description |
|-------------|--------------|-------------|
| `generate()` | `generate_async()` | Generate responses from messages |
| `generate_events()` | `generate_events_async()` | Generate events from event history |
| - | `stream_async()` | Stream tokens asynchronously |

```{note}
Use asynchronous methods (`generate_async`, `stream_async`) in async contexts for better performance. The synchronous `generate()` method cannot be called from within an async context.
```

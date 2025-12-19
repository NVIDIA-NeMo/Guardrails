---
title:
  page: "Run the NeMo Guardrails Library with the Python APIs"
  nav: "Run Rails"
description: "Use RailsConfig and LLMRails classes to load configurations and generate guarded responses."
topics: ["AI Safety", "LLM Guardrails"]
tags: ["Python", "SDK", "API", "Streaming", "Events"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

# Run the NeMo Guardrails Library with the Python APIs

This section covers how to use the NeMo Guardrails library programmatically through the Python API. Learn about the core classes, generation methods, and advanced features for integrating guardrails into your applications.

## Core Classes

The NeMo Guardrails library provides two core classes for running guardrails:

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

Load guardrails configurations with RailsConfig and generate responses with LLMRails.
:::

:::{grid-item-card} Generation Options
:link: generation-options
:link-type: doc

Configure generation behavior with options for logging, LLM parameters, and rail selection.
:::

:::{grid-item-card} Streaming
:link: streaming
:link-type: doc

Stream LLM responses in real-time with the stream_async method and output rails support.
:::

:::{grid-item-card} Event-Based API
:link: event-based-api
:link-type: doc

Use generate_events for low-level control over guardrails execution and event handling.
:::

::::

## When to Use Each API

| API | Use Case |
|-----|----------|
| `generate()` / `generate_async()` | Standard chat interactions with messages |
| `stream_async()` | Real-time token streaming for responsive UIs |
| `generate_events()` / `generate_events_async()` | Low-level event control for custom integrations |

## Synchronous vs Asynchronous

The NeMo Guardrails library provides both synchronous and asynchronous methods:

| Synchronous | Asynchronous | Description |
|-------------|--------------|-------------|
| `generate()` | `generate_async()` | Generate responses from messages |
| `generate_events()` | `generate_events_async()` | Generate events from event history |
| - | `stream_async()` | Stream tokens asynchronously |

```{note}
Use asynchronous methods (`generate_async`, `stream_async`) in async contexts for better performance. The synchronous `generate()` method cannot be called from within an async context.
```

---
title:
  page: "Overview of the Python API"
  nav: "Overview"
description: "RailsConfig and LLMRails core classes for generating guarded responses."
keywords: ["NeMo Guardrails", "RailsConfig", "LLMRails", "Python API", "generate", "generate_async"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "ai_platforms"]
content:
  type: concept
  difficulty: technical_intermediate
  audience: ["data_scientist", "engineer"]
---

# Overview of the NeMo Guardrails Library Python API

The NeMo Guardrails library Python API provides two core classes for running guardrails:

- **`RailsConfig`**: Loads and manages guardrails configuration from files or content.
- **`LLMRails`**: The main interface for generating responses with guardrails applied.

Upon initializing the core classes (`RailsConfig` and `LLMRails`), the library loads the configuration files you created in the previous chapter [Configure Rails](../../configure-rails/index.md).

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

## When to Use Each API

| API | Use Case |
|-----|----------|
| `generate()` / `generate_async()` | Standard chat interactions with messages |
| `stream_async()` | Real-time token streaming |
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

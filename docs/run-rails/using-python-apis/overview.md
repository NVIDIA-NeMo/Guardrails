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

Upon initializing the core classes (`RailsConfig` and `LLMRails`), the library loads the configuration files you created in the previous chapter [](../../configure-rails/index.md).

## Quick Start

The following steps show how to run a sample guardrailed chat request using the NeMo Guardrails library Python API.

### Prerequisites

Meet the following prerequisites to use the NeMo Guardrails library Python API.

1. If you haven't already, install the NeMo Guardrails library with the `nvidia` extra, following the instructions in [](../../getting-started/installation-guide.md).

1. Set up an environment variable for your NVIDIA API key.

    ```console
    export NVIDIA_API_KEY="your-nvidia-api-key"
    ```

    This is required to access NVIDIA-hosted models on [build.nvidia.com](https://build.nvidia.com). The provided example configurations ([examples/configs](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/configs)) and code examples throughout the documentation use NVIDIA-hosted models.

### Run a Sample Guardrailed Chat Request

The following example shows the minimal code to load the prepared configuration files in the `config` directory and generate a response using the `LLMRails` class.

```python
from nemoguardrails import LLMRails, RailsConfig

# Load configuration from the config directory
config = RailsConfig.from_path("examples/configs")

# Create the LLMRails instance
rails = LLMRails(config)

# Generate a response
response = rails.generate(messages=[
    {
        "role": "user",
        "content": "What is the capital of France?",
        "config_id": "content_safety"
    }
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

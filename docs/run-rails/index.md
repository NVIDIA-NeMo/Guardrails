# Run Rails

This section covers how to use the NeMo Guardrails toolkit programmatically through the Python API. Learn about the core classes, generation methods, and advanced features for integrating guardrails into your applications.

## Core Classes

The NeMo Guardrails toolkit provides two core classes for running guardrails:

- **`RailsConfig`**: Loads and manages guardrails configuration from files or content.
- **`LLMRails`**: The main interface for generating responses with guardrails applied.

## Quick Start

```python
from nemoguardrails import LLMRails, RailsConfig

# Load configuration from a directory
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

Learn about `RailsConfig` and `LLMRails`, the two fundamental classes for loading configurations and generating responses with guardrails.
:::

:::{grid-item-card} Generation Options
:link: generation-options
:link-type: doc

Fine-grained control over LLM generation, including output variables, LLM parameters, logging, and selectively disabling rails.
:::

:::{grid-item-card} Streaming
:link: streaming
:link-type: doc

Configure and use streaming responses with guardrails, including Python API, CLI, and server API usage.
:::

:::{grid-item-card} Event-based API
:link: event-based-api
:link-type: doc

Use the low-level event-based API for fine-grained control over the guardrails interaction flow.
:::

:::{grid-item-card} Tools Integration
:link: tools-integration
:link-type: doc

Integrate LangChain tools with guardrails while maintaining safety controls through input and output rails.
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

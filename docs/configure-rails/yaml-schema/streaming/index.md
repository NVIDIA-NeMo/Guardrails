---
title: Streaming Configuration
description: Configure streaming for LLM token generation and output rail processing in config.yml.
---

# Streaming Configuration

NeMo Guardrails supports two levels of streaming configuration:

1. **Global streaming** - Controls LLM token generation
2. **Output rail streaming** - Controls how output rails process streamed tokens

## Configuration Comparison

| Aspect | Global `streaming` | Output Rail `streaming.enabled` |
|--------|-------------------|--------------------------------|
| **Scope** | LLM token generation | Output rail processing |
| **Required for** | Any streaming | Streaming with output rails |
| **Affects** | How LLM produces tokens | How rails process token chunks |
| **Default** | `False` | `False` |

## Quick Example

When using streaming with output rails, both configurations are required:

```yaml
# Global: Enable LLM streaming
streaming: True

rails:
  output:
    flows:
      - self check output
    # Output rail streaming: Enable chunked processing
    streaming:
      enabled: True
      chunk_size: 200
      context_size: 50
```

## Streaming Configuration Details

The following guides provide detailed documentation for each streaming configuration area.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Global Streaming
:link: global-streaming
:link-type: doc

Enable streaming mode for LLM token generation in config.yml.
:::

:::{grid-item-card} Output Rail Streaming
:link: output-rail-streaming
:link-type: doc

Configure how output rails process streamed tokens in chunked mode.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

global-streaming
output-rail-streaming
```

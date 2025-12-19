---
title: Streaming Configuration
description: Configure streaming for output rail processing in config.yml.
---

# Streaming Configuration

NeMo Guardrails supports streaming out of the box when using the `stream_async()` method. No configuration is required to enable basic streaming.

When you have **output rails** configured, you need to explicitly enable streaming for them to process tokens in chunked mode.

## Quick Example

When using streaming with output rails:

```yaml
rails:
  output:
    flows:
      - self check output
    streaming:
      enabled: True
      chunk_size: 200
      context_size: 50
```

## Streaming Configuration Details

The following guides provide detailed documentation for streaming configuration.

::::{grid} 1 1 2 2
:gutter: 3

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

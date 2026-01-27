---
title:
  page: Streaming Configuration
  nav: Streaming
description: Configure streaming for LLM token generation and output rail processing in config.yml.
topics:
- Configuration
- Streaming
tags:
- Streaming
- Output Rails
- config.yml
- YAML
content:
  type: reference
  difficulty: technical_intermediate
  audience:
  - engineer
  - AI Engineer
---

# Output Streaming Configuration

The NeMo Guardrails library supports streaming out of the box when using the `stream_async()` method. No configuration is required to enable basic streaming.

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

:::{grid-item-card} Streaming LLM Responses
:link: global-streaming
:link-type: doc

Enable and use streaming mode for LLM responses in real-time in the NeMo Guardrails library.
+++
{bdg-secondary}`How To`
:::

:::{grid-item-card} Output Rail Streaming
:link: output-rail-streaming
:link-type: doc

Configure how output rails process streamed tokens in chunked mode.
+++
{bdg-secondary}`Reference`
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

Streaming LLM Responses <global-streaming>
Output Rail Streaming <output-rail-streaming>
```

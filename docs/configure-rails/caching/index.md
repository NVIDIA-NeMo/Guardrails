---
title: Caching Instructions and Prompts
description: Configure in-memory caching for LLM calls and KV cache reuse to improve performance and reduce latency.
---

# Caching Instructions and Prompts

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Memory Model Cache
:link: model-memory-cache
:link-type: doc

Configure in-memory caching to avoid repeated LLM calls for identical prompts using LFU eviction.
:::

:::{grid-item-card} KV Cache Reuse for NemoGuard NIM
:link: kv-cache-reuse
:link-type: doc

Enable KV cache reuse in NVIDIA NIM for LLMs to reduce inference latency for NemoGuard models.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

Caching Instructions <model-memory-cache.md>
KV Cache Reuse for LLM NIM <kv-cache-reuse.md>
```

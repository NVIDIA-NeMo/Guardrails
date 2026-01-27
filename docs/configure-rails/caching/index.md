---
title:
  page: "Caching Instructions and Prompts"
  nav: "Caching"
description: "Configure in-memory caching for LLM calls and KV cache reuse to improve performance and reduce latency."
keywords: ["nemo guardrails caching", "LLM cache", "KV cache reuse", "performance optimization"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "performance"]
content:
  type: how_to
  difficulty: technical_intermediate
  audience: ["engineer"]
---

# Caching Instructions and Prompts

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Memory Model Cache
:link: model-memory-cache
:link-type: doc

Configure in-memory caching to avoid repeated LLM calls for identical prompts using LFU eviction.
+++
{bdg-secondary}`How To`
:::

:::{grid-item-card} KV Cache Reuse
:link: kv-cache-reuse
:link-type: doc

Enable KV cache reuse in NVIDIA NIM for LLMs to reduce inference latency for NemoGuard models.
+++
{bdg-secondary}`How To`
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

Memory Model Cache <model-memory-cache.md>
KV Cache Reuse <kv-cache-reuse.md>
```

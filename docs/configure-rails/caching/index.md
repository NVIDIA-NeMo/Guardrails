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

The NVIDIA NeMo Guardrails library provides two caching strategies to reduce inference latency.
The in-memory model cache stores LLM responses and returns them for repeated prompts without calling the LLM again.
KV cache reuse is a NIM-level optimization that avoids computation of the system prompt on each NemoGuard NIM call.
You can enable either or both strategies independently.

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

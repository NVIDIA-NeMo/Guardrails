---
title:
  page: "Use the Python API"
  nav: "Python API"
description: "Run guardrailed inference using the NeMo Guardrails Python API."
keywords: ["NeMo Guardrails Python API", "LLMRails", "RailsConfig", "guardrailed inference", "Python SDK"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "ai_platforms"]
content:
  type: get_started
  difficulty: technical_intermediate
  audience: ["data_scientist", "engineer"]
---

# Using the NeMo Guardrails Library Python API

This section covers how to use the NeMo Guardrails library Python API to run guardrailed inference and integrate the guardrails into your application.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Overview
:link: overview
:link-type: doc

RailsConfig and LLMRails core classes for generating guarded responses.
:::

:::{grid-item-card} Core Classes
:link: core-classes
:link-type: doc

RailsConfig and LLMRails class reference for loading and running guardrails.
:::

:::{grid-item-card} Generation Options
:link: generation-options
:link-type: doc

Configure logging, LLM parameters, and rail selection for generation.
:::

:::{grid-item-card} Streaming
:link: streaming
:link-type: doc

Stream LLM responses in real-time with the stream_async method.
:::

:::{grid-item-card} Event-Based API
:link: event-based-api
:link-type: doc

Use generate_events for low-level control over guardrails execution.
:::
::::

```{toctree}
:hidden:

Overview <overview>
Core Classes <core-classes>
Generation Options <generation-options>
Streaming <streaming>
Event-Based API <event-based-api>
```

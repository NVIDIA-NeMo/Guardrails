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
+++
{bdg-secondary}`Concept`
:::

:::{grid-item-card} Core Classes
:link: core-classes
:link-type: doc

RailsConfig and LLMRails class reference for loading and running guardrails.
+++
{bdg-secondary}`Reference`
:::

:::{grid-item-card} Generation Options
:link: generation-options
:link-type: doc

Configure logging, LLM parameters, and rail selection for generation.
+++
{bdg-secondary}`Reference`
:::

:::{grid-item-card} Streaming
:link: streaming
:link-type: doc

Stream LLM responses in real-time with the stream_async method.
+++
{bdg-secondary}`Tutorial`
:::

:::{grid-item-card} Check Messages
:link: check-messages
:link-type: doc

Validate messages against input and output rails using check_async and check.
+++
{bdg-secondary}`Reference`
:::

:::{grid-item-card} Event-Based API
:link: event-based-api
:link-type: doc

Use generate_events for low-level control over guardrails execution.
+++
{bdg-secondary}`Reference`
:::
::::

```{toctree}
:hidden:

Overview <overview>
Core Classes <core-classes>
Generation Options <generation-options>
Streaming <streaming>
Check Messages <check-messages>
Event-Based API <event-based-api>
```

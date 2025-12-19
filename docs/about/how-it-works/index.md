---
title:
  page: "How It Works"
  nav: "How It Works"
description: "Learn the sequence diagrams and architecture for building guardrails."
topics: ["AI"]
tags: ["Guardrails", "Architecture", "Colang"]
content:
  type: "Concept"
  difficulty: "Intermediate"
  audience: ["Developer", "Machine Learning Engineer"]
---

# How It Works

The NeMo Guardrails library is for building guardrails for your LLM applications. It provides a set of tools and libraries for building guardrails for your LLM applications.

Read the following pages to learn more about how the library works and how you can use it to build a guardrails system for your LLM applications.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Sequence Diagrams
:link: guardrails-process
:link-type: doc

View sequence diagrams showing input, retrieval, dialog, execution, and output stages.
:::

:::{grid-item-card} Architecture
:link: ../architecture/README
:link-type: doc

Explore the event-driven runtime, canonical messages, and server design.
:::

::::

```{toctree}
:hidden:

Rails Sequence Diagrams <guardrails-process.md>
Detailed Architecture <../architecture/README.md>
```

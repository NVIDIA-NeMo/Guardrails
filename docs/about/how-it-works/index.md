# How It Works

The NeMo Guardrails toolkit is for building guardrails for your LLM applications. It provides a set of tools and libraries for building guardrails for your LLM applications.

Read the following pages to learn more about how the toolkit works and how you can use it to build a guardrails system for your LLM applications.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} How Guardrails Work
:link: how-rails-work
:link-type: doc

Learn how the NeMo Guardrails toolkit applies guardrails at multiple stages of the LLM interaction.
:::

:::{grid-item-card} Guardrails Process
:link: user-guides/guardrails-process
:link-type: doc

Learn about the five main categories of rails (input, dialog, output, retrieval, and execution) and how they work together to protect your LLM applications.
:::

:::{grid-item-card} Architecture
:link: architecture/README
:link-type: doc

Explore the event-driven architecture, canonical forms, LLM interaction patterns, and server design that power NeMo Guardrails.
:::

::::

```{toctree}
:hidden:

Rails Overview  <how-rails-work.md>
Rails Sequence Diagrams <guardrails-process.md>
Detailed Architecture <../architecture/README.md>
```

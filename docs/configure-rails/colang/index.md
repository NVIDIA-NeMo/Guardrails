---
title: Colang Guide
description: Learn Colang, the event-driven language for defining guardrails flows, user messages, and bot responses.
---

# Colang Guide

Colang is an *event-driven interaction modeling language* that is interpreted by a Python runtime.
This section describes how to use Colang to define guardrails flows in `.co` files.

The initial releases of NeMo Guardrails (versions 0.1 through 0.7) use Colang 1.0.
Beginning with version 0.8, NeMo Guardrails introduces support for Colang 2.0, while maintaining Colang 1.0 as the default until Colang completes its beta phase.

| NeMo Guardrails Version | Colang Version |
|-------------------------|----------------|
| 0.1 - 0.7 | 1.0 |
| 0.8 | 2.0-alpha |
| >= 0.9 | 2.0-beta |

## Motivation

Large Language Models (LLMs) are increasingly used in different types of conversational and interactive systems, such as chat-based assistants, voice assistants, multi-modal interactive avatars, non-playable characters in games, and fully autonomous agents.
These applications use the LLMs to do more than generate text responses.
They need to trigger actions and follow complex business processes.

```{image} colang-2/images/use_cases_llms.png
:align: center
:width: 458
:alt: Use cases for LLMs in interactive systems
```

Widely adopted approaches for achieving this include:

1. Generating code and executing it in a sand-boxed environment (for example, generate Python code).
2. Generating the response using specific templates, which allow easier parsing of bot responses and actions that should be taken (for example, Chain of Thought patterns).
3. Function calling and constrained output generation (for example, JSON mode) for models that support it.

Retrieval Augmented Generation (RAG) plays a crucial role by integrating application-level and user-specific context into the generation.
A comprehensive guardrails toolkit for LLMs should seamlessly accommodate all these interaction patterns.

## Configuration Sections

The following sections provide detailed documentation for using Colang:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Colang 2.0 Guide
:link: colang-2/index
:link-type: doc

whats-changed getting-started/index language-reference/index migration-guide
:::

:::{grid-item-card} Colang 1.0 Guide
:link: colang-language-syntax-guide
:link-type: doc

The original Colang syntax for defining user messages, bot messages, and dialog flows.
:::

:::{grid-item-card} Migrating from Colang 1 to Colang 2
:link: colang-2/migration-guide
:link-type: doc

Convert Colang 1.0 configurations to Colang 2.x using the nemoguardrails convert tool.
:::

::::

## Colang 1.0

When referring to Colang, both the language and its runtime environment are implied.
The initial Colang 1.0 language and runtime have several limitations.

**Language limitations:**

- Primarily supports text-based interactions with specialized constructs for user and bot messages.
- Limited support for natural language instructions, such as extracting user-provided values or bot message instructions.
- Lack of support for executing multiple actions or initiating multiple interaction flows concurrently.
- Does not allow the modeling of parallel interaction streams, such as simultaneous chat and avatar posture adjustments in interactive avatar systems.
- Absence of a formal language description.

**Runtime limitations:**

- No explicit state object to manage continuous interaction.
- Performance degrades as the number of events increases.

## Colang 2.0

Colang 2.0 represents a complete overhaul of both the language and runtime.

### Colang 2.0-alpha

Key enhancements include:

- A more powerful flows engine supporting multiple parallel flows and advanced pattern matching over the stream of events.
- A standard library to simplify bot development.
- Smaller set of core abstractions: flows, events, and actions.
- Explicit entry point through the `main` flow and explicit activation of flows.
- Asynchronous actions execution.
- Adoption of terminology and syntax akin to Python to reduce the learning curve for new developers.

### Colang 2.0-beta

Additional enhancements:

- An import mechanism for the standard library to further streamline development.
- The new *generation operator* (`...`).
- Standalone and flow parameter expression evaluation.

**Current limitations** (to be fixed in future releases):

- Guardrails Library is not yet fully usable from within Colang 2.0.
- Some generation options not supported (for example, log activated rails).

### Migration from Alpha to Beta

You can migrate your Colang 2.0-alpha bots to 2.0-beta using the following command:

```bash
nemoguardrails convert "path/to/2.0-alpha/version/bots" --from-version "2.0-alpha"
```

Additionally, you can add the `--validate` flag to check if the migrated files do not raise any Colang syntax errors.

## Interaction Model

While there are many changes in the syntax and the underlying mechanics between Colang 1.0 and Colang 2.0, one core element has remained the same: *interaction model*.

In both Colang 1.0 and Colang 2.0, the interaction between the application (or user) and the LLM is an event-driven one.
Examples of events include: user saying something, the LLM generating a response, triggering an action, the result of an action, the retrieval of additional info, the triggering of a guardrail, and more.
In other words, the evolution of a system is modeled as a series of events, with the guardrails layer responsible for recognizing and enforcing patterns within the stream.

The diagram below depicts a simplified version of the role of the events stream (the boxes with yellow background represent events).

```{image} colang-2/images/guardrails_events_stream.png
:align: center
:width: 649
:alt: Event-driven interaction model showing the flow of events between user, guardrails, and LLM
```

This event-driven interaction model is part of what makes Colang a powerful modeling language, enabling the description of any type of interaction (text-based, voice-based, multi-modal, agent, multi-agent, and so on) and adding guardrails to it.

## Getting Started

If you've used Colang 1.0 before, check out the [What's Changed](colang-2/whats-changed) page.
If not, you can get started with the [Hello World](colang-2/getting-started/hello-world) example.

```{toctree}
:hidden:
:maxdepth: 2

colang-2/index
colang-1/index
usage-examples/index
```

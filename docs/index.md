<!--
  SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Guardrails Library Developer Guide

{bdg-link-primary}`PyPI <https://pypi.org/project/nemoguardrails/>`
{bdg-link-secondary}`GitHub <https://github.com/NVIDIA/NeMo-Guardrails>`

The NeMo Guardrails library is an open-source Python package for adding programmable guardrails to LLM-based applications. It intercepts inputs and outputs, applies configurable safety checks, and blocks or modifies content based on defined policies.

## About the NeMo Guardrails Library

Learn about the library and its capabilities in the following sections.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Overview
:link: about/overview
:link-type: doc

Add programmable guardrails to LLM applications with this open-source Python library.
:::

:::{grid-item-card} Guardrail Types
:link: about/rail-types
:link-type: doc

Apply input, retrieval, dialog, execution, and output rails to protect LLM applications.
:::

:::{grid-item-card} How It Works
:link: about/how-it-works/index
:link-type: doc

Learn the sequence diagrams and architecture for building guardrails.
:::

:::{grid-item-card} Supported LLMs
:link: about/supported-llms
:link-type: doc

Connect to NVIDIA NIM, OpenAI, Azure, Anthropic, HuggingFace, and LangChain providers.
:::

::::

## Get Started

Follow these steps to start using the NeMo Guardrails library.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Install
:link: getting-started/installation-guide
:link-type: doc

Install NeMo Guardrails with pip, configure your environment, and verify the installation.
:::

:::{grid-item-card} Tutorials
:link: getting-started/tutorials/index
:link-type: doc

Follow hands-on tutorials to deploy content safety, topic control, and jailbreak detection.
:::
::::

---

## Next Steps

Once you've completed the get-started tutorials, explore the following areas to deepen your understanding.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Configure Rails
:link: configure-rails/overview
:link-type: doc

Learn to write config.yml, Colang flows, and custom actions for guardrails.
:::

:::{grid-item-card} Run Rails
:link: run-rails/index
:link-type: doc

Use RailsConfig and LLMRails classes to load configurations and generate guarded responses.
:::

:::{grid-item-card} Evaluate
:link: evaluation/README
:link-type: doc

Measure accuracy and performance of dialog, fact-checking, moderation, and hallucination rails.
:::

:::{grid-item-card} Logging
:link: observability/logging/index
:link-type: doc

Debug guardrails with verbose mode, explain method, and generation log options.
:::

:::{grid-item-card} Deploy
:link: deployment/index
:link-type: doc

Deploy guardrails using the local server, Docker containers, or production microservices.
:::

:::{grid-item-card} LangChain
:link: integration/langchain/index
:link-type: doc

Integrate NeMo Guardrails with LangChain chains, runnables, and LangGraph workflows.
:::

::::

```{toctree}
:caption: About NeMo Guardrails Library
:name: About NeMo Guardrails Library
:hidden:

Overview <about/overview.md>
Rail Types <about/rail-types.md>
How It Works <about/how-it-works/index.md>
Supported LLMs <about/supported-llms.md>
Release Notes <about/release-notes.md>
```

```{toctree}
:caption: Get Started
:name: Get Started
:hidden:

getting-started/installation-guide
getting-started/tutorials/index
getting-started/integrate-into-application
```

```{toctree}
:caption: Configure Rails
:name: Configure Rails
:hidden:

Before Configuring Rails <configure-rails/before-configuration.md>
Configuration Overview <configure-rails/overview.md>
Core Configuration <configure-rails/yaml-schema/index.md>
Custom Actions <configure-rails/actions/index.md>
Custom Initialization <configure-rails/custom-initialization/index.md>
Colang <configure-rails/colang/index.md>
Other Configurations <configure-rails/other-configurations/index.md>
Caching <configure-rails/caching/index.md>
```

```{toctree}
:caption: Run Rails
:name: Run Rails
:hidden:

Run Rails <run-rails/index.md>
Core Classes <run-rails/core-classes.md>
Generation Options <run-rails/generation-options.md>
Streaming <run-rails/streaming.md>
Event-based API <run-rails/event-based-api.md>
```

```{toctree}
:caption: Evaluation
:name: Evaluation
:hidden:

evaluation/README
evaluation/llm-vulnerability-scanning
```

```{toctree}
:caption: Observability
:name: Observability
:hidden:

Logging <observability/logging/index.md>
Tracing <observability/tracing/index.md>
```

```{toctree}
:caption: Deployment Guides
:hidden:

Deployment Options <deployment/index>
Local Server Setup <deployment/local-server/index>
Using Docker <deployment/using-docker>
Using NeMo Guardrails Microservice <deployment/using-microservice>
Blueprint with NemoGuard NIMs <integration/safeguarding-ai-virtual-assistant-blueprint>
```

```{toctree}
:caption: Integration with Third-Party Libraries
:hidden:

LangChain <integration/langchain/index.md>
AlignScore <integration/align-score-deployment>
Integrate LangChain Tools <integration/tools-integration.md>
```

```{toctree}
:caption: Security
:name: Security
:hidden:

security/guidelines
```

```{toctree}
:caption: Reference
:name: Reference
:hidden:

troubleshooting
faqs
python-api/index
cli/index
glossary
```

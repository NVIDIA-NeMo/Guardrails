<!--
  SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Guardrails Toolkit Developer Guide

The NeMo Guardrails toolkit is an open-source Python package for adding programmable guardrails to LLM-based applications. It intercepts inputs and outputs, applies configurable safety checks, and blocks or modifies content based on defined policies.

## About the NeMo Guardrails Toolkit

Learn about the toolkit and its capabilities in the following sections.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Overview
:link: about/overview
:link-type: doc

Learn about the NeMo Guardrails toolkit and its capabilities.
:::

:::{grid-item-card} Use Cases
:link: about/use-cases
:link-type: doc

Browse the different use cases of the NeMo Guardrails toolkit.
:::

:::{grid-item-card} How It Works
:link: about/how-it-works/index
:link-type: doc

Learn how the NeMo Guardrails toolkit works.
:::

:::{grid-item-card} Supported LLMs and Providers
:link: about/supported-llms
:link-type: doc

Browse the LLMs and their providers supported by the toolkit.
:::

::::

## Get Started

Follow these steps to start using the NeMo Guardrails toolkit.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Install
:link: getting-started/installation-guide
:link-type: doc

Install the toolkit with pip and set up your environment.
:::

:::{grid-item-card} Tutorials
:link: getting-started/tutorials/index
:link-type: doc

Follow hands-on tutorials to build your first guardrails configuration.
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

Learn to write config.yml, Colang flows, and custom actions.
:::

:::{grid-item-card} Run Rails
:link: run-rails/index
:link-type: doc

Use the Python SDK and understand core classes like RailsConfig and LLMRails.
:::

:::{grid-item-card} Evaluate
:link: evaluation/README
:link-type: doc

Evaluate the performance of the rails.
:::

:::{grid-item-card} Observability
:link: observability/logging/index
:link-type: doc

Monitor and troubleshoot your guardrails applications.
:::

:::{grid-item-card} Deploy
:link: deployment/index
:link-type: doc

Deploy your guardrails using the toolkit's local server, Docker, or as a production microservice.
:::

:::{grid-item-card} Integrate
:link: integration/langchain/index
:link-type: doc

Connect with LangChain, LangGraph, and other frameworks.
:::

::::

```{toctree}
:caption: About NeMo Guardrails Toolkit
:name: About NeMo Guardrails Toolkit
:hidden:

Overview <about/overview.md>
Use Cases <about/use-cases.md>
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

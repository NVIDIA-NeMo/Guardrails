<!--
  SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Guardrails Toolkit Developer Guide

## Introduction

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} About NeMo Guardrails Toolkit
:link: about/index
:link-type: doc

This section covers the basics of the NeMo Guardrails toolkit.
:::

:::{grid-item-card} Get Started
:link: getting-started/index
:link-type: doc

Get started with the NeMo Guardrails toolkit.
:::

::::

## Using the NeMo Guardrails Toolkit

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Configuration Overview
:link: configure-rails/index
:link-type: doc

Prepare configuration files including config.yml, Colang flows, actions.py, config.py, and knowledge base documents.
:::

:::{grid-item-card} Run Rails
:link: run-rails/index
:link-type: doc

This section covers how to use the NeMo Guardrails toolkit programmatically through the Python API. Learn about the core classes, generation methods, and advanced features for integrating...
:::

:::{grid-item-card} Deployment Options
:link: deployment/index
:link-type: doc

You can deploy the NeMo Guardrails toolkit in the following ways.
:::

:::{grid-item-card} Evaluation
:link: evaluation/index
:link-type: doc

README llm-vulnerability-scanning
:::

::::

```{toctree}
:caption: About NeMo Guardrails Toolkit
:name: About NeMo Guardrails Toolkit
:hidden:

Overview <about/overview.md>
How It Works <about/how-it-works.md>
Use Cases <about/use-cases.md>
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
Configuration Overview <configure-rails/index.md>
Core Configuration <configure-rails/yaml-schema/index.md>
Custom Actions <configure-rails/actions/index.md>
Custom Initialization <configure-rails/custom-initialization/index.md>
Colang <configure-rails/colang/index.md>
Other Configurations <configure-rails/other-configurations/index.md>
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
Tools Integration <run-rails/tools-integration.md>
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
```

```{toctree}
:caption: Integration with Third-Party Libraries
:hidden:

LangChain <integration/langchain/index.md>
Vertex AI <integration/vertexai.md>
AlignScore <integration/align-score-deployment>
Llama Guard <integration/llama-guard-deployment>
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

python-api/index
cli/index
glossary
faqs
```

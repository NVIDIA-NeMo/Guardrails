<!--
  SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Guardrails Library Developer Guide

{bdg-link-primary}`PyPI <https://pypi.org/project/nemoguardrails/>`
{bdg-link-secondary}`GitHub <https://github.com/NVIDIA-NeMo/Guardrails>`

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
:link: about/how-it-works
:link-type: doc

High level explanation of how Guardrails works.
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

Follow hands-on tutorials to deploy Nemotron Content Safety, Nemotron Topic Control, and Nemotron Jailbreak Detect NIMs.
:::
::::

---

## Next Steps

Once you've completed the get-started tutorials, explore the following areas to deepen your understanding.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} About Configuring Guardrails
:link: configure-rails/index
:link-type: doc

Configure YAML files, Colang flows, custom actions, and other components to control LLM behavior.
:::

:::{grid-item-card} About Running Guardrailed Inference
:link: run-rails/index
:link-type: doc

Run guardrailed inference using the Python API or Guardrails API server.
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

:::{grid-item-card} Deployment Options
:link: deployment/index
:link-type: doc

Deploy guardrails using the local API server, Docker containers, or production microservices.
:::

:::{grid-item-card} LangChain Frameworks
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
How It Works <about/how-it-works.md>
Guardrail Types <about/rail-types.md>
Supported LLMs <about/supported-llms.md>
Release Notes <about/release-notes.md>
```

```{toctree}
:caption: Get Started
:name: Get Started
:hidden:

Install <getting-started/installation-guide>
Tutorials <getting-started/tutorials/index>
Integrate <getting-started/integrate-into-application.md>
```

```{toctree}
:caption: Configure Guardrails
:name: Configure Guardrails
:hidden:

About Configuring Guardrails <configure-rails/index.md>
Overview <configure-rails/overview.md>
Prerequisites <configure-rails/before-configuration.md>
Configuring YAML File <configure-rails/yaml-schema/index.md>
YAML Schema Reference <configure-rails/configuration-reference.md>
Guardrail Catalog <configure-rails/guardrail-catalog.md>
Custom Actions <configure-rails/actions/index.md>
Custom Initialization <configure-rails/custom-initialization/index.md>
Colang <configure-rails/colang/index.md>
Other Configurations <configure-rails/other-configurations/index.md>
Caching Instructions and Prompts <configure-rails/caching/index.md>
Exceptions and Error Handling <configure-rails/exceptions.md>
```

```{toctree}
:caption: Run Guardrailed Inference
:name: Run Guardrailed Inference
:hidden:

About Running Guardrailed Inference <run-rails/index.md>
Python API <run-rails/using-python-apis/index.md>
Guardrails API Server <run-rails/using-fastapi-server/index.md>
```

```{toctree}
:caption: Evaluation
:name: Evaluation
:hidden:

Evaluate <evaluation/README>
Vulnerability Scanning <evaluation/llm-vulnerability-scanning>
```

```{toctree}
:caption: Observability
:name: Observability
:hidden:

Logging <observability/logging/index.md>
Tracing <observability/tracing/index.md>
```

```{toctree}
:caption: More Deployment Options
:hidden:

Deployment Options <deployment/index>
Docker <deployment/using-docker>
NeMo Microservice <deployment/using-microservice>
Blueprint Integration <integration/safeguarding-ai-virtual-assistant-blueprint>
```

```{toctree}
:caption: Integration with Third-Party Libraries
:hidden:

LangChain Frameworks <integration/langchain/index.md>
AlignScore <integration/align-score-deployment>
Tools Integration <integration/tools-integration.md>
```

```{toctree}
:caption: Reference
:name: Reference
:hidden:

Architecture <reference/colang-architecture-guide.md>
Sequence Diagrams <reference/guardrails-sequence-diagrams.md>
Use Case Diagrams <reference/use-case-diagrams.md>
Python API <reference/python-api/index>
CLI <reference/cli/index>
Guardrails API Server Endpoints <reference/api-server-endpoints/index>
```

```{toctree}
:caption: Troubleshooting
:name: Troubleshooting
:hidden:

Troubleshooting <troubleshooting>
```

```{toctree}
:caption: Resources
:name: Resources
:hidden:

FAQs <resources/faqs.md>
Glossary <resources/glossary.md>
Security <resources/security/guidelines.md>
```

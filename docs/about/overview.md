---
title: Overview
description: Learn about the NeMo Guardrails toolkit and its capabilities.
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Overview of the NeMo Guardrails Toolkit

The NVIDIA NeMo Guardrails toolkit is an open-source Python package for adding programmable guardrails to LLM-based applications. It intercepts inputs and outputs, applies configurable safety checks, and blocks or modifies content based on defined policies.

```{mermaid}
%%{init: {'theme': 'neutral', 'themeVariables': { 'background': 'transparent' }}}%%

flowchart TB
  A("Application Code")
  B("NeMo Guardrails Toolkit")
  C("Large Language Model (LLM)")

  A <--> B

subgraph NemoGuard["NemoGuard NIMs"]
  direction TB
  D("NemoGuard Content Safety")
  E("NemoGuard Topic Control")
  F("NemoGuard Jailbreak Detection")
end

  B <--> NemoGuard
  NemoGuard <--> C

  style A fill:#d8d8e8,stroke:#999
  style B fill:#f0f7e6,stroke:#76b900,stroke-width:2px
  style C fill:#d8d8e8,stroke:#999
  style D fill:#f0f7e6,stroke:#76b900
  style E fill:#f0f7e6,stroke:#76b900
  style F fill:#f0f7e6,stroke:#76b900
```

*Application code interacting with LLMs through the NeMo Guardrails toolkit.*

---

## What You Can Do with the NeMo Guardrails Toolkit

The following are the top use cases of the NeMo Guardrails toolkit that you can apply to protect your LLM applications.

::::{grid} 1 1 2 2
:gutter: 3
:class-container: sd-equal-height

:::{grid-item-card} Text Content Safety
:link: ../getting-started/tutorials/nemotron-safety-guard-deployment
:link-type: doc

Deploy Nemotron Safety Guard to detect harmful content in text inputs and outputs.
:::

:::{grid-item-card} Multimodal Content Safety
:link: ../getting-started/tutorials/multimodal
:link-type: doc

Add safety checks to images and text using vision models as LLM-as-a-judge.
:::

:::{grid-item-card} Jailbreak Detection
:link: ../getting-started/tutorials/nemoguard-jailbreakdetect-deployment
:link-type: doc

Deploy NemoGuard Jailbreak Detection NIM to block adversarial prompts.
:::

:::{grid-item-card} Topic Control
:link: ../getting-started/tutorials/nemoguard-topiccontrol-deployment
:link-type: doc

Deploy NemoGuard Topic Control NIM to restrict conversations to allowed topics.
:::

:::{grid-item-card} PII Handling
Identify and mask Personally Identifiable Information in inputs and outputs using regex patterns, Presidio integration, or custom detection logic.
:::

:::{grid-item-card} Knowledge Base / RAG
In RAG scenarios, verify LLM responses against retrieved source documents to detect unsupported claims or hallucinations.
:::

:::{grid-item-card} Agentic Workflows
Apply execution rails to secure LLM agents that perform multi-step reasoning or interact with external systems. Validate agent decisions, restrict allowed actions, and enforce policies before execution proceeds.
:::

:::{grid-item-card} Tool Integration
Validate inputs and outputs when the LLM calls external tools or APIs. Execution rails intercept tool calls to check parameters, sanitize inputs, and filter responses before returning results to the LLM.
:::

::::

---

## Tools

The following are the tools you can use to interact with the NeMo Guardrails toolkit.

### Python SDK

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

response = rails.generate(
    messages=[{"role": "user", "content": "Hello!"}]
)
```

The `generate` method accepts the same message format as the OpenAI Chat Completions API.

### CLI Server

```bash
nemoguardrails server --config ./config --port 8000
```

The server exposes an HTTP API compatible with OpenAI's `/v1/chat/completions` endpoint.

---

## Toolkit vs Microservice

This documentation covers the open-source NeMo Guardrails toolkit. The NeMo Guardrails Microservice is a separate product that packages the same core functionality for Kubernetes deployment.

|                  | Toolkit                          | Microservice                     |
|------------------|----------------------------------|----------------------------------|
| Distribution     | PyPI (`pip install`)             | Container image                  |
| Deployment       | Self-managed                     | Kubernetes with Helm             |
| Scaling          | Application-level                | Managed by orchestrator          |
| Configuration    | Same YAML/Colang format          | Same YAML/Colang format          |

Configurations are portable between the toolkit and microservice.

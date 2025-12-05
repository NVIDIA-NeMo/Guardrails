<!--
  SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Overview

The NeMo Guardrails toolkit is an open-source Python package for adding programmable guardrails to LLM-based applications. It intercepts inputs and outputs, applies configurable safety checks, and blocks or modifies content based on defined policies.

```{image} ../_static/images/programmable_guardrails.png
:alt: "Programmable Guardrails"
:width: 800px
:align: center
```

---

## Capabilities

The following are the capabilities of the NeMo Guardrails toolkit.

### Content Filtering

Apply input and output rails to detect and block harmful, toxic, or policy-violating content. Rails can reject content entirely or modify it (for example, mask sensitive data) before processing continues.

### Jailbreak Detection

Detect adversarial prompts designed to bypass LLM safety measures. The toolkit supports both LLM-based self-check methods and dedicated NemoGuard NIM models for jailbreak detection.

### Topic Control

Restrict conversations to allowed topics. Define canonical user intents and configure the system to block or redirect off-topic requests.

### PII Handling

Identify and mask Personally Identifiable Information in inputs and outputs using regex patterns, Presidio integration, or custom detection logic.

### Fact Checking

In RAG scenarios, verify LLM responses against retrieved source documents to detect unsupported claims or hallucinations.

### Agentic Workflows

Apply execution rails to secure LLM agents that perform multi-step reasoning or interact with external systems. Validate agent decisions, restrict allowed actions, and enforce policies before execution proceeds.

### Tool Integration

Validate inputs and outputs when the LLM calls external tools or APIs. Execution rails intercept tool calls to check parameters, sanitize inputs, and filter responses before returning results to the LLM.

---

## Usage

The following are the ways to use the NeMo Guardrails toolkit.

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

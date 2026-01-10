---
title:
  page: "About Running Guardrailed Inference"
  nav: "About Running Guardrailed Inference"
description: "Run guardrailed inference using the Python API or FastAPI server."
keywords: ["NeMo Guardrails", "guardrailed inference", "LLMRails", "RailsConfig", "FastAPI server"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "ai_platforms"]
content:
  type: get_started
  difficulty: technical_intermediate
  audience: ["data_scientist", "engineer"]
---

# About Running Guardrailed Inference Using the NeMo Guardrails Library Tools

After you [configure your guardrails](../configure-rails/index.md), you can run guardrailed inference using the tools provided by the NeMo Guardrails library: the Python API and the FastAPI server.

These tools enable you to interact with your LLM as usual—sending prompts and receiving responses—while the guardrails system monitors and controls all communication in the background.
The guardrails intercept inputs and outputs, apply your configured rails, and ensure that interactions remain within the boundaries you defined.

## Choosing the Right Tool

| Use Case | Recommended Tool | Benefits |
|----------|------------------|----------|
| Embedding guardrails directly in a Python application | Python API | No network overhead; guardrails run in the same process as your application. |
| Rapid prototyping and development | Python API | Less setup; test configurations in a notebook or script without starting a server. |
| Fine-grained control over generation | Python API | Access to `generate_events()`, custom streaming handlers, and internal state. |
| Production deployments with multiple clients | FastAPI Server | Handles concurrent requests; can be load-balanced and horizontally scaled. |
| Non-Python clients (JavaScript, Go, Java, etc.) | FastAPI Server | Language-agnostic REST API; no Python dependencies required for clients. |
| Microservices architecture | FastAPI Server | Independent service with single responsibility; enables separate scaling. |
| Centralized guardrails management | FastAPI Server | Update configurations once; all clients benefit without redeployment. |

## Next Steps

After you've chosen the right tool, proceed to one of the following guides to learn how to run guardrailed inference.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Python API
:link: using-python-apis/index
:link-type: doc

Use the Python API to run guardrailed inference.
:::

:::{grid-item-card} FastAPI Server
:link: using-fastapi-server/index
:link-type: doc

Use the FastAPI server to run guardrailed inference.
:::
::::

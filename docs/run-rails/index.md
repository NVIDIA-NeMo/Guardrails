---
title:
  page: "About Running Guardrailed Inference"
  nav: "About Running Guardrailed Inference"
description: "Run guardrailed inference using the Python API or Guardrails API server."
keywords: ["NeMo Guardrails", "guardrailed inference", "LLMRails", "RailsConfig", "Guardrails API server"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "ai_platforms"]
content:
  type: get_started
  difficulty: technical_intermediate
  audience: ["data_scientist", "engineer"]
---

# About Running Guardrailed Inference Using the NeMo Guardrails Library Tools

After you [configure your guardrails](../configure-rails/index.md), you can run guardrailed inference using the tools provided by the NeMo Guardrails library: the Python API and the Guardrails API server.

These tools enable you to interact with your application's main LLM as usual by sending prompts and receiving responses while the guardrails system monitors and controls all communication in the background.
The guardrails intercept inputs and outputs between uses and the LLM and the execution of actions done by the LLM. The tools apply your configured guardrails and ensure that any inputs, generated responses, or actions remain within the boundaries you defined.

---

## Install the NeMo Guardrails Library

To use the Python API or the Guardrails API server, you need to install the NeMo Guardrails library. See [](../getting-started/installation-guide.md) for instructions.

---

## Choosing the Right Tool

Both the Python API and the Guardrails API server are production-ready approaches for integrating guardrails into your application.

### Python API for Edge and Embedded Applications

Build guardrails directly into your Python application. You call guardrails functions in your code, and everything runs in the same process with no network overhead.

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("path/to/config")
rails = LLMRails(config)

response = rails.generate(messages=[
    {"role": "user", "content": "Hello!"}
])
```

This approach is best for:

- Edge deployments where no networked backend is available.
- Applications that require low latency and no external dependencies.
- Rapid prototyping and development in notebooks or scripts.
- Fine-grained control over generation and custom streaming handlers.

### Guardrails API Server for Networked and Multi-Client Applications

The Guardrails API server provides a RESTful API interface to the guardrails library. Instead of calling Python functions directly, you make HTTP requests to the server endpoints, making integration straightforward for any language or platform.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"config_id": "content_safety", "messages": [{"role": "user", "content": "Hello!"}]}'
```

This approach is best for:

- Networked systems where multiple clients need access to guardrails.
- Non-Python clients such as JavaScript, Go, and Java that communicate through HTTP.

---

## Next Steps

After choosing the right tool for your implementation model, proceed to the corresponding guide from the following.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Python API
:link: using-python-apis/index
:link-type: doc

Run guardrailed inference using the NeMo Guardrails Python API.
+++
{bdg-secondary}`Get Started`
:::

:::{grid-item-card} Guardrails API Server
:link: using-fastapi-server/index
:link-type: doc

Expose guardrails through an HTTP API using the Guardrails API server.
+++
{bdg-secondary}`Get Started`
:::
::::

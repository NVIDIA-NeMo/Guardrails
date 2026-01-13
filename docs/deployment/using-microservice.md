---
title:
  page: "NeMo Guardrails Microservice for Production"
  nav: "NeMo Microservice"
description: "Deploy guardrails as a production-ready microservice using the NeMo microservices platform."
keywords: ["NeMo Guardrails", "microservice", "production", "NeMo Microservices Platform", "NMP"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "ai_platforms"]
content:
  type: how_to
  difficulty: technical_advanced
  audience: ["devops_engineer", "ai_engineer"]
---

# NeMo Guardrails Microservice for Production

The NeMo Guardrails microservice in the [NeMo microservices platform](https://docs.nvidia.com/nemo/microservices/latest/about/index.html) is the production-grade, high availability, scalable, end-to-end data flywheel deployment option, available for the [NVIDIA AI Enterprise plan](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/).

The NeMo microservices platform provides a fully-featured guardrails microservice that provides the same capabilities as the NeMo Guardrails library through its own API, as the guardrails microservice is built on top of the library.

| Capability                               | NeMo Guardrails Library API Server | NeMo Guardrails Microservice |
|------------------------------------------|:----------------------:|:---------------------------:|
| REST API for guardrailed chat completions            | ✓                      | ✓                           |
| Guardrails configuration management  | By managing the config folder structure                      | Through the platform's config API endpoints                           |
| NeMo microservices platform integration               | No need                     | Required                           |
| Production-grade reliability             | Limited (self-managed and no high availability support out of the box)                | Supported by the platform                           |
| Support                       | Open source community                      | NVIDIA AI Enterprise plan                           |

## When to Use Each Option

Use the following guidelines to decide when to use each option.

### NeMo Guardrails Library API Server

The [Guardrails API server](../run-rails/using-fastapi-server/index.md) included in the NeMo Guardrails library is suitable for:

- Direct integration of guardrails into your application.
- Proof-of-concept deployments.
- Development and testing environments.
- Evaluating guardrails configurations before production.

For this option, refer to the following topics:

- [](../run-rails/using-fastapi-server/index.md)
- [](../reference/api-server-endpoints/index.md)

### NeMo Guardrails Microservice

The NeMo Guardrails microservice is recommended for:

- Production deployments requiring enterprise-grade reliability.
- Organizations already using or planning to adopt the NeMo microservices platform.

:::{important}
The NeMo Guardrails microservice operates within the NeMo microservices platform ecosystem and requires other microservices to function. It is not a standalone solution.
:::

For deployment instructions and API documentation for the NeMo Guardrails microservice, refer to [NVIDIA NeMo Microservices Documentation](https://docs.nvidia.com/nemo/microservices/latest/about/index.html).

---
title:
  page: "Use the Guardrails API Server"
  nav: "Guardrails API Server"
description: "Expose guardrails through an HTTP API using the Guardrails API server."
keywords: ["NeMo Guardrails server", "FastAPI", "REST API", "chat completions", "guardrails HTTP"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "ai_platforms"]
content:
  type: get_started
  difficulty: technical_intermediate
  audience: ["data_scientist", "engineer"]
---

# Use the Guardrails API Server

The NeMo Guardrails library includes the Guardrails API server that exposes guardrails through an HTTP API.
This section covers how to run the server and interact with it.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Overview
:link: overview
:link-type: doc

The Guardrails API server is a tool for running guardrails in a secure, isolated environment.
+++
{bdg-secondary}`Concept`
:::

:::{grid-item-card} Run the Server
:link: run-guardrails-server
:link-type: doc

Start the Guardrails API server, configure CORS, and enable auto-reload.
+++
{bdg-secondary}`Tutorial`
:::

:::{grid-item-card} Chat Completions
:link: chat-with-guardrailed-model
:link-type: doc

Send chat requests, use streaming, and manage conversation threads.
+++
{bdg-secondary}`Tutorial`
:::

:::{grid-item-card} List Configurations
:link: list-guardrail-configs
:link-type: doc

Retrieve available guardrails configurations from the server.
+++
{bdg-secondary}`Reference`
:::

:::{grid-item-card} Actions Server
:link: actions-server
:link-type: doc

Run guardrail actions in a secure, isolated environment.
+++
{bdg-secondary}`Tutorial`
:::

::::

```{toctree}
:hidden:

Overview <overview>
Run the Server <run-guardrails-server>
Chat Completions <chat-with-guardrailed-model>
List Configurations <list-guardrail-configs>
Actions Server <actions-server>
```

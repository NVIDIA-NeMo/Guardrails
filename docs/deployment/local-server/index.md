---
title:
  page: "Local Server Setup"
  nav: "Local Server"
description: "Set up and run guardrails and actions servers locally for development and testing."
topics: ["Deployment", "AI Safety"]
tags: ["Server", "FastAPI", "Local Development"]
content:
  type: "How-To"
  difficulty: "Beginner"
  audience: ["Developer", "AI Engineer"]
---

# Local Server Setup

The NeMo Guardrails library enables you to create a guardrails local server and deploy it using a **guardrails server** and an **actions server**.

## Overview

| Server | Purpose | Default Port |
|--------|---------|--------------|
| **Guardrails Server** | Loads guardrails configurations and exposes HTTP API for chat completions | 8000 |
| **Actions Server** | Runs custom actions securely in a separate environment | 8001 |

## Sections

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Guardrails Server
:link: guardrails-server
:link-type: doc

Configure and run the guardrails server with HTTP API endpoints and Chat UI.
:::

:::{grid-item-card} Actions Server
:link: actions-server
:link-type: doc

Deploy actions in a secure, isolated environment separate from the guardrails server.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

Guardrails Server <guardrails-server>
Actions Server <actions-server>
```

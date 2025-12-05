# Local Server Setup

The NeMo Guardrails toolkit enables you to create a guardrails local server and deploy it using a **guardrails server** and an **actions server**.

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

The Guardrails server loads a predefined set of guardrails configurations at startup and exposes an HTTP API to use them. The server uses [FastAPI](https://fastapi.tiangolo.com/), and the...
:::

:::{grid-item-card} Actions Server
:link: actions-server
:link-type: doc

The Actions Server enables you to run the actions invoked from the guardrails more securely (see [Security Guidelines](../../security/guidelines.md) for more details). The action server should be...
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

guardrails-server
actions-server
```

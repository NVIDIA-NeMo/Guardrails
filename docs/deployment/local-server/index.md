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

Configure and run the main guardrails server with FastAPI, including endpoints, CORS, threads, and Chat UI.
:::

:::{grid-item-card} Actions Server
:link: actions-server
:link-type: doc

Deploy a separate server to run custom actions securely, with endpoints for listing and executing actions.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

guardrails-server
actions-server
```

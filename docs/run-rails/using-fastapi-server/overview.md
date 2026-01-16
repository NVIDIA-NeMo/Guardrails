---
title:
  page: "Overview of the Guardrails API Server"
  nav: "Overview"
description: "The Guardrails API server is a tool for running guardrails in a secure, isolated environment."
keywords: ["NeMo Guardrails server", "FastAPI", "REST API", "chat completions", "guardrails HTTP"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "ai_platforms"]
content:
  type: concept
  difficulty: technical_intermediate
  audience: ["data_scientist", "engineer"]
---

# Overview of the NeMo Guardrails Library API Server

The NeMo Guardrails API server:

- Loads guardrails configurations at startup.
- Exposes a REST API compatible with OpenAI's chat completions format.
- Includes a built-in Chat UI for testing.
- Supports multiple configurations and combining them per-request.

## Quick Start

The following steps show how to start the NeMo Guardrails API server using the provided configuration files and test it by sending requests to the endpoints.

### Prerequisites

Meet the following prerequisites to use the NeMo Guardrails API server.

1. If you haven't already, install the NeMo Guardrails library with the `nvidia` extra, following the instructions in [](../../getting-started/installation-guide.md).

2. Set up an environment variable for your NVIDIA API key.

    ```console
    export NVIDIA_API_KEY="your-nvidia-api-key"
    ```

    This is required to access NVIDIA-hosted models on [build.nvidia.com](https://build.nvidia.com). The provided example configurations ([examples/configs](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/configs)) and code examples throughout the documentation use NVIDIA-hosted models.

### Start the Server

Point the server to a parent directory containing multiple configuration subdirectories:

```console
$ cd NeMo-Guardrails
$ nemoguardrails server --config examples/configs
```

List available configurations:

```console
$ curl http://localhost:8000/v1/rails/configs

[
  {"id": "content_safety"},
  {"id": "jailbreak_detection"},
  {"id": "topic_safety"},
  {"id": "llama_guard"},
  ...
]
```

Each subdirectory with a `config.yml` or `config.yaml` file becomes an available config ID.

### Send a Request

Send a chat completion request to the server:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "content_safety",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### View the Chat UI

Open `http://localhost:8000` in your browser to access the built-in Chat UI for testing.

## Related Topics

- [](../../reference/api-server-endpoints/index.md)

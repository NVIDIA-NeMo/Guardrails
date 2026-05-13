---
title:
  page: "Overview of the NVIDIA NeMo Guardrails Library API Server"
  nav: "Overview"
description: "The NVIDIA NeMo Guardrails library API server runs guardrails in a secure, isolated environment."
keywords: ["NVIDIA NeMo Guardrails library server", "FastAPI", "REST API", "chat completions", "guardrails HTTP"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "ai_platforms"]
content:
  type: concept
  difficulty: technical_intermediate
  audience: ["data_scientist", "engineer"]
---

# Overview of the NVIDIA NeMo Guardrails Library API Server

The NVIDIA NeMo Guardrails library API server provides the following capabilities:

- Loads guardrails configurations at startup.
- Exposes an [OpenAI-compatible REST API](https://platform.openai.com/docs/api-reference/chat/create) for chat completions and model listing.
- Works with the [OpenAI Python SDK](https://github.com/openai/openai-python). Use `OpenAI(base_url="http://localhost:8000/v1")`.
- Includes a built-in chat UI for testing.
- Supports multiple configurations and combines them for each request.

## Quick Start

The following steps show how to start the NVIDIA NeMo Guardrails library API server with the provided configuration files and send test requests to the endpoints.

### Prerequisites

Meet the following prerequisites before you use the NVIDIA NeMo Guardrails library API server.

1. Install the NVIDIA NeMo Guardrails library with the `server` extra. For instructions, refer to [Extra Dependencies](../../getting-started/installation-guide.md#extra-dependencies).

   ```bash
   pip install nemoguardrails[server]
   ```

2. Set the environment variable for your NVIDIA API key.

    ```console
    export NVIDIA_API_KEY="your-nvidia-api-key"
    ```

    This key is required to access NVIDIA-hosted models on [build.nvidia.com](https://build.nvidia.com). The provided [example configurations](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/configs) and code examples throughout the documentation use NVIDIA-hosted models.

### Start the Server

Follow these steps to start the server:

1. Point the server to a parent directory that contains multiple configuration subdirectories:

    ```console
    $ cd Guardrails
    $ nemoguardrails server --config examples/configs
    ```

1. To check if the server is running and list the available configurations, use the following command:

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

Each subdirectory that contains a `config.yml` or `config.yaml` file becomes an available configuration ID.

### Send a Request

Send a chat completion request to the server:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "guardrails": {
      "config_id": "content_safety"
    }
  }'
```

### View the Chat UI

Open `http://localhost:8000` in your browser to access the built-in chat UI for testing.

## Related Topics

- [API Server Endpoints](../../reference/api-server-endpoints/index.md)
- [Introduction to LLM Benchmarking](https://docs.nvidia.com/nim/benchmarking/llm/latest/overview.html)

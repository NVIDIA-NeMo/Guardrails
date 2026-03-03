---
title:
  page: "List Available Models"
  nav: "List Models"
description: "Query the available LLM models from the configured provider."
keywords: ["list models", "guardrails API", "OpenAI compatible", "model discovery", "v1/models"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "ai_platforms"]
content:
  type: reference
  difficulty: technical_intermediate
  audience: ["data_scientist", "engineer"]
---

# List Available Models

Use the `GET /v1/models` endpoint to retrieve the list of LLM models available from the configured provider.
This endpoint is compatible with the [OpenAI Models API](https://platform.openai.com/docs/api-reference/models/list)
and proxies the request to the upstream model provider configured via environment variables.

## Request

```bash
curl http://localhost:8000/v1/models
```

No request body or query parameters are required.

## Response

The endpoint returns a standard OpenAI models list:

```json
{
  "data": [
    {
      "id": "meta/llama-3.1-8b-instruct",
      "object": "model",
      "created": 1700000000,
      "owned_by": "system"
    },
    {
      "id": "meta/llama-3.1-70b-instruct",
      "object": "model",
      "created": 1700000000,
      "owned_by": "system"
    }
  ]
}
```

## Using Python

```python
import requests

base_url = "http://localhost:8000"

response = requests.get(f"{base_url}/v1/models")
models = response.json()

print("Available models:")
for model in models["data"]:
    print(f"  - {model['id']}")
```

## Using the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-used"
)

models = client.models.list()

for model in models.data:
    print(model.id)
```

## Provider Configuration

The server determines which upstream provider to query based on the `MAIN_MODEL_ENGINE` and related environment variables.
The following providers are supported.

### OpenAI

```bash
export MAIN_MODEL_ENGINE="openai"
export MAIN_MODEL_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="your-openai-api-key"
```

### NVIDIA NIM

```bash
export MAIN_MODEL_ENGINE="nim"
export MAIN_MODEL_BASE_URL="http://localhost:8080/v1"
```

### vLLM / TRT-LLM

```bash
export MAIN_MODEL_ENGINE="vllm"  # or "trt_llm"
export MAIN_MODEL_BASE_URL="http://localhost:8080/v1"
```

### Anthropic

```bash
export MAIN_MODEL_ENGINE="anthropic"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Azure OpenAI

The engine name `azure_openai` is also accepted as an alias for `azure`.

```bash
export MAIN_MODEL_ENGINE="azure"
export AZURE_OPENAI_API_KEY="your-azure-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_VERSION="2024-06-01"  # optional, defaults to 2024-06-01
```

### Cohere

```bash
export MAIN_MODEL_ENGINE="cohere"
export COHERE_API_KEY="your-cohere-api-key"
# Optional: override the Cohere API base URL (defaults to https://api.cohere.com)
# export COHERE_BASE_URL="https://custom-cohere-endpoint.example.com"
```

### Custom OpenAI-Compatible Endpoint

For any provider that exposes an OpenAI-compatible `/v1/models` endpoint, set `MAIN_MODEL_BASE_URL`:

```bash
export MAIN_MODEL_ENGINE="my-custom-engine"
export MAIN_MODEL_BASE_URL="http://my-provider:8080/v1"
```

The server falls back to querying `MAIN_MODEL_BASE_URL` when the engine is not in the built-in provider table.

## Authentication

The endpoint forwards the `Authorization` header from the incoming request to the upstream provider.
If no `Authorization` header is present, the server uses the API key from the appropriate environment variable for the configured engine.

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 502 | The upstream provider is unreachable or returned an error. |
| 4xx | Proxied from the upstream provider (e.g., 401 for invalid API key). |

```{note}
If the engine is not in the built-in provider table and `MAIN_MODEL_BASE_URL` is not set, the endpoint returns an empty model list instead of an error.
```

## Related Topics

- [](run-guardrails-server.md)
- [](chat-with-guardrailed-model.md)
- [](../../reference/api-server-endpoints/index.md)

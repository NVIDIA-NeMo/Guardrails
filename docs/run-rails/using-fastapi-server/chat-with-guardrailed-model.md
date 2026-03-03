---
title:
  page: "Chat with Guardrailed Model"
  nav: "Chat Completions"
description: "Send chat requests, use streaming, and manage conversation threads."
keywords: ["chat completions", "guardrails API", "streaming responses", "conversation threads", "config_id", "OpenAI compatible"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "ai_platforms"]
content:
  type: tutorial
  difficulty: technical_intermediate
  audience: ["data_scientist", "engineer"]
---

# Chat with Guardrailed Model

Use the `/v1/chat/completions` endpoint to send messages and receive guarded responses from the server.
The endpoint is compatible with the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create),
with additional guardrails-specific fields nested under a `guardrails` object.

## Basic Request

Send a POST request to the chat completions endpoint.
The `model` field is required and specifies which LLM to use.
Guardrails-specific fields such as `config_id` are nested under the `guardrails` object.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [
      {"role": "user", "content": "Hello! What can you do for me?"}
    ],
    "guardrails": {
      "config_id": "content_safety"
    }
  }'
```

### Response

The response follows the standard OpenAI `ChatCompletion` format, with an additional `guardrails` object containing guardrails-specific output data.

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "meta/llama-3.1-8b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I can help you with your questions. What would you like to know?"
      },
      "finish_reason": "stop"
    }
  ],
  "guardrails": {
    "config_id": "content_safety",
    "state": null,
    "llm_output": null,
    "output_data": null,
    "log": null
  }
}
```

The `guardrails` response object may include additional fields depending on your request options:

- **`state`** — State object for continuing the conversation. Return this in subsequent requests to resume.
- **`llm_output`** — Additional LLM output data (when `guardrails.options.llm_output` is `true`).
- **`output_data`** — Values for requested context variables (when `guardrails.options.output_vars` is set).
- **`log`** — Logging information (when `guardrails.options.log` is configured).

## Using the OpenAI Python SDK

Since the server is OpenAI-compatible, you can use the [OpenAI Python SDK](https://github.com/openai/openai-python) to interact with it.
Pass guardrails-specific fields using the `extra_body` parameter.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-used"  # Required by OpenAI SDK but not used by the guardrails server
)

response = client.chat.completions.create(
    model="meta/llama-3.1-8b-instruct",
    messages=[
        {"role": "user", "content": "Hello! What can you do for me?"}
    ],
    extra_body={
        "guardrails": {
            "config_id": "content_safety"
        }
    }
)

print(response.choices[0].message.content)
```

## Using Python Requests

```python
import requests

base_url = "http://localhost:8000"

response = requests.post(f"{base_url}/v1/chat/completions", json={
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [
        {"role": "user", "content": "Hello! What can you do for me?"}
    ],
    "guardrails": {
        "config_id": "content_safety"
    }
})

print(response.json())
```

## Combine Multiple Configurations

You can combine multiple guardrails configurations in a single request using `config_ids` inside the `guardrails` object.
Use either `config_id` or `config_ids`, but not both — they are mutually exclusive.

```python
response = requests.post(f"{base_url}/v1/chat/completions", json={
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [
        {"role": "user", "content": "Hello!"}
    ],
    "guardrails": {
        "config_ids": ["main", "input_checking", "output_checking"]
    }
})
```

The configurations combine in the order specified.
If there are conflicts, the last configuration takes precedence.

```{note}
All configurations must use the same model type and engine.
```

### Example: Atomic Configurations

Create reusable *atomic configurations* that you can combine as needed:

1. `input_checking`: Uses the self-check input rail
2. `output_checking`: Uses the self-check output rail
3. `main`: Uses the base LLM with no guardrails

**Without input checking:**

```python
response = requests.post(f"{base_url}/v1/chat/completions", json={
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "You are stupid."}],
    "guardrails": {
        "config_id": "main"
    }
})
print(response.json()["choices"][0]["message"]["content"])
# LLM responds to the message
```

**With input checking:**

```python
response = requests.post(f"{base_url}/v1/chat/completions", json={
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "You are stupid."}],
    "guardrails": {
        "config_ids": ["main", "input_checking"]
    }
})
print(response.json()["choices"][0]["message"]["content"])
# "I'm sorry, I can't respond to that."
```

The input rail blocks the inappropriate message before it reaches the LLM.

## Use the Default Configuration

If the server was started with `--default-config-id`, you can omit the `guardrails` object:

```python
response = requests.post(f"{base_url}/v1/chat/completions", json={
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [
        {"role": "user", "content": "Hello!"}
    ]
})
```

## Streaming Responses

Enable streaming to receive partial responses as server-sent events (SSE).
Each chunk follows the OpenAI streaming format.

### Using curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true,
    "guardrails": {
      "config_id": "content_safety"
    }
  }'
```

The server sends chunks in SSE format:

```text
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"meta/llama-3.1-8b-instruct","choices":[{"delta":{"content":"Once"},"index":0,"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"meta/llama-3.1-8b-instruct","choices":[{"delta":{"content":" upon"},"index":0,"finish_reason":null}]}

data: [DONE]
```

### Using the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-used"
)

stream = client.chat.completions.create(
    model="meta/llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
    extra_body={
        "guardrails": {
            "config_id": "content_safety"
        }
    }
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### Using Python Requests

```python
import requests

response = requests.post(
    f"{base_url}/v1/chat/completions",
    json={
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": True,
        "guardrails": {
            "config_id": "content_safety"
        }
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode())
```

## Conversation Threads

Use `thread_id` inside the `guardrails` object to maintain conversation history on the server.
This is useful when you can only send the latest message rather than the full history.

```{tip}
The `thread_id` must be between 16 and 255 characters long.
```

```python
# First message
response = requests.post(f"{base_url}/v1/chat/completions", json={
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "My name is Alice."}],
    "guardrails": {
        "config_id": "content_safety",
        "thread_id": "user-session-12345678"
    }
})

# Follow-up message (server remembers the conversation)
response = requests.post(f"{base_url}/v1/chat/completions", json={
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "What is my name?"}],
    "guardrails": {
        "config_id": "content_safety",
        "thread_id": "user-session-12345678"
    }
})
# The assistant remembers "Alice"
```

:::{note}
The `thread_id` is currently not implemented in the NeMo Guardrails microservices.
:::

### Configure Thread Storage

To use threads, register a datastore in the server's `config.py`:

```python
# config.py in the root of your configurations folder
from nemoguardrails.server.api import register_datastore
from nemoguardrails.server.datastore.memory_store import MemoryStore

# For testing
register_datastore(MemoryStore())

# For production, use Redis:
# from nemoguardrails.server.datastore.redis_store import RedisStore
# register_datastore(RedisStore(redis_url="redis://localhost:6379"))
```

```{caution}
To use `RedisStore`, install `aioredis >= 2.0.1`.
```

### Thread Limitations

- Threads are not supported in streaming mode.
- Threads are stored indefinitely with no automatic cleanup.

## Add Context

Include additional context data in your request using the `context` field inside the `guardrails` object:

```python
response = requests.post(f"{base_url}/v1/chat/completions", json={
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "What is my account balance?"}],
    "guardrails": {
        "config_id": "content_safety",
        "context": {
            "user_id": "12345",
            "account_type": "premium"
        }
    }
})
```

## Control Generation Options

Use the `options` field inside the `guardrails` object to control which rails are applied and what information is returned:

```python
response = requests.post(f"{base_url}/v1/chat/completions", json={
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "guardrails": {
        "config_id": "content_safety",
        "options": {
            "rails": {
                "input": True,
                "output": True,
                "dialog": False
            },
            "log": {
                "activated_rails": True
            }
        }
    }
})
```

### Standard OpenAI Parameters

You can also pass standard OpenAI parameters such as `temperature`, `max_tokens`, `top_p`, `stop`, `presence_penalty`, and `frequency_penalty` at the top level:

```python
response = requests.post(f"{base_url}/v1/chat/completions", json={
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7,
    "max_tokens": 256,
    "guardrails": {
        "config_id": "content_safety"
    }
})
```

For complete details on generation options, see [](../../reference/api-server-endpoints/index.md).

## Related Topics

- [](run-guardrails-server.md)
- [](list-guardrail-configs.md)
- [](../../reference/api-server-endpoints/index.md)

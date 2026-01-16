---
title:
  page: "Chat with Guardrailed Model"
  nav: "Chat Completions"
description: "Send chat requests, use streaming, and manage conversation threads."
keywords: ["chat completions", "guardrails API", "streaming responses", "conversation threads", "config_id"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "ai_platforms"]
content:
  type: tutorial
  difficulty: technical_intermediate
  audience: ["data_scientist", "engineer"]
---

# Chat with Guardrailed Model

Use the `/v1/chat/completions` endpoint to send messages and receive guarded responses from the server.

:::{note}
While the endpoint is in the same format as the OpenAI's chat completions API endpoint, it is currently not compatible with the OpenAI API.
:::

## Basic Request

Send a POST request to the chat completions endpoint:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "content_safety",
    "messages": [
      {"role": "user", "content": "Hello! What can you do for me?"}
    ]
  }'
```

### Response

```json
{
  "messages": [
    {
      "role": "assistant",
      "content": "I can help you with your questions. What would you like to know?"
    }
  ]
}
```

## Using Python

```python
import requests

base_url = "http://localhost:8000"

response = requests.post(f"{base_url}/v1/chat/completions", json={
    "config_id": "content_safety",
    "messages": [
        {"role": "user", "content": "Hello! What can you do for me?"}
    ]
})

print(response.json())
```

## Combine Multiple Configurations

You can combine multiple guardrails configurations in a single request using `config_ids`:

```python
response = requests.post(f"{base_url}/v1/chat/completions", json={
    "config_ids": ["main", "input_checking", "output_checking"],
    "messages": [
        {"role": "user", "content": "Hello!"}
    ]
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
    "config_id": "main",
    "messages": [{"role": "user", "content": "You are stupid."}]
})
print(response.json())
# LLM responds to the message
```

**With input checking:**

```python
response = requests.post(f"{base_url}/v1/chat/completions", json={
    "config_ids": ["main", "input_checking"],
    "messages": [{"role": "user", "content": "You are stupid."}]
})
print(response.json())
# {"messages": [{"role": "assistant", "content": "I'm sorry, I can't respond to that."}]}
```

The input rail blocks the inappropriate message before it reaches the LLM.

## Use the Default Configuration

If the server was started with `--default-config-id`, you can omit the configuration:

```python
response = requests.post(f"{base_url}/v1/chat/completions", json={
    "messages": [
        {"role": "user", "content": "Hello!"}
    ]
})
```

## Streaming Responses

Enable streaming to receive partial responses as they are generated:

```python
import requests

response = requests.post(
    f"{base_url}/v1/chat/completions",
    json={
        "config_id": "content_safety",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode())
```

## Conversation Threads

Use `thread_id` to maintain conversation history on the server.
This is useful when you can only send the latest message rather than the full history.

```{tip}
The `thread_id` must be at least 16 characters long for security reasons.
```

```python
# First message
response = requests.post(f"{base_url}/v1/chat/completions", json={
    "config_id": "content_safety",
    "thread_id": "user-session-12345678",
    "messages": [{"role": "user", "content": "My name is Alice."}]
})

# Follow-up message (server remembers the conversation)
response = requests.post(f"{base_url}/v1/chat/completions", json={
    "config_id": "content_safety",
    "thread_id": "user-session-12345678",
    "messages": [{"role": "user", "content": "What is my name?"}]
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

Include additional context data in your request:

```python
response = requests.post(f"{base_url}/v1/chat/completions", json={
    "config_id": "content_safety",
    "messages": [{"role": "user", "content": "What is my account balance?"}],
    "context": {
        "user_id": "12345",
        "account_type": "premium"
    }
})
```

## Control Generation Options

Use the `options` field to control which rails are applied and what information is returned:

```python
response = requests.post(f"{base_url}/v1/chat/completions", json={
    "config_id": "content_safety",
    "messages": [{"role": "user", "content": "Hello"}],
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
})
```

For complete details on generation options, see [](../../reference/api-server-endpoints/index.md).

## Related Topics

- [](run-guardrails-server.md)
- [](list-guardrail-configs.md)
- [](../../reference/api-server-endpoints/index.md)

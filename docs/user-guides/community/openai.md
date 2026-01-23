## OpenAI API Compatibility for NeMo Guardrails

The NVIDIA NeMo Guardrails library provides server-side compatibility with OpenAI API endpoints, enabling applications that use OpenAI clients to seamlessly integrate for adding guardrails to LLM interactions. Point your OpenAI client to `http://localhost:8000` (or your server URL) and use the standard `/v1/chat/completions` endpoint.

## Feature Support Matrix

The following table outlines which OpenAI API features are currently supported when using the NVIDIA NeMo Guardrails library:

| Feature | Status | Notes |
| :------ | :----: | :---- |
| **Basic Chat Completion** | ✔ Supported | Full support for standard chat completions with guardrails applied |
| **Streaming Responses** | ✔ Supported | Server-Sent Events (SSE) streaming with `stream=true` |
| **List Models** | ✖ Unsupported | Use `/v1/rails/configs` to list available guardrails configurations |
| **Multimodal Input** | ✖ Unsupported | Support for text and image inputs (vision models) with guardrails but not yet OpenAI compatible  |
| **Function Calling** | ✖ Unsupported | Not yet implemented; guardrails need structured output support |
| **Tools** | ✖ Unsupported | Related to function calling; requires action flow integration |
| **Response Format (JSON Mode)** | ✖ Unsupported | Structured output with guardrails requires additional validation logic |

## Basic Chat Completion

The request requires two key fields:
* `model`: The LLM model to use (e.g., "gpt-4o", "llama-3.1-8b")
* `guardrails.config_id`: The guardrails configuration to apply

```
$ curl -X POST http://0.0.0.0:8000/v1/chat/completions \
   -H 'Accept: application/json' \
   -H 'Content-Type: application/json' \
   -d '{
      "model": "gpt-4o",
      "messages": [
         {
            "role": "user",
            "content": "What can you do for me?"
         }
      ],
      "guardrails": {
         "config_id": "nemoguards"
      },
      "max_tokens": 256,
      "temperature": 1,
      "top_p": 1
   }'
```

## Streaming Chat Completion

```
$ curl -X POST http://0.0.0.0:8000/v1/chat/completions \
   -H 'Accept: application/json' \
   -H 'Content-Type: application/json' \
   -d '{
      "model": "gpt-4o",
      "messages": [
         {
            "role": "user",
            "content": "What can you do for me?"
         }
      ],
      "guardrails": {
         "config_id": "nemoguards"
      },
      "max_tokens": 256,
      "stream": true,
      "temperature": 1,
      "top_p": 1
   }'
```

## Using with the OpenAI Python Client

```python
from openai import OpenAI

# Point to your NeMo Guardrails server
client = OpenAI(
    api_key=None,
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    extra_body={
        "guardrails": {
            "config_id": "nemoguards"
        }
    }
)

print(response.choices[0].message.content)
```

## Guardrails Options

The `guardrails` field supports additional options:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={
        "guardrails": {
            "config_id": "nemoguards",
            "context": {"user_id": "123"},
            "options": {
                "rails": {"input": True, "output": True},
                "log": {"activated_rails": True, "llm_calls": True}
            }
        }
    }
)
```

| Field | Description |
| :---- | :---------- |
| `config_id` | The guardrails configuration ID to use |
| `config_ids` | List of configuration IDs to combine (alternative to `config_id`) |
| `context` | Additional context data for the conversation |
| `options` | Generation options (rails settings, logging, etc.) |
| `state` | State object to continue a stateful conversation |
| `thread_id` | Thread ID for server-managed conversation history |

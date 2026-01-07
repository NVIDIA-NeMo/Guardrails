## OpenAI API Compatibility for NeMo Guardrails

NeMo Guardrails provides server-side compatibility with OpenAI API endpoints, enabling applications that use OpenAI clients to seamlessly integrate with NeMo Guardrails for adding guardrails to LLM interactions. Point your OpenAI client to `http://localhost:8000` (or your server URL) and use the standard `/v1/chat/completions` and `/v1/models` endpoint.

## Feature Support Matrix

The following table outlines which OpenAI API features are currently supported when using NeMo Guardrails:

| Feature | Status | Notes |
| :------ | :----: | :---- |
| **Basic Chat Completion** | ✔ Supported | Full support for standard chat completions with guardrails applied |
| **Streaming Responses** | ✔ Supported | Server-Sent Events (SSE) streaming with `stream=true` |
| **List Models** | ✔ Supported | Returns available guardrails configurations as models |
| **Multimodal Input** | ✖ Unsupported | Support for text and image inputs (vision models) with guardrails but not yet OpenAI compatible  |
| **Function Calling** | ✖ Unsupported | Not yet implemented; guardrails need structured output support |
| **Tools** | ✖ Unsupported | Related to function calling; requires action flow integration |
| **Response Format (JSON Mode)** | ✖ Unsupported | Structured output with guardrails requires additional validation logic |

## Example Usage

Export the main model's base URL, engine, and API key as environment variables:

```
export MAIN_MODEL_BASE_URL="http://model-server/v1"
export MAIN_MODEL_ENGINE="openai"  # or "nim", "vllm", etc.
export MAIN_MODEL_API_KEY="model-server-api-key"  # or leave empty if not needed
```

**NOTE**: By default these values are:

* `MAIN_MODEL_BASE_URL`: `https://localhost:8000/v1`
* `MAIN_MODEL_ENGINE`: `nim`
* `MAIN_MODEL_API_KEY`: `None`

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

## List Available Models

```
$ curl -X GET http://0.0.0.0:8000/v1/models \
   -H 'Accept: application/json'
```

*Example output*:

```
{
   "object": "list",
   "data": [
      {
         "id": "gpt-3.5-turbo-instruct",
         "object": "model",
         "created": 1234567890,
         "owned_by": "nemo-guardrails",
         "config_id": "nemoguards"
      }
   ]
}
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

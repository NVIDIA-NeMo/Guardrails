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

```
$ curl -X POST http://0.0.0.0:8000/v1/chat/completions \
   -H 'Accept: application/json' \
   -H 'Content-Type: application/json' \
   -d '{
      "model": "nemoguards", // Maps to config_id="nemoguards"
      "messages": [
         {
            "role": "user",
            "content": "What can you do for me?"
         }
      ],
      "max_tokens": 256,
      "temperature": 1,
      "top_p": 1
   }'
```

**NOTE**: You can also explicitly specify `config_id` if needed:

```
{
   "config_id": "my-config",
   "messages": [...]
}
```

## Streaming Chat Completion

```
$ curl -X POST http://0.0.0.0:8000/v1/chat/completions \
   -H 'Accept: application/json' \
   -H 'Content-Type: application/json' \
   -d '{
      "model": "nemoguards",
      "messages": [
         {
            "role": "user",
            "content": "What can you do for me?"
         }
      ],
      "max_tokens": 256,
      "stream": true,
      "temperature": 1,
      "top_p": 1,
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
         "config_id": "abc"
      }
   ]
}
```

## Using with the OpenAI Python Client

```
from openai import OpenAI

# Point to your NeMo Guardrails server
client = OpenAI(
    api_key=None,
    base_url="http://localhost:8000/v1"
)

# Use the model field to specify your guardrails config
response = client.chat.completions.create(
    model="nemoguards",  # Your config ID
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

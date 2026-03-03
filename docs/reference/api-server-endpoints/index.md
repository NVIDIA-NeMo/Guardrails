# NeMo Guardrails Library API Server Endpoints Reference

This reference documents the REST API endpoints provided by the NeMo Guardrails library API server.
The server exposes an OpenAI-compatible Chat Completions API with additional guardrails-specific extensions.

## Starting the Server

Start the server using the CLI:

```bash
nemoguardrails server --port 8000 --config /path/to/config
```

For more information about server options, see [](../../run-rails/using-fastapi-server/run-guardrails-server.md).

---

## Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/completions` | Generate a guarded chat completion |
| `GET` | `/v1/models` | List available models from the configured provider |
| `GET` | `/v1/rails/configs` | List available guardrails configurations |
| `GET` | `/v1/challenges` | Get red teaming challenges |
| `GET` | `/` | Chat UI (if enabled) or health status |

---

## POST /v1/chat/completions

Generate a chat completion with guardrails applied.
The request and response formats are compatible with the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create),
with guardrails-specific fields nested under a `guardrails` object.

### Request Body

```json
{
  "model": "meta/llama-3.1-8b-instruct",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 256,
  "guardrails": {
    "config_id": "my-config"
  }
}
```

#### OpenAI Fields

```{list-table}
:header-rows: 1
:widths: 20 15 10 55

* - Field
  - Type
  - Required
  - Description

* - `model`
  - string
  - **Yes**
  - The LLM model to use for chat completion (e.g., `"meta/llama-3.1-8b-instruct"`, `"gpt-4o"`).

* - `messages`
  - array of objects
  - No
  - The list of messages in the current conversation. Each message has `role` and `content` fields. Although the OpenAI API requires this field, the Guardrails server treats it as optional to support stateful continuation via `guardrails.state`. When omitted, defaults to an empty list.

* - `stream`
  - boolean
  - No
  - If `true`, returns partial message deltas as server-sent events. Default: `false`.

* - `max_tokens`
  - integer
  - No
  - The maximum number of tokens to generate.

* - `temperature`
  - float
  - No
  - Sampling temperature (0-2). Higher values make output more random.

* - `top_p`
  - float
  - No
  - Top-p (nucleus) sampling parameter.

* - `stop`
  - string or array
  - No
  - Stop sequence(s) where the model stops generating.

* - `presence_penalty`
  - float
  - No
  - Presence penalty parameter (-2.0 to 2.0).

* - `frequency_penalty`
  - float
  - No
  - Frequency penalty parameter (-2.0 to 2.0).
```

#### Guardrails Fields

Guardrails-specific fields are nested under the `guardrails` object in the request body.

```{list-table}
:header-rows: 1
:widths: 20 15 10 55

* - Field
  - Type
  - Required
  - Description

* - `guardrails.config_id`
  - string
  - No
  - The ID of the guardrails configuration to use. If not set, uses the server's default configuration. Mutually exclusive with `config_ids`.

* - `guardrails.config_ids`
  - array of strings
  - No
  - List of configuration IDs to combine. Mutually exclusive with `config_id`.

* - `guardrails.thread_id`
  - string
  - No
  - ID of an existing thread for conversation persistence. Must be 16-255 characters.

* - `guardrails.context`
  - object
  - No
  - Additional context data to add to the conversation.

* - `guardrails.options`
  - object
  - No
  - Additional options for controlling the generation. See [Generation Options](#generation-options).

* - `guardrails.state`
  - object
  - No
  - A state object to continue a previous interaction. Must contain an `events` or `state` key, or be an empty dict `{}` to start a new conversation.
```

### Generation Options

The `guardrails.options` field controls which rails are applied and what information is returned.

```json
{
  "guardrails": {
    "config_id": "my-config",
    "options": {
      "rails": {
        "input": true,
        "output": true,
        "dialog": true,
        "retrieval": true
      },
      "llm_params": {
        "temperature": 0.7
      },
      "llm_output": false,
      "output_vars": ["relevant_chunks"],
      "log": {
        "activated_rails": true,
        "llm_calls": false
      }
    }
  }
}
```

#### Rails Options

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Field
  - Type
  - Description

* - `input`
  - boolean | array
  - Enable input rails. Set to `false` to disable, or provide a list of specific rail names.

* - `output`
  - boolean | array
  - Enable output rails. Set to `false` to disable, or provide a list of specific rail names.

* - `dialog`
  - boolean
  - Enable dialog rails. Default: `true`.

* - `retrieval`
  - boolean | array
  - Enable retrieval rails. Set to `false` to disable, or provide a list of specific rail names.

* - `tool_input`
  - boolean | array
  - Enable tool input rails. Default: `true`.

* - `tool_output`
  - boolean | array
  - Enable tool output rails. Default: `true`.
```

#### Other Options

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Field
  - Type
  - Description

* - `llm_params`
  - object
  - Additional parameters to pass to the LLM call (e.g., `temperature`, `max_tokens`).

* - `llm_output`
  - boolean
  - Whether to include custom LLM output in the response. Default: `false`.

* - `output_vars`
  - boolean | array
  - Context variables to return. Set to `true` for all, or provide a list of variable names.
```

#### Log Options

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Field
  - Type
  - Description

* - `activated_rails`
  - boolean
  - Include information about which rails were activated. Default: `false`.

* - `llm_calls`
  - boolean
  - Include details about all LLM calls (prompts, completions, token usage). Default: `false`.

* - `internal_events`
  - boolean
  - Include the array of internal generated events. Default: `false`.

* - `colang_history`
  - boolean
  - Include conversation history in Colang format. Default: `false`.
```

### Response Body

The response follows the standard OpenAI `ChatCompletion` format with an additional `guardrails` object.

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1709424000,
  "model": "meta/llama-3.1-8b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm doing well, thank you!"
      },
      "finish_reason": "stop"
    }
  ],
  "guardrails": {
    "config_id": "content_safety",
    "llm_output": null,
    "output_data": null,
    "log": null,
    "state": null
  }
}
```

#### Response Fields

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Field
  - Type
  - Description

* - `id`
  - string
  - A unique identifier for the chat completion (e.g., `"chatcmpl-abc123"`).

* - `object`
  - string
  - Always `"chat.completion"`.

* - `created`
  - integer
  - Unix timestamp of when the completion was created.

* - `model`
  - string
  - The model used for the completion.

* - `choices`
  - array
  - Array of completion choices. Each choice contains `index`, `message` (with `role` and `content`), and `finish_reason`.

* - `guardrails`
  - object
  - Guardrails-specific output data. See below.
```

#### Guardrails Response Fields

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Field
  - Type
  - Description

* - `guardrails.config_id`
  - string
  - The guardrails configuration ID associated with this response.

* - `guardrails.state`
  - object
  - State object for continuing the conversation in future requests.

* - `guardrails.llm_output`
  - object
  - Additional LLM output data. Only included if `guardrails.options.llm_output` was `true`.

* - `guardrails.output_data`
  - object
  - Values for requested output variables. Only included if `guardrails.options.output_vars` was set.

* - `guardrails.log`
  - object
  - Logging information based on `guardrails.options.log` settings.
```

### Examples

#### Basic Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "guardrails": {
      "config_id": "content_safety"
    }
  }'
```

#### Request with Streaming

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true,
    "guardrails": {
      "config_id": "content_safety"
    }
  }'
```

Streaming responses use Server-Sent Events (SSE). Each chunk is a `chat.completion.chunk` object:

```text
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"meta/llama-3.1-8b-instruct","choices":[{"delta":{"content":"Once"},"index":0,"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"meta/llama-3.1-8b-instruct","choices":[{"delta":{"content":" upon"},"index":0,"finish_reason":null}]}

data: [DONE]
```

#### Request with Specific Rails

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "guardrails": {
      "config_id": "content_safety",
      "options": {
        "rails": {
          "input": ["check jailbreak"],
          "output": false,
          "dialog": false
        }
      }
    }
  }'
```

#### Request with Logging

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "guardrails": {
      "config_id": "content_safety",
      "options": {
        "log": {
          "activated_rails": true,
          "llm_calls": true
        }
      }
    }
  }'
```

#### Request with OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-used"
)

response = client.chat.completions.create(
    model="meta/llama-3.1-8b-instruct",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    extra_body={
        "guardrails": {
            "config_id": "content_safety"
        }
    }
)

print(response.choices[0].message.content)
```

---

## GET /v1/models

List the available LLM models from the configured upstream provider.
This endpoint proxies the request to the provider specified by `MAIN_MODEL_ENGINE` and returns the results in the standard OpenAI models list format.

For a guide on configuring providers, see [](../../run-rails/using-fastapi-server/list-models.md).

### Request

No request body or query parameters. The `Authorization` header, if present, is forwarded to the upstream provider.

```bash
curl http://localhost:8000/v1/models
```

### Response Body

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

#### Response Fields

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Field
  - Type
  - Description

* - `data`
  - array
  - List of model objects.

* - `data[].id`
  - string
  - The model identifier (e.g., `"gpt-4o"`, `"meta/llama-3.1-8b-instruct"`).

* - `data[].object`
  - string
  - Always `"model"`.

* - `data[].created`
  - integer
  - Unix timestamp of the model's creation.

* - `data[].owned_by`
  - string
  - The organization that owns the model.
```

#### Error Responses

```{list-table}
:header-rows: 1
:widths: 15 85

* - Status
  - Description

* - 502
  - The upstream provider is unreachable or returned an error.

* - 4xx
  - Proxied from the upstream provider (e.g., 401 for an invalid API key).
```

```{note}
If the engine is not in the built-in provider table and `MAIN_MODEL_BASE_URL` is not set, the endpoint returns an empty model list instead of an error.
```

### Example with OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")
for model in client.models.list().data:
    print(model.id)
```

---

## GET /v1/rails/configs

List all available guardrails configurations.

Returns an array of configuration objects.

```json
[
  {"id": "content_safety"},
  {"id": "customer-service"},
  {"id": "content-moderation"}
]
```

```bash
curl http://localhost:8000/v1/rails/configs
```

---

## GET /v1/challenges

Get the list of available red teaming challenges.

Returns an array of challenge objects. The structure depends on the registered challenges.

```json
[
  {
    "id": "jailbreak-1",
    "description": "Attempt to bypass safety guardrails",
    "category": "jailbreak"
  }
]
```

```bash
curl http://localhost:8000/v1/challenges
```

```{note}
Challenges must be registered via a `challenges.json` file in the configuration directory or programmatically using `register_challenges()`.
```

---

## GET /

Root endpoint that serves the Chat UI or returns a health status.

**Chat UI Disabled**: When the Chat UI is disabled (`--disable-chat-ui`), returns a health status:

```json
{"status": "ok"}
```

**Chat UI Enabled**: When the Chat UI is enabled (default), serves the interactive chat interface.

---

## Error Responses

Errors from the chat completions endpoint are returned as `ChatCompletion` objects with the error message in the assistant's content, or as HTTP exceptions.

### Configuration Error

When the guardrails configuration cannot be loaded:

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
        "content": "Could not load the ['my-config'] guardrails configuration. An internal error has occurred."
      },
      "finish_reason": "stop"
    }
  ]
}
```

### Missing Configuration

When no `config_id` is provided and no default is set, the server returns an HTTP 422 error:

```json
{
  "detail": "No guardrails config_id provided and server has no default configuration"
}
```

### Thread ID Validation Error

When `thread_id` is less than 16 characters:

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
        "content": "The `thread_id` must have a minimum length of 16 characters."
      },
      "finish_reason": "stop"
    }
  ],
  "guardrails": {
    "config_id": "my-config"
  }
}
```

### Invalid State Format

When the `guardrails.state` object does not contain an `events` or `state` key, the server returns an HTTP 422 error:

```json
{
  "detail": "Invalid state format: state must contain 'events' or 'state' key. Use an empty dict {} to start a new conversation."
}
```

### Internal Server Error

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
        "content": "Internal server error"
      },
      "finish_reason": "stop"
    }
  ],
  "guardrails": {
    "config_id": "my-config"
  }
}
```

### Streaming Errors

During streaming, errors are sent as SSE events with an `error` object:

```text
data: {"error": {"message": "...", "type": "...", "param": "...", "code": "..."}}
```

---

## Environment Variables

The server supports the following environment variables:

```{list-table}
:header-rows: 1
:widths: 35 65

* - Variable
  - Description

* - `DEFAULT_CONFIG_ID`
  - Default guardrails configuration ID when none is specified in the request.

* - `MAIN_MODEL_ENGINE`
  - The LLM engine to use when the `model` field is specified in the request (e.g., `"openai"`, `"nim"`, `"vllm"`, `"anthropic"`, `"azure"` or `"azure_openai"`, `"cohere"`). Default: `"openai"`.

* - `MAIN_MODEL_BASE_URL`
  - Base URL for the LLM provider when the `model` field is specified in the request. Useful for self-hosted models (e.g., `"http://localhost:8080/v1"`).

* - `OPENAI_API_KEY`
  - API key for OpenAI models.

* - `NVIDIA_API_KEY`
  - API key for NVIDIA-hosted models on build.nvidia.com.

* - `ANTHROPIC_API_KEY`
  - API key for Anthropic models. Used when `MAIN_MODEL_ENGINE` is `"anthropic"`.

* - `AZURE_OPENAI_API_KEY`
  - API key for Azure OpenAI. Used when `MAIN_MODEL_ENGINE` is `"azure"` or `"azure_openai"`.

* - `AZURE_OPENAI_ENDPOINT`
  - Azure OpenAI resource endpoint URL (e.g., `"https://your-resource.openai.azure.com"`). Required when `MAIN_MODEL_ENGINE` is `"azure"`.

* - `AZURE_OPENAI_API_VERSION`
  - Azure OpenAI API version. Default: `"2024-06-01"`.

* - `COHERE_API_KEY`
  - API key for Cohere models. Used when `MAIN_MODEL_ENGINE` is `"cohere"`.

* - `COHERE_BASE_URL`
  - Override the Cohere API base URL. Default: `"https://api.cohere.com"`.

* - `NEMO_GUARDRAILS_SERVER_ENABLE_CORS`
  - Set to `"true"` to enable CORS. Default: `"false"`.

* - `NEMO_GUARDRAILS_SERVER_ALLOWED_ORIGINS`
  - Comma-separated list of allowed CORS origins. Default: `"*"`.
```

---

## Related Topics

- [Run the Guardrails Server](../../run-rails/using-fastapi-server/run-guardrails-server.md)
- [Deployment Guide](../../deployment/index.md)

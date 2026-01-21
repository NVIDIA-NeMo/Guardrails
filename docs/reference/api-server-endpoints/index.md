# NeMo Guardrails Library API Server Endpoints Reference

This reference documents the REST API endpoints provided by the NeMo Guardrails library API server.

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
| `GET` | `/v1/rails/configs` | List available guardrails configurations |
| `GET` | `/v1/challenges` | Get red teaming challenges |
| `GET` | `/` | Chat UI (if enabled) or health status |

---

## POST /v1/chat/completions

Generate a chat completion with guardrails applied.

### Request Body

```json
{
  "config_id": "my-config",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "stream": false,
  "options": {}
}
```

#### Request Fields

```{list-table}
:header-rows: 1
:widths: 20 15 10 55

* - Field
  - Type
  - Required
  - Description

* - `config_id`
  - string
  - No
  - The ID of the configuration to use. If not set, uses the server's default configuration.

* - `config_ids`
  - array of strings
  - No
  - List of configuration IDs to combine. Mutually exclusive with `config_id`.

* - `messages`
  - array of objects
  - No
  - The list of messages in the current conversation. Each message has `role` and `content` fields.

* - `thread_id`
  - string
  - No
  - ID of an existing thread for conversation persistence. Must be 16-255 characters.

* - `context`
  - object
  - No
  - Additional context data to add to the conversation.

* - `stream`
  - boolean
  - No
  - If `true`, returns partial message deltas as server-sent events. Default: `false`.

* - `options`
  - object
  - No
  - Additional options for controlling the generation. See [Generation Options](#generation-options).

* - `state`
  - object
  - No
  - A state object to continue a previous interaction.
```

### Generation Options

The `options` field controls which rails are applied and what information is returned.

```json
{
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

```json
{
  "messages": [
    {"role": "assistant", "content": "I'm doing well, thank you!"}
  ],
  "llm_output": null,
  "output_data": null,
  "log": null,
  "state": null
}
```

#### Response Fields

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Field
  - Type
  - Description

* - `messages`
  - array of objects
  - The new messages in the conversation (typically the assistant's response).

* - `llm_output`
  - object
  - Additional output from the LLM. Only included if `options.llm_output` is `true`.

* - `output_data`
  - object
  - Values for requested output variables. Only included if `options.output_vars` is set.

* - `log`
  - object
  - Logging information based on `options.log` settings.

* - `state`
  - object
  - State object for continuing the interaction in future requests.
```

### Examples

#### Basic Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "content_safety",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

#### Request with Streaming

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "content_safety",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

#### Request with Specific Rails

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "content_safety",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "options": {
      "rails": {
        "input": ["check jailbreak"],
        "output": false,
        "dialog": false
      }
    }
  }'
```

#### Request with Logging

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "content_safety",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "options": {
      "log": {
        "activated_rails": true,
        "llm_calls": true
      }
    }
  }'
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

### Configuration Error

When no configuration is provided and no default is set:

```json
{
  "messages": [
    {
      "role": "assistant",
      "content": "Could not load the [config_id] guardrails configuration. An internal error has occurred."
    }
  ]
}
```

### Thread ID Validation Error

When `thread_id` is less than 16 characters:

```json
{
  "messages": [
    {
      "role": "assistant",
      "content": "The `thread_id` must have a minimum length of 16 characters."
    }
  ]
}
```

### Internal Server Error

```json
{
  "messages": [
    {
      "role": "assistant",
      "content": "Internal server error."
    }
  ]
}
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
  - Default configuration ID when none is specified in the request.

* - `NEMO_GUARDRAILS_SERVER_ENABLE_CORS`
  - Set to `"true"` to enable CORS. Default: `"false"`.

* - `NEMO_GUARDRAILS_SERVER_ALLOWED_ORIGINS`
  - Comma-separated list of allowed CORS origins. Default: `"*"`.
```

---

## Related Topics

- [Run the Guardrails Server](../../run-rails/using-fastapi-server/run-guardrails-server.md)
- [Deployment Guide](../../deployment/index.md)

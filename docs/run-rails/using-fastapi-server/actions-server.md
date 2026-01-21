---
title:
  page: "Use the Actions Server"
  nav: "Actions Server"
description: "Run guardrail actions in a secure, isolated environment."
keywords: ["actions server", "nemoguardrails actions-server", "custom actions", "action dispatcher", "security isolation"]
topics: ["generative_ai", "developer_tools", "cybersecurity"]
tags: ["llms", "ai_inference", "security_for_ai"]
content:
  type: tutorial
  difficulty: technical_intermediate
  audience: ["data_scientist", "engineer"]
---

# Use the Actions Server

The Actions Server enables you to run actions invoked from guardrails in a secure, isolated environment separate from the main guardrails server.

## Why Use an Actions Server?

Actions are custom Python functions that guardrails can invoke during processing.
Running actions in a separate server provides:

- **Security isolation**: Actions run in a separate process, limiting the impact of potentially unsafe code.
- **Resource management**: Separate resource allocation for action execution.
- **Scalability**: Scale the actions server independently from the guardrails server.

```{note}
Using an actions server is optional but highly recommended for production deployments.
If no actions server is configured, actions run in the same process as the guardrails server.
```

For more details on security considerations, see [](../../resources/security/guidelines.md).

## Start the Actions Server

Launch the actions server using the CLI:

```bash
nemoguardrails actions-server [--port PORT]
```

The default port is `8001`.

### Example

```bash
nemoguardrails actions-server --port 8001
```

On startup, the actions server automatically registers:

- All predefined actions from the NeMo Guardrails library.
- All actions defined in Python files in the current folder and sub-folders.

## Endpoints

The actions server exposes two endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/actions/list` | List all available actions |
| `POST` | `/v1/actions/run` | Execute an action |

View the OpenAPI documentation at:

- **Swagger UI**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`

---

## GET /v1/actions/list

List all available actions registered with the server.

### Request

```bash
curl http://localhost:8001/v1/actions/list
```

### Response

Returns an array of action names:

```json
[
  "apify",
  "bing_search",
  "google_search",
  "google_serper",
  "openweather_query",
  "searx_search",
  "serp_api_query",
  "wikipedia_query",
  "wolframalpha_query",
  "zapier_nla_query"
]
```

### Python Example

```python
import requests

actions_server = "http://localhost:8001"

response = requests.get(f"{actions_server}/v1/actions/list")
actions = response.json()

print("Available actions:")
for action in actions:
    print(f"  - {action}")
```

---

## POST /v1/actions/run

Execute an action with the specified parameters.

### Request Body

```json
{
  "action_name": "string",
  "action_parameters": {}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `action_name` | string | Name of the action to execute |
| `action_parameters` | object | Parameters to pass to the action |

### Response Body

```json
{
  "status": "success",
  "result": "string"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Execution status: `success` or `failed` |
| `result` | string | The result returned by the action |

### Example: Wolfram Alpha Query

```bash
curl -X POST http://localhost:8001/v1/actions/run \
  -H "Content-Type: application/json" \
  -d '{
    "action_name": "wolframalpha_query",
    "action_parameters": {
      "query": "What is the largest prime factor of 1024?"
    }
  }'
```

**Response:**

```json
{
  "status": "success",
  "result": "2"
}
```

### Python Example

```python
import requests

actions_server = "http://localhost:8001"

# Execute an action
response = requests.post(f"{actions_server}/v1/actions/run", json={
    "action_name": "wikipedia_query",
    "action_parameters": {
        "query": "NeMo Guardrails"
    }
})

result = response.json()
if result["status"] == "success":
    print(f"Result: {result['result']}")
else:
    print("Action failed")
```

---

## Configure Guardrails to Use the Actions Server

To configure a guardrails configuration to use a remote actions server, add the `actions_server_url` to your `config.yml`:

```yaml
actions_server_url: "http://localhost:8001"

models:
  - type: main
    engine: openai
    model: gpt-4

rails:
  input:
    flows:
      - self check input
```

When configured, the guardrails server sends action execution requests to the actions server instead of running them locally.

---

## Register Custom Actions

The actions server automatically discovers actions in the current directory.
Create a Python file with your custom actions:

```python
# my_actions.py
from nemoguardrails.actions import action

@action(name="get_weather")
async def get_weather(location: str):
    """Get the current weather for a location."""
    # Your implementation here
    return f"The weather in {location} is sunny."

@action(name="lookup_user")
async def lookup_user(user_id: str):
    """Look up user information."""
    # Your implementation here
    return {"name": "Alice", "id": user_id}
```

Start the actions server from the directory containing your actions:

```bash
cd /path/to/my-actions
nemoguardrails actions-server --port 8001
```

Your custom actions are now available:

```bash
curl http://localhost:8001/v1/actions/list
# ["get_weather", "lookup_user", "apify", "bing_search", ...]
```

---

## Example: Complete Workflow

This example demonstrates running both servers together.

1. Start the Actions Server.

    ```bash
    nemoguardrails actions-server --port 8001
    ```

2. Configure a guardrails configuration to use the actions server.

    ```yaml
    # config/my-bot/config.yml
    actions_server_url: "http://localhost:8001"

    models:
      - type: main
        engine: openai
        model: gpt-4

    rails:
      input:
        flows:
          - self check input
    ```

3. Start the guardrails server.

    ```bash
    nemoguardrails server --config ./config --port 8000
    ```

4. Test the servers.

    ```python
    import requests

    # Verify actions are available
    actions = requests.get("http://localhost:8001/v1/actions/list").json()
    print(f"Available actions: {len(actions)}")

    # Chat with the guardrailed model
    response = requests.post("http://localhost:8000/v1/chat/completions", json={
        "config_id": "my-bot",
        "messages": [{"role": "user", "content": "What's the weather in New York?"}]
    })
    print(response.json())
    ```

---

## Deployment Considerations

For production deployments:

1. **Network isolation**: Deploy the actions server in a separate network segment.
2. **Authentication**: Add authentication between the guardrails and actions servers.
3. **Resource limits**: Configure appropriate resource limits for action execution.
4. **Monitoring**: Monitor action execution times and failure rates.
5. **Logging**: Enable detailed logging for debugging and auditing.

## Related Topics

- [](run-guardrails-server.md)
- [](../../resources/security/guidelines.md)
- [](../../configure-rails/actions/index.md)

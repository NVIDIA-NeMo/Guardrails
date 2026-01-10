---
title:
  page: "List Guardrail Configurations"
  nav: "List Configurations"
description: "Retrieve available guardrails configurations from the server."
keywords: ["rails configs", "list configurations", "guardrails API", "config discovery"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "ai_platforms"]
content:
  type: reference
  difficulty: technical_intermediate
  audience: ["data_scientist", "engineer"]
---

# List Guardrail Configurations

Use the `/v1/rails/configs` endpoint to retrieve the list of available guardrails configurations from the server.

## Request

```bash
curl http://localhost:8000/v1/rails/configs
```

## Response

The endpoint returns an array of configuration objects, each with an `id` field:

```json
[
  {"id": "my-bot"},
  {"id": "customer-service"},
  {"id": "content-moderation"}
]
```

## Using Python

```python
import requests

base_url = "http://localhost:8000"

response = requests.get(f"{base_url}/v1/rails/configs")
configs = response.json()

print("Available configurations:")
for config in configs:
    print(f"  - {config['id']}")
```

**Example output:**

```text
Available configurations:
  - input_checking
  - output_checking
  - main
```

## Use a Configuration

After retrieving the available configurations, use a configuration ID in your chat requests:

```python
# Get available configs
response = requests.get(f"{base_url}/v1/rails/configs")
configs = response.json()

# Use the first available config
if configs:
    config_id = configs[0]["id"]

    response = requests.post(f"{base_url}/v1/chat/completions", json={
        "config_id": config_id,
        "messages": [{"role": "user", "content": "Hello!"}]
    })
    print(response.json())
```

## How Configurations Are Discovered

The server discovers configurations based on how it was started:

**Multi-config mode** (default): The server scans the configuration directory for sub-folders containing a `config.yml` or `config.yaml` file.
Each sub-folder becomes an available configuration with its folder name as the ID.

```text
configs/
├── my-bot/           → config_id: "my-bot"
│   └── config.yml
├── customer-service/ → config_id: "customer-service"
│   └── config.yml
└── moderation/       → config_id: "moderation"
    └── config.yml
```

**Single-config mode**: If the server is pointed to a folder containing a `config.yml` file directly (not in sub-folders), only that configuration is available.
The folder name becomes the configuration ID.

```bash
nemoguardrails server --config ./my-single-bot
```

The endpoint returns:

```json
[{"id": "my-single-bot"}]
```

## Related Topics

- [Run the Guardrails Server](run-guardrails-server.md)
- [Chat with Guardrailed Model](chat-with-guardrailed-model.md)
- [Server Endpoints Reference](../../reference/api-server-endpoints/index.md)

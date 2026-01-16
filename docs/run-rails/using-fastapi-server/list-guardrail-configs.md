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
  {"id": "content_safety"},
  {"id": "jailbreak_detection"},
  {"id": "topic_safety"}
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
  - content_safety
  - jailbreak_detection
  - topic_safety
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
examples/configs/
├── content_safety/        → config_id: "content_safety"
│   └── config.yml
├── jailbreak_detection/   → config_id: "jailbreak_detection"
│   └── config.yml
└── topic_safety/          → config_id: "topic_safety"
    └── config.yml
```

**Single-config mode**: If the server is pointed to a folder containing a `config.yml` file directly (not in sub-folders), only that configuration is available.
The folder name becomes the configuration ID.

```bash
nemoguardrails server --config examples/configs/content_safety
```

The endpoint returns:

```json
[{"id": "content_safety"}]
```

## Related Topics

- [](run-guardrails-server.md)
- [](chat-with-guardrailed-model.md)
- [](../../reference/api-server-endpoints/index.md)

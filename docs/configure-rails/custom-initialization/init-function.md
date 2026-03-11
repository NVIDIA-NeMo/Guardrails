---
title:
  page: The Init Function for NeMo Guardrails
  nav: Init Function
description: Define the init() function to initialize resources and register action parameters at startup.
topics:
- Configuration
- Customization
tags:
- init
- config.py
- Action Parameters
- Python
- Initialization
content:
  type: how_to
  difficulty: technical_intermediate
  audience:
  - engineer
  - AI Engineer
---

# The Init Function

If `config.py` contains an `init` function, it is called during `LLMRails` initialization. Use it to set up shared resources and register action parameters.

```{important}
The `init` function **must be synchronous** (`def init`, not `async def init`). The framework calls it without `await`, so an async function would silently do nothing.
```

Any top-level code in `config.py` runs at import time, before `init()` is called. This can be used for provider registration that does not require the `LLMRails` instance.

## Basic Usage

```python
from nemoguardrails import LLMRails

def init(app: LLMRails):
    # Initialize database connection
    db = DatabaseConnection()

    # Register as action parameter (available to all actions)
    app.register_action_param("db", db)
```

## Registering Action Parameters

Action parameters registered in `config.py` are automatically injected into actions that declare them. The runtime matches parameters by name, i.e., the parameter name in the action must match the name used during registration.

**config.py:**

```python
import os

from nemoguardrails import LLMRails

def init(app: LLMRails):
    # Initialize shared resources
    db = DatabaseConnection(host="localhost", port=5432)
    api_client = ExternalAPIClient(api_key=os.environ.get("API_KEY"))

    # Register as action parameters
    app.register_action_param("db", db)
    app.register_action_param("api_client", api_client)
```

**actions.py:**

```python
from nemoguardrails.actions import action

@action()
async def fetch_user_data(user_id: str, db=None):
    """The 'db' parameter is injected from config.py."""
    return await db.get_user(user_id)

@action()
async def call_external_service(query: str, api_client=None):
    """The 'api_client' parameter is injected from config.py."""
    return await api_client.search(query)
```

## Built-in Action Parameters

In addition to parameters you register, the runtime automatically injects these built-in parameters into any action that declares them:

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `RailsConfig` | The full rails configuration object |
| `context` | `dict` | The current conversation context |
| `events` | `list` | The event history |
| `llm` | LLM instance | The main LLM (auto-registered during initialization) |
| `llm_task_manager` | `LLMTaskManager` | Manages LLM task execution |

See [Custom Data](custom-data.md) for details on accessing `config.custom_data` inside actions.

## Accessing the Configuration

The `app` parameter provides access to the full configuration:

```python
def init(app: LLMRails):
    # Access the RailsConfig object
    config = app.config

    # Access custom data from config.yml
    custom_settings = config.custom_data

    # Access model configurations
    models = config.models
```

## Example: Database Connection

```python
import os
import psycopg2
from nemoguardrails import LLMRails

def init(app: LLMRails):
    # Create connection pool
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        database=os.environ.get("DB_NAME", "mydb"),
        user=os.environ.get("DB_USER", "user"),
        password=os.environ.get("DB_PASSWORD"),
    )

    app.register_action_param("db_conn", conn)
```

## Example: API Client Initialization

```python
import os
import httpx
from nemoguardrails import LLMRails

def init(app: LLMRails):
    # Get API key from custom_data in config.yml
    api_key = os.environ.get("API_KEY") or app.config.custom_data.get("api_key")

    # Create HTTP client with authentication
    client = httpx.AsyncClient(
        base_url="https://api.example.com",
        headers={"Authorization": f"Bearer {api_key}"}
    )

    app.register_action_param("http_client", client)
```

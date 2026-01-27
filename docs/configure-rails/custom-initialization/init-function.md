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

Action parameters registered in `config.py` are automatically injected into actions that declare them:

**config.py:**

```python
from nemoguardrails import LLMRails

def init(app: LLMRails):
    # Initialize shared resources
    db = DatabaseConnection(host="localhost", port=5432)
    api_client = ExternalAPIClient(api_key="...")

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
import asyncpg
from nemoguardrails import LLMRails

async def create_db_pool():
    return await asyncpg.create_pool(
        host="localhost",
        database="mydb",
        user="user",
        password="password"
    )

def init(app: LLMRails):
    import asyncio

    # Create connection pool
    loop = asyncio.get_event_loop()
    db_pool = loop.run_until_complete(create_db_pool())

    # Register for use in actions
    app.register_action_param("db_pool", db_pool)
```

## Example: API Client Initialization

```python
import httpx
from nemoguardrails import LLMRails

def init(app: LLMRails):
    # Get API key from custom_data in config.yml
    api_key = app.config.custom_data.get("api_key")

    # Create HTTP client with authentication
    client = httpx.AsyncClient(
        base_url="https://api.example.com",
        headers={"Authorization": f"Bearer {api_key}"}
    )

    app.register_action_param("http_client", client)
```

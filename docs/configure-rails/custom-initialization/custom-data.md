---
title: Custom Configuration Data
description: Pass and access custom data from config.yml in your initialization code and actions.
---

# Custom Configuration Data

The `custom_data` field in `config.yml` allows you to pass additional configuration to your custom initialization code and actions.

## Defining Custom Data

Add a `custom_data` section to your `config.yml`:

```yaml
models:
  - type: main
    engine: openai
    model: gpt-4

custom_data:
  api_endpoint: "https://api.example.com"
  api_key: "${API_KEY}"  # Environment variable
  max_retries: 3
  timeout_seconds: 30
  feature_flags:
    enable_caching: true
    debug_mode: false
```

## Accessing in config.py

Access custom data in your `init` function:

```python
from nemoguardrails import LLMRails

def init(app: LLMRails):
    # Access custom_data from the configuration
    custom_data = app.config.custom_data

    # Get individual values
    api_endpoint = custom_data.get("api_endpoint")
    api_key = custom_data.get("api_key")
    max_retries = custom_data.get("max_retries", 3)  # with default

    # Access nested values
    feature_flags = custom_data.get("feature_flags", {})
    enable_caching = feature_flags.get("enable_caching", False)

    # Use to configure your providers
    client = APIClient(
        endpoint=api_endpoint,
        api_key=api_key,
        max_retries=max_retries
    )

    app.register_action_param("api_client", client)
```

## Accessing in Actions

You can also access custom data directly in actions via the `config` parameter:

```python
from nemoguardrails.actions import action

@action()
async def my_action(config=None):
    """Access custom_data via the config parameter."""
    custom_data = config.custom_data
    timeout = custom_data.get("timeout_seconds", 30)

    # Use the configuration
    return await do_something(timeout=timeout)
```

## Environment Variables

Use environment variable substitution for sensitive values:

**config.yml:**

```yaml
custom_data:
  database_url: "${DATABASE_URL}"
  api_key: "${API_KEY}"
  secret_key: "${SECRET_KEY:-default_value}"  # with default
```

**Shell:**

```bash
export DATABASE_URL="postgresql://user:pass@localhost/db"
export API_KEY="sk-..."
```

## Example: Multi-Environment Configuration

**config.yml:**

```yaml
custom_data:
  environment: "${ENV:-development}"

  # Database configuration
  database:
    host: "${DB_HOST:-localhost}"
    port: "${DB_PORT:-5432}"
    name: "${DB_NAME:-myapp}"

  # API configuration
  api:
    base_url: "${API_BASE_URL:-http://localhost:8000}"
    timeout: 30

  # Feature toggles
  features:
    rate_limiting: "${ENABLE_RATE_LIMIT:-false}"
    caching: true
```

**config.py:**

```python
from nemoguardrails import LLMRails

def init(app: LLMRails):
    custom_data = app.config.custom_data

    env = custom_data.get("environment")
    db_config = custom_data.get("database", {})
    api_config = custom_data.get("api", {})

    # Configure based on environment
    if env == "production":
        # Production-specific setup
        pass
    else:
        # Development setup
        pass

    # Initialize database
    db = Database(
        host=db_config.get("host"),
        port=db_config.get("port"),
        name=db_config.get("name")
    )

    app.register_action_param("db", db)
```

## Best Practices

1. **Use environment variables for secrets**: Never hardcode API keys or passwords.

2. **Provide defaults**: Use `.get("key", default)` for optional values.

3. **Document your custom_data schema**: Add comments in config.yml explaining expected fields.

4. **Validate configuration**: Check required fields in `init()` and raise clear errors.

```python
def init(app: LLMRails):
    custom_data = app.config.custom_data

    # Validate required fields
    required_fields = ["api_endpoint", "api_key"]
    missing = [f for f in required_fields if not custom_data.get(f)]

    if missing:
        raise ValueError(f"Missing required custom_data fields: {missing}")
```

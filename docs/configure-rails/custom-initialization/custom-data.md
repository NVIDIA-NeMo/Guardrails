---
title:
  page: Custom Configuration Data for NeMo Guardrails
  nav: Custom Data
description: Pass and access custom data from config.yml in your initialization code and actions.
topics:
- Configuration
- Customization
tags:
- custom_data
- config.yml
- Python
content:
  type: how_to
  difficulty: technical_intermediate
  audience:
  - engineer
  - AI Engineer
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
    max_retries = custom_data.get("max_retries", 3)  # with default

    # Access nested values
    feature_flags = custom_data.get("feature_flags", {})
    enable_caching = feature_flags.get("enable_caching", False)

    # Load sensitive values from environment variables
    import os
    api_key = os.environ.get("API_KEY")

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

## Sensitive Configuration

For sensitive values like API keys, use the `api_key_env_var` field on model configurations or load environment variables in your `init()` function:

**config.py:**

```python
import os
from nemoguardrails import LLMRails

def init(app: LLMRails):
    custom_data = app.config.custom_data

    api_key = os.environ.get("API_KEY")
    db_url = os.environ.get("DATABASE_URL", "postgresql://localhost/myapp")

    client = APIClient(
        endpoint=custom_data.get("api_endpoint"),
        api_key=api_key,
    )

    app.register_action_param("api_client", client)
```

```{note}
The `custom_data` field in `config.yml` uses standard YAML parsing and does **not** support inline environment variable substitution (e.g., `${VAR}`). Load sensitive values from environment variables in your `init()` function instead.
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

---
title:
  page: Creating Custom Actions
  nav: Creating Actions
description: Create custom actions using the @action decorator to integrate Python logic.
keywords:
  - nemo guardrails create action
  - guardrails action decorator
  - custom python action guardrails
  - async action guardrails
topics:
  - generative_ai
  - developer_tools
tags:
  - llms
  - ai_inference
  - security_for_ai
content:
  type: how_to
  difficulty: technical_intermediate
  audience:
    - engineer
---

# Creating Custom Actions

This section describes how to create custom actions in the `actions.py` file.

## The `@action` Decorator

Use the `@action` decorator from `nemoguardrails.actions` to define custom actions:

```python
from nemoguardrails.actions import action

@action()
async def my_custom_action():
    """A simple custom action."""
    return "result"
```

## Decorator Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `name` | `str` | Custom name for the action | Function name |
| `is_system_action` | `bool` | Always run locally, bypassing the actions server | `False` |
| `execute_async` | `bool` | Don't block event processing while the action runs (Colang 2.x only) | `False` |
| `output_mapping` | `Callable[[Any], bool]` | Function to interpret the action result for blocking decisions | `default_output_mapping` |

### Custom Action Name

Override the default action name:

```python
@action(name="validate_user_input")
async def check_input(text: str):
    """Validates user input."""
    return len(text) > 0
```

Call from Colang:

```text
$is_valid = execute validate_user_input(text=$user_message)
```

### System Actions

When `is_system_action=True`, the action always runs locally, even when an `actions_server_url` is configured. This is important for actions that need access to special parameters like `context`, `llm`, `config`, and `events`, which are only injected for locally-run actions.

```{note}
When no `actions_server_url` is configured, all actions run locally and receive special parameters regardless of the `is_system_action` setting. The flag only affects behavior when an actions server is in use.
```

```python
@action(is_system_action=True)
async def check_policy_compliance(context: Optional[dict] = None):
    """Check if message complies with policy."""
    message = context.get("last_user_message", "")
    # Validation logic
    return True
```

### Async Execution

When `execute_async=True`, the event processing loop does not wait for the action to complete before continuing. The action runs in the background and the result is picked up later via polling. This is useful for long-running operations where you don't need the result immediately.

```{note}
This flag is only supported in the Colang 2.x runtime. In the Colang 1.0 runtime, it is stored in metadata but has no effect.
```

```python
@action(execute_async=True)
async def call_external_api(endpoint: str):
    """Call an external API without blocking event processing."""
    response = await http_client.get(endpoint)
    return response.json()
```

### Output Mapping

The `output_mapping` parameter controls how the action's return value is interpreted to determine if output should be blocked. It accepts a callable that takes the return value and returns `True` if the output is **not safe** (should be blocked).

When no `output_mapping` is provided, the default behavior is:
- **Boolean results**: `True` means allowed, `False` means blocked
- **Numeric results**: Values below `0.5` are blocked
- **Other types**: Allowed by default

```python
@action(output_mapping=lambda value: value)
async def check_hallucination(context: Optional[dict] = None):
    """Return True if hallucination detected (blocked), False if safe."""
    return detect_hallucination(context.get("bot_message", ""))
```

```python
@action(is_system_action=True, output_mapping=lambda value: not value)
async def check_output_safety(context: Optional[dict] = None):
    """Return True if safe (allowed), mapped to not-blocked."""
    return is_safe(context.get("bot_message", ""))
```

You can also define a custom mapping function for more complex logic:

```python
def my_custom_mapping(result):
    if isinstance(result, dict):
        return result.get("score", 1.0) < 0.7
    return False

@action(output_mapping=my_custom_mapping)
async def score_safety(context: Optional[dict] = None):
    """Return a dict with a safety score."""
    return {"score": compute_score(context.get("bot_message", ""))}
```

## Function Parameters

Actions can accept parameters of the following types:

| Type | Example |
|------|---------|
| `str` | `"hello"` |
| `int` | `42` |
| `float` | `3.14` |
| `bool` | `True` |
| `list` | `["a", "b", "c"]` |
| `dict` | `{"key": "value"}` |

### Basic Parameters

```python
@action()
async def greet_user(name: str, formal: bool = False):
    """Generate a greeting."""
    if formal:
        return f"Good day, {name}."
    return f"Hello, {name}!"
```

Call from Colang:

```text
$greeting = execute greet_user(name="Alice", formal=True)
```

### Optional Parameters with Defaults

```python
@action()
async def search_documents(
    query: str,
    max_results: int = 10,
    include_metadata: bool = False
):
    """Search documents with optional parameters."""
    results = perform_search(query, limit=max_results)
    if include_metadata:
        return {"results": results, "count": len(results)}
    return results
```

## Return Values

Actions can return various types:

### Simple Return

```python
@action()
async def get_status():
    return "active"
```

### Dictionary Return

```python
@action()
async def get_user_info(user_id: str):
    return {
        "id": user_id,
        "name": "John Doe",
        "role": "admin"
    }
```

### Boolean Return (for validation)

```python
@action(is_system_action=True)
async def is_safe_content(context: Optional[dict] = None):
    content = context.get("bot_message", "")
    # Returns True if safe, False if blocked
    return not contains_harmful_content(content)
```

## Error Handling

Handle errors gracefully within actions:

```python
@action()
async def fetch_data(url: str):
    """Fetch data with error handling."""
    try:
        response = await http_client.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        # Log the error
        print(f"Error fetching data: {e}")
        # Return a safe default or raise
        return None
```

## Example Actions

### Input Validation Action

```python
from typing import Optional
from nemoguardrails.actions import action

@action(is_system_action=True)
async def check_input_length(context: Optional[dict] = None):
    """Ensure user input is not too long."""
    user_message = context.get("last_user_message", "")
    max_length = 1000

    if len(user_message) > max_length:
        return False  # Block the input

    return True  # Allow the input
```

### Output Filtering Action

```python
@action(is_system_action=True)
async def filter_sensitive_data(context: Optional[dict] = None):
    """Check for sensitive data in bot response."""
    bot_response = context.get("bot_message", "")

    sensitive_patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
        r"\b\d{16}\b",              # Credit card pattern
    ]

    import re
    for pattern in sensitive_patterns:
        if re.search(pattern, bot_response):
            return True  # Contains sensitive data

    return False  # No sensitive data found
```

### External API Action

```python
import aiohttp

@action(execute_async=True)
async def query_knowledge_base(query: str, top_k: int = 5):
    """Query an external knowledge base API."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.example.com/search",
            json={"query": query, "limit": top_k}
        ) as response:
            data = await response.json()
            return data.get("results", [])
```

## Related Topics

- [Built-in Actions](built-in-actions) - Default actions in the library
- [Action Parameters](action-parameters) - Special parameters provided automatically
- [Registering Actions](registering-actions) - Different ways to register actions

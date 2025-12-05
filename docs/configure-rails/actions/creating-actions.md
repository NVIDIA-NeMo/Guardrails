---
title: Creating Custom Actions
description: Create custom actions using the @action decorator to integrate Python logic into guardrails flows.
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
| `is_system_action` | `bool` | Mark as system action (runs in guardrails context) | `False` |
| `execute_async` | `bool` | Execute asynchronously without blocking | `False` |

### Custom Action Name

Override the default action name:

```python
@action(name="validate_user_input")
async def check_input(text: str):
    """Validates user input."""
    return len(text) > 0
```

Call from Colang:

```colang
$is_valid = execute validate_user_input(text=$user_message)
```

### System Actions

System actions have access to the guardrails context and are typically used for input/output validation:

```python
@action(is_system_action=True)
async def check_policy_compliance(context: Optional[dict] = None):
    """Check if message complies with policy."""
    message = context.get("last_user_message", "")
    # Validation logic
    return True
```

### Async Execution

For long-running operations, use `execute_async=True` to prevent blocking:

```python
@action(execute_async=True)
async def call_external_api(endpoint: str):
    """Call an external API without blocking."""
    response = await http_client.get(endpoint)
    return response.json()
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

```colang
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

- [Action Parameters](action-parameters) - Special parameters provided automatically
- [Registering Actions](registering-actions) - Different ways to register actions
- [Built-in Actions](built-in-actions) - Default actions in the toolkit

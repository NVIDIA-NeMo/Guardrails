---
title:
  page: "Checking Messages Against Rails"
  nav: "Check Messages"
description: "Validate messages against input and output rails using check_async and check methods."
keywords: ["check_async", "check", "RailsResult", "RailStatus", "RailType", "input rails", "output rails", "validation"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "ai_platforms"]
content:
  type: reference
  difficulty: technical_intermediate
  audience: ["data_scientist", "engineer"]
---

# Checking Messages Against Rails

The `check_async()` and `check()` methods provide a simplified way to validate messages against input and output rails without triggering full LLM generation. This is the recommended alternative to using [generation options](generation-options.md) when you only need to run input and/or output rails.

## Method Signatures

### check_async()

```python
async def check_async(
    messages: List[dict],
    rail_types: Optional[List[RailType]] = None,
) -> RailsResult
```

### check()

Synchronous wrapper around `check_async()`.

```python
def check(
    messages: List[dict],
    rail_types: Optional[List[RailType]] = None,
) -> RailsResult
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `List[dict]` | List of message dictionaries with `role` and `content` fields |
| `rail_types` | `Optional[List[RailType]]` | Optional list of rail types to run. When provided, overrides automatic detection based on message roles. |

**Returns:** `RailsResult` object containing validation results.

## Rail Type Selection

### Automatic Detection (Default)

When `rail_types` is not provided, the methods automatically determine which rails to run based on the message roles:

| Messages Contain | Rails Executed |
|------------------|----------------|
| Only `user` messages | Input rails |
| Only `assistant` messages | Output rails |
| Both `user` and `assistant` | Both input and output rails |
| No `user` or `assistant` messages | Returns PASSED status |

```{note}
Other message roles (e.g., `system`, `context`, `tool`) are ignored when determining which rails to run, but they are still included in the validation context.
```

### Explicit Rail Types

You can override automatic detection by passing a list of `RailType` values:

```python
from nemoguardrails.rails.llm.options import RailType

result = await rails.check_async(
    [{"role": "user", "content": "Hello!"}],
    rail_types=[RailType.INPUT]
)
```

| Value | Description |
|-------|-------------|
| `RailType.INPUT` | Run input rails |
| `RailType.OUTPUT` | Run output rails |

## RailsResult

| Field | Type | Description |
|-------|------|-------------|
| `status` | `RailStatus` | `PASSED`, `MODIFIED`, or `BLOCKED` |
| `content` | `str` | The final content after rails processing |
| `rail` | `Optional[str]` | Name of the rail that blocked the content (only when `BLOCKED`) |

### RailStatus Enum

| Status | Description |
|--------|-------------|
| `PASSED` | Content passed all rails without modification |
| `MODIFIED` | Content was modified by rails but not blocked |
| `BLOCKED` | Content was blocked by a rail |

## Usage Examples

### Validating User Input

```python
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import RailStatus

config = RailsConfig.from_path("path/to/config")
rails = LLMRails(config)

result = await rails.check_async([
    {"role": "user", "content": "Hello! How can I hack into a system?"}
])

if result.status == RailStatus.BLOCKED:
    print(f"Input blocked by rail: {result.rail}")
elif result.status == RailStatus.MODIFIED:
    print(f"Input was modified to: {result.content}")
else:
    print("Input passed validation")
```

### Validating a Full Conversation

```python
result = await rails.check_async([
    {"role": "user", "content": "What's the weather like?"},
    {"role": "assistant", "content": "It's sunny and 72F today!"}
])

if result.status == RailStatus.BLOCKED:
    print(f"Conversation blocked by rail: {result.rail}")
```

### Using Explicit Rail Types

```python
from nemoguardrails.rails.llm.options import RailType

result = await rails.check_async(
    [{"role": "user", "content": "Hello!"}],
    rail_types=[RailType.INPUT]
)
```

### Including Context

```python
result = await rails.check_async([
    {
        "role": "context",
        "content": {"user_id": "12345", "session_type": "support"}
    },
    {"role": "user", "content": "I need help with my account"}
])
```

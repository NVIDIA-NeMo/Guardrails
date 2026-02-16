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

The `check_async()` and `check()` methods validate messages against input and output rails without triggering full LLM generation. Use these methods instead of [generation options](generation-options.md) when you only need to run rails without generating a response.

## Method Signatures

Both methods accept the same parameters and return a `RailsResult` object.

### check_async()

The primary asynchronous method for checking messages against rails.

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

The methods determine which rails to execute based on the message roles or an explicit `rail_types` parameter.

### Automatic Detection (Default)

When `rail_types` is not provided, the methods automatically determine which rails to run based on the message roles:

| Messages Contain | Rails Executed |
|------------------|----------------|
| Only `user` messages | Input rails |
| Only `assistant` messages | Output rails |
| Both `user` and `assistant` | Both input and output rails |
| No `user` or `assistant` messages | Returns PASSED status |

```{note}
The methods ignore other message roles such as `system`, `context`, `tool` when determining which rails to run but still include them in the validation context.
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

The `RailsResult` object contains the outcome of the rails check.

| Field | Type | Description |
|-------|------|-------------|
| `status` | `RailStatus` | `PASSED`, `MODIFIED`, or `BLOCKED` |
| `content` | `str` | The final content after rails processing |
| `rail` | `Optional[str]` | Name of the rail that blocked the content (only when `BLOCKED`) |

### RailStatus Enum

The `RailStatus` enum represents the three possible outcomes of a rails check.

| Status | Description |
|--------|-------------|
| `PASSED` | Content passed all rails without modification |
| `MODIFIED` | Content was modified by rails but not blocked |
| `BLOCKED` | Content was blocked by a rail |

## Usage Examples

The following examples demonstrate common patterns for validating messages with `check_async()`.

### Validating User Input

Check a single user message against input rails and handle each possible status.

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

Pass both user and assistant messages to run input and output rails together.

```python
result = await rails.check_async([
    {"role": "user", "content": "What's the weather like?"},
    {"role": "assistant", "content": "It's sunny and 72F today!"}
])

if result.status == RailStatus.BLOCKED:
    print(f"Conversation blocked by rail: {result.rail}")
```

### Including Context

Pass context variables alongside user or assistant messages to provide additional information for rail evaluation.
Context messages use the `context` role with a dictionary value for `content`.

```python
result = await rails.check_async([
    {
        "role": "context",
        "content": {"user_id": "12345", "session_type": "support"}
    },
    {"role": "user", "content": "I need help with my account"}
])
```

For more information about context variables, refer to [](core-classes.md#passing-context).

---

## Related Resources

- [](core-classes.md) - `LLMRails` and `RailsConfig` class reference
- [](generation-options.md) - Fine-grained control over generation with the `options` parameter

---
title: Observability
description: Monitor and troubleshoot your guardrails applications.
---

# Logging and Debugging

This guide covers the various methods for logging, debugging, and understanding what happens during guardrails generation.

## Overview

The NeMo Guardrails toolkit provides multiple ways to inspect and debug guardrails generation:

| Method | Use Case |
|--------|----------|
| **Verbose Mode** | Real-time console logging during development |
| **Explain Method** | Quick summary of the last generation |
| **Generation Options (log)** | Detailed structured logs returned with responses |
| **Output Variables** | Return specific context variables |

## Verbose Mode

Enable detailed console logging by setting `verbose=True` when creating the `LLMRails` instance:

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("path/to/config")
rails = LLMRails(config, verbose=True)
```

This outputs detailed information about:

- LLM calls and their prompts/completions
- Rail activations and decisions
- Action executions
- Flow transitions

## Explain Method

Get a quick summary of the last generation using the `explain()` method:

```python
response = rails.generate(messages=[
    {"role": "user", "content": "Hello!"}
])

info = rails.explain()
info.print_llm_calls_summary()
```

The `ExplainInfo` object provides methods to inspect:

- LLM calls summary
- Colang history
- Generated events

## Generation Options: Log

For detailed structured logging, use the `log` generation option. This returns comprehensive information about what happened during generation.

### Enabling Log Options

```python
response = rails.generate(
    messages=[{"role": "user", "content": "Hello!"}],
    options={
        "log": {
            "activated_rails": True,
            "llm_calls": True,
            "internal_events": True,
            "colang_history": True
        }
    }
)
```

### Log Option Reference

| Option | Description |
|--------|-------------|
| `activated_rails` | Detailed information about rails activated during generation |
| `llm_calls` | Information about all LLM calls (prompt, completion, tokens, timing) |
| `internal_events` | Array of internal generated events |
| `colang_history` | Conversation history in Colang format |

### Response Structure

```json
{
  "response": [...],
  "log": {
    "activated_rails": [...],
    "stats": {...},
    "llm_calls": [...],
    "internal_events": [...],
    "colang_history": "..."
  }
}
```

### Using print_summary()

The log object has a `print_summary()` method for a human-readable overview:

```python
response.log.print_summary()
```

**Example output:**

```text
# General stats

- Total time: 2.85s
  - [0.56s][19.64%]: INPUT Rails
  - [1.40s][49.02%]: DIALOG Rails
  - [0.58s][20.22%]: GENERATION Rails
  - [0.31s][10.98%]: OUTPUT Rails
- 5 LLM calls, 2.74s total duration, 1641 total prompt tokens, 103 total completion tokens, 1744 total tokens.

# Detailed stats

- [0.56s] INPUT (self check input): 1 actions (self_check_input), 1 llm calls [0.56s]
- [0.43s] DIALOG (generate user intent): 1 actions (generate_user_intent), 1 llm calls [0.43s]
- [0.96s] DIALOG (generate next step): 1 actions (generate_next_step), 1 llm calls [0.95s]
- [0.58s] GENERATION (generate bot message): 2 actions (retrieve_relevant_chunks, generate_bot_message), 1 llm calls [0.49s]
- [0.31s] OUTPUT (self check output): 1 actions (self_check_output), 1 llm calls [0.31s]
```

### Accessing Detailed Data

Access specific log components programmatically:

```python
# Access LLM calls
for call in response.log.llm_calls:
    print(f"Task: {call.task}")
    print(f"Duration: {call.duration}s")
    print(f"Prompt tokens: {call.prompt_tokens}")
    print(f"Completion tokens: {call.completion_tokens}")
    print(f"Total tokens: {call.total_tokens}")

# Access activated rails
for rail in response.log.activated_rails:
    print(f"Type: {rail.type}, Name: {rail.name}")
    print(f"Decisions: {rail.decisions}")
    print(f"Duration: {rail.duration}s")

# Access stats
stats = response.log.stats
print(f"Total duration: {stats.total_duration}s")
print(f"Input rails: {stats.input_rails_duration}s")
print(f"Dialog rails: {stats.dialog_rails_duration}s")
print(f"Output rails: {stats.output_rails_duration}s")
```

## Output Variables

Return specific context variables using the `output_vars` option:

### Return Specific Variables

```python
response = rails.generate(
    messages=[{"role": "user", "content": "Hello!"}],
    options={
        "output_vars": ["triggered_input_rail", "triggered_output_rail"]
    }
)

print(response.output_data)
# {'triggered_input_rail': None, 'triggered_output_rail': None}
```

### Return All Context Variables

Set `output_vars` to `True` to return the complete context:

```python
response = rails.generate(
    messages=[{"role": "user", "content": "Hello!"}],
    options={
        "output_vars": True
    }
)

# Access all context data
print(response.output_data.keys())
```

### Common Output Variables

| Variable | Description |
|----------|-------------|
| `last_user_message` | The last user message |
| `last_bot_message` | The last bot message |
| `triggered_input_rail` | Name of input rail that triggered (if any) |
| `triggered_output_rail` | Name of output rail that triggered (if any) |
| `relevant_chunks` | Retrieved knowledge base chunks |
| `allowed` | Whether the input was allowed |

## Combining Log and Output Variables

Use both options together for comprehensive debugging:

```python
response = rails.generate(
    messages=[{"role": "user", "content": "Tell me about the company."}],
    options={
        "output_vars": ["triggered_input_rail", "relevant_chunks"],
        "log": {
            "activated_rails": True,
            "llm_calls": True
        }
    }
)

# Check if any rail was triggered
if response.output_data.get("triggered_input_rail"):
    print(f"Input blocked by: {response.output_data['triggered_input_rail']}")

# Inspect what happened
response.log.print_summary()
```

## Debugging Common Issues

### Input Blocked Unexpectedly

```python
response = rails.generate(
    messages=[{"role": "user", "content": "Your message"}],
    options={
        "output_vars": ["triggered_input_rail"],
        "log": {"activated_rails": True}
    }
)

if response.output_data.get("triggered_input_rail"):
    # Find the input rail that blocked
    for rail in response.log.activated_rails:
        if rail.type == "input" and rail.stop:
            print(f"Blocked by: {rail.name}")
            # Check the LLM decision
            for action in rail.executed_actions:
                for llm_call in action.llm_calls:
                    print(f"Prompt: {llm_call.prompt}")
                    print(f"Completion: {llm_call.completion}")
```

### Understanding Flow Execution

```python
response = rails.generate(
    messages=[{"role": "user", "content": "Hello!"}],
    options={
        "log": {
            "internal_events": True,
            "colang_history": True
        }
    }
)

# View internal events
for event in response.log.internal_events:
    print(f"{event['type']}: {event}")

# View Colang history
print(response.log.colang_history)
```

### Analyzing LLM Performance

```python
response = rails.generate(
    messages=[{"role": "user", "content": "Hello!"}],
    options={"log": {"llm_calls": True}}
)

total_tokens = 0
total_duration = 0

for call in response.log.llm_calls:
    print(f"Task: {call.task}")
    print(f"  Duration: {call.duration:.2f}s")
    print(f"  Tokens: {call.total_tokens}")
    total_tokens += call.total_tokens
    total_duration += call.duration

print(f"\nTotal: {total_tokens} tokens in {total_duration:.2f}s")
```

## Server API Logging

When using the server API, include options in the request body:

```json
{
    "config_id": "my_config",
    "messages": [{"role": "user", "content": "Hello!"}],
    "options": {
        "output_vars": ["triggered_input_rail"],
        "log": {
            "activated_rails": true,
            "llm_calls": true
        }
    }
}
```

## Complete Debugging Example

```python
from nemoguardrails import LLMRails, RailsConfig

# Enable verbose mode for console output
config = RailsConfig.from_path("path/to/config")
rails = LLMRails(config, verbose=True)

# Generate with full logging
response = rails.generate(
    messages=[{"role": "user", "content": "What is the company policy?"}],
    options={
        "output_vars": True,
        "log": {
            "activated_rails": True,
            "llm_calls": True,
            "internal_events": True,
            "colang_history": True
        }
    }
)

# Print summary
print("=== Generation Summary ===")
response.log.print_summary()

# Check for blocked content
print("\n=== Rail Triggers ===")
print(f"Input rail triggered: {response.output_data.get('triggered_input_rail')}")
print(f"Output rail triggered: {response.output_data.get('triggered_output_rail')}")

# Analyze LLM calls
print("\n=== LLM Calls ===")
for call in response.log.llm_calls:
    print(f"{call.task}: {call.total_tokens} tokens, {call.duration:.2f}s")

# View final response
print(f"\n=== Response ===")
print(response.response[0]["content"])
```

## Related Resources

- [Tracing](../tracing/index.md) - Production monitoring and observability with OpenTelemetry

---
title:
  page: "Exceptions and Error Handling"
  nav: "Exceptions"
description: "Raise and handle exceptions in guardrails flows to control error behavior and custom responses."
keywords: ["nemo guardrails exceptions", "guardrails error handling", "InputRailException", "OutputRailException"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "security_for_ai"]
content:
  type: reference
  difficulty: technical_intermediate
  audience: ["engineer"]
---

# Exceptions and Error Handling

NeMo Guardrails supports raising exceptions from within flows.
An exception is an event whose name ends with `Exception`, e.g., `InputRailException`.
When an exception is raised, the final output is a message with the role set to `exception` and the content
set to additional information about the exception. For example:

```text
define flow input rail example
  # ...
  create event InputRailException(message="Input not allowed.")
```

```json
{
  "role": "exception",
  "content": {
    "type": "InputRailException",
    "uid": "45a452fa-588e-49a5-af7a-0bab5234dcc3",
    "event_created_at": "9999-99-99999:24:30.093749+00:00",
    "source_uid": "NeMoGuardrails",
    "message": "Input not allowed."
  }
}
```

---

## Guardrails Library Exception

By default, all the guardrails included in the [](guardrail-catalog/index.md) return a predefined message
when a rail is triggered. You can change this behavior by setting the `enable_rails_exceptions` key to `True` in your
`config.yml` file:

```yaml
enable_rails_exceptions: True
```

When this setting is enabled, the rails are triggered, they will return an exception message.
To understand better what is happening under the hood, here's how the `self check input` rail is implemented:

```text
define flow self check input
  $allowed = execute self_check_input
  if not $allowed
    if $config.enable_rails_exceptions
      create event InputRailException(message="Input not allowed. The input was blocked by the 'self check input' flow.")
    else
      bot refuse to respond
      stop
```

```{note}
In Colang 2.x, you must change `$config.enable_rails_exceptions` to `$system.config.enable_rails_exceptions` and `create event` to `send`.
```

When the `self check input` rail is triggered, the following exception is returned.

```json
{
  "role": "exception",
  "content": {
    "type": "InputRailException",
    "uid": "45a452fa-588e-49a5-af7a-0bab5234dcc3",
    "event_created_at": "9999-99-99999:24:30.093749+00:00",
    "source_uid": "NeMoGuardrails",
    "message": "Input not allowed. The input was blocked by the 'self check input' flow."
  }
}
```

---

## Exception Types

The NeMo Guardrails library includes additional exception types for specific integrations:

- `LlamaGuardInputRailException` / `LlamaGuardOutputRailException`
- `JailbreakDetectionRailException`
- `ContentSafetyCheckInputException` / `ContentSafetyCheckOutputException`
- `FactCheckRailException`
- `SelfCheckHallucinationRailException`
- `InjectionDetectionRailException`

Each library rail raises its own exception type when `enable_rails_exceptions` is enabled.

---

## Custom Exception Handling

You can create custom exception types by following the naming convention of ending with `Exception`:

```text
define flow custom validation
  if not $custom_condition
    create event CustomValidationException(message="Custom validation failed.")
```

---

## Exception Response Format

All exceptions follow a consistent JSON format:

```json
{
  "role": "exception",
  "content": {
    "type": "ExceptionType",
    "uid": "unique-identifier",
    "event_created_at": "timestamp",
    "source_uid": "source-identifier",
    "message": "Human-readable error message"
  }
}
```

### Field Descriptions

- **type**: The exception type (e.g., `InputRailException`)
- **uid**: A unique identifier for the exception instance
- **event_created_at**: Timestamp when the exception was created
- **source_uid**: Identifier for the source that created the exception
- **message**: Human-readable description of what went wrong

---

## Handling Exceptions in Applications

When integrating NeMo Guardrails with your application, you should handle exceptions appropriately:

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

try:
    response = rails.generate(messages=[{"role": "user", "content": "Hello"}])

    if response.get("role") == "exception":
        # Handle the exception
        exception_content = response.get("content", {})
        exception_type = exception_content.get("type")
        exception_message = exception_content.get("message")

        # Log the exception or take appropriate action
        print(f"Exception {exception_type}: {exception_message}")

        # Provide fallback response to user
        fallback_response = "I'm sorry, but I cannot process that request at the moment."
    else:
        # Process normal response
        print(response.get("content", ""))

except Exception as e:
    # Handle other errors
    print(f"Error: {e}")
```

---

## Best Practices

1. **Use Descriptive Messages**: Provide clear, actionable error messages in your exceptions.

2. **Log Exceptions**: Always log exceptions for debugging and monitoring purposes.

3. **Graceful Degradation**: Provide fallback responses when exceptions occur.

4. **User-Friendly Messages**: Translate technical exception messages into user-friendly responses.

5. **Exception Categories**: Use appropriate exception types to categorize different kinds of errors.

6. **Configuration Control**: Use the `enable_rails_exceptions` setting to control whether rails return exceptions or predefined messages.

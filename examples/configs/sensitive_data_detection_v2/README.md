# Presidio-based Sensitive Data Detection Example

This example demonstrates how to detect and redact sensitive data using [Presidio](https://github.com/Microsoft/presidio).

## Prerequisites

- `Presidio`

  You can install it with:

  ```bash
  uv sync --group dev --extra sdd
  ```

- `en_core_web_lg` spaCy model

  You can download it with:

  ```bash
  uv run python -m spacy download en_core_web_lg
  ```

## Running example

To test this configuration, run the CLI chat from the `examples/configs/sensitive_data_detection_v2` directory:

```bash
uv run nemoguardrails chat --config=.
```

## Documentation

- [Presidio-based Sensitive Data Detection configuration](../../../docs/user-guides/guardrails-library.md#presidio-based-sensitive-data-detection)
- [Presidio Integration guide](../../../docs/user-guides/community/presidio.md)

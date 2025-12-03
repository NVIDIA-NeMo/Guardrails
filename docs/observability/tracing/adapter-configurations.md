# Adapter Configurations

You can set up the following adapters for tracing.

The following table summarizes the list of adapters supported by NeMo Guardrails and their use cases.

| Adapter Type | Use Case | Configuration |
|---------|----------|---------------|
| [FileSystem](filesystem-adapter) | Development, debugging, local logging | `filepath: "./logs/traces.jsonl"` |
| [OpenTelemetry](opentelemetry-adapter) | Production, monitoring platforms, distributed systems | Requires SDK configuration |
| [Custom](custom-adapter) | Specialized backends or formats | Implement `InteractionLogAdapter` |

The following sections explain how to configure each adapter in `config.yml`.

(filesystem-adapter)=

## FileSystem Adapter

For development and debugging, use the `FileSystem` adapter to log traces locally.

```yaml
tracing:
  enabled: true
  adapters:
    - name: FileSystem
      filepath: "./logs/traces.jsonl"
```

For working examples, refer to the [Tracing Guardrails Quickstart](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/docs/getting-started/8-tracing/1_tracing_quickstart.ipynb) notebook.

(opentelemetry-adapter)=

## OpenTelemetry Adapter

For production environments with observability platforms.

```yaml
tracing:
  enabled: true
  adapters:
    - name: OpenTelemetry
```

For working examples, refer to the [Tracing Guardrails with Jaeger](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/docs/getting-started/8-tracing/2_tracing_with_jaeger.ipynb) notebook.

```{important}
OpenTelemetry requires additional SDK configuration in your application code. See the sections below for setup instructions.
```

(custom-adapter)=

## Custom Adapter

You can create custom adapters and use them in your application code.

1. Create custom adapters for specialized backends or formats for your use case.

    ```python
    from nemoguardrails.tracing.adapters.base import InteractionLogAdapter

    class MyCustomAdapter(InteractionLogAdapter):
        name = "MyCustomAdapter"

        def __init__(self, custom_option: str):
            self.custom_option = custom_option

        def transform(self, interaction_log):
            # Transform logic for your backend
            pass
    ```

2. Register the adapter in `config.py`.

    ```python
    from nemoguardrails.tracing.adapters.registry import register_log_adapter
    register_log_adapter(MyCustomAdapter, "MyCustomAdapter")
    ```

3. Use the adapter in `config.yml`.

    ```yaml
    tracing:
      enabled: true
      adapters:
        - name: MyCustomAdapter
          custom_option: "value"
    ```

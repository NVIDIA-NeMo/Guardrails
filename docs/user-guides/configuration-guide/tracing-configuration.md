# Tracing Configuration

NeMo Guardrails includes tracing capabilities to monitor and debug your guardrails interactions. Tracing helps you understand:

- Which rails are activated during conversations
- LLM call patterns and performance
- Flow execution paths and timing
- Error conditions and debugging information

### Basic Configuration

Enable tracing in your `config.yml`:

```yaml
tracing:
  enabled: true
  adapters:
    - name: FileSystem
      filepath: "./logs/traces.jsonl"
```

This configuration logs traces to local JSON files, which is suitable for development and debugging.

### OpenTelemetry Integration

For production environments and integration with observability platforms:

```yaml
tracing:
  enabled: true
  adapters:
    - name: OpenTelemetry
```

```{important}
Install tracing dependencies: `pip install nemoguardrails[tracing]`
```

```{note}
OpenTelemetry integration requires configuring the OpenTelemetry SDK in your application code. NeMo Guardrails follows OpenTelemetry best practices where libraries use only the API and applications configure the SDK. See the [Tracing Guide](/docs/user-guides/tracing/index.md) for detailed setup instructions and examples.
```

### Configuration Options

| Adapter | Use Case | Configuration |
|---------|----------|---------------|
| FileSystem | Development, debugging, simple logging | `filepath: "./logs/traces.jsonl"` |
| OpenTelemetry | Production, monitoring platforms, distributed systems | Requires application-level SDK configuration |

For advanced configuration, custom adapters, and production deployment examples, see the [detailed tracing guide](user-guides/tracing/index.md).

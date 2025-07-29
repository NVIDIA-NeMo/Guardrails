# Tracing Guide

This guide explains how to set up tracing with NeMo Guardrails to monitor and debug your guardrails interactions.

## Overview

Tracing provides observability into your guardrails execution, helping you:

- Track which rails are activated during conversations
- Monitor LLM calls and their performance
- Debug flow execution and identify bottlenecks
- Analyze conversation patterns and errors

## Quick Start

Here's a minimal working example to see tracing in action:

```bash
pip install nemoguardrails[tracing] opentelemetry-sdk
```

Create a simple tracing example:

```python
# trace_example.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from nemoguardrails import LLMRails, RailsConfig

# Configure OpenTelemetry
resource = Resource.create({"service.name": "guardrails-quickstart"})
tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

# Configure guardrails with tracing
config_yaml = """
models:
  - type: main
    engine: openai
    model: gpt-4o-mini

rails:
  config:
    streaming: true

tracing:
  enabled: true
  adapters:
    - name: OpenTelemetry
"""

config = RailsConfig.from_content(yaml_content=config_yaml)
rails = LLMRails(config)
response = rails.generate(messages=[{"role": "user", "content": "Hello!"}])
print(f"Response: {response}")
```

```bash
python trace_example.py
```

## Configuration

### FileSystem Adapter

For development and debugging, use the FileSystem adapter to log traces locally:

```yaml
tracing:
  enabled: true
  adapters:
    - name: FileSystem
      filepath: "./logs/traces.jsonl"
```

### OpenTelemetry Adapter

For production environments with observability platforms:

```yaml
tracing:
  enabled: true
  adapters:
    - name: OpenTelemetry
```

```{important}
OpenTelemetry requires additional SDK configuration in your application code. See the sections below for setup instructions.
```

## Adapter Types

| Adapter | Use Case | Configuration |
|---------|----------|---------------|
| FileSystem | Development, debugging, local logging | `filepath: "./logs/traces.jsonl"` |
| OpenTelemetry | Production, monitoring platforms, distributed systems | Requires SDK configuration |
| Custom | Specialized backends or formats | Implement `InteractionLogAdapter` |

## OpenTelemetry Integration

NeMo Guardrails follows OpenTelemetry best practices: libraries use only the API while applications configure the SDK.

### Installation

```bash
# Basic tracing support
pip install nemoguardrails[tracing]

# For development with SDK
pip install nemoguardrails[tracing] opentelemetry-sdk

# Production with exporters
pip install opentelemetry-sdk opentelemetry-exporter-otlp
```

### Setup Examples

#### Console Output (Development)

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

# Configure OpenTelemetry before NeMo Guardrails
resource = Resource.create({"service.name": "my-guardrails-app"})
tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider)

console_exporter = ConsoleSpanExporter()
tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))

# Configure NeMo Guardrails
from nemoguardrails import LLMRails, RailsConfig

config_yaml = """
models:
  - type: main
    engine: openai
    model: gpt-4o-mini

tracing:
  enabled: true
  adapters:
    - name: OpenTelemetry
"""

config = RailsConfig.from_content(yaml_content=config_yaml)

rails = LLMRails(config)
```

#### OTLP Exporter (Production)

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

resource = Resource.create({"service.name": "my-guardrails-app"})
tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider)

otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

# Use with NeMo Guardrails as above
```

## Ecosystem Compatibility

NeMo Guardrails works with the entire OpenTelemetry ecosystem including:

- **Exporters**: Jaeger, Zipkin, Prometheus, New Relic, Datadog, AWS X-Ray, Google Cloud Trace
- **Collectors**: OpenTelemetry Collector, vendor-specific collectors
- **Backends**: Any system accepting OpenTelemetry traces

See the [OpenTelemetry Registry](https://opentelemetry.io/ecosystem/registry/) for the complete list.


## Custom Adapters

Create custom adapters for specialized backends or formats:

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

Register in `config.py`:

```python
from nemoguardrails.tracing.adapters.registry import register_log_adapter
register_log_adapter(MyCustomAdapter, "MyCustomAdapter")
```

Use in `config.yml`:

```yaml
tracing:
  enabled: true
  adapters:
    - name: MyCustomAdapter
      custom_option: "value"
```

## Migration Guide

### Breaking Changes

Old configuration in `config.yml` is no longer supported:

```yaml
#  No longer supported
tracing:
  enabled: true
  adapters:
    - name: OpenTelemetry
      service_name: "my-service"
      exporter: "console"
```

New approach - configure OpenTelemetry in your application:

```python
#  Configure in application code
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

tracer_provider = TracerProvider()
trace.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

# Simple config in config.yml
config_yaml = """
tracing:
  enabled: true
  adapters:
    - name: OpenTelemetry
"""
config = RailsConfig.from_content(yaml_content=config_yaml)
```

### Deprecated Functions

- `register_otel_exporter()` - will be removed in v0.16.0
- Configure exporters directly in your application instead

This follows OpenTelemetry best practices where libraries use only the API and applications configure the SDK.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No traces appear | Configure OpenTelemetry SDK in application code; verify `tracing.enabled: true` |
| Connection errors | Check collector is running; test with `ConsoleSpanExporter` first |
| Import errors | Install dependencies: `pip install nemoguardrails[tracing]` |
| Wrong service name | Set `Resource` with `service.name` in application code |

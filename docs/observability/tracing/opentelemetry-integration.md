# Advanced OpenTelemetry Integration

NeMo Guardrails follows OpenTelemetry best practices; libraries use only the API while applications configure the SDK. The following sections explain how to install and configure the OpenTelemetry SDK.

## Installation

Choose one of the following options for installing the NeMo Guardrails toolkit with tracing support, the OpenTelemetry SDK, and the OpenTelemetry Protocol (OTLP) exporter.

- For basic tracing support in the NeMo Guardrails toolkit:

  ```bash
  pip install nemoguardrails[tracing]
  ```

- For development with the OpenTelemetry SDK:

  ```bash
  pip install nemoguardrails[tracing] opentelemetry-sdk
  ```

- For production with the OpenTelemetry SDK and the OpenTelemetry Protocol (OTLP) exporter:

  ```bash
  pip install nemoguardrails[tracing] opentelemetry-sdk opentelemetry-exporter-otlp
  ```

## Configuration Examples

The following examples show how to configure the NeMo Guardrails client with the OpenTelemetry SDK for development and production use cases.

### Console Output (Development)

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

### OTLP Exporter (Production)

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

## OpenTelemetry Ecosystem Compatibility

NeMo Guardrails works with the entire OpenTelemetry ecosystem including:

- **Exporters**: Jaeger, Zipkin, Prometheus, New Relic, Datadog, AWS X-Ray, Google Cloud Trace
- **Collectors**: OpenTelemetry Collector, vendor-specific collectors
- **Backends**: Any system accepting OpenTelemetry traces

See the [OpenTelemetry Registry](https://opentelemetry.io/ecosystem/registry/) for the complete list.

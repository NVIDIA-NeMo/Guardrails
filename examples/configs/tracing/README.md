# NeMo Guardrails Tracing

This guide explains how to set up tracing with NeMo Guardrails to monitor and debug your guardrails interactions.

## What is Tracing?

Tracing helps you understand what happens inside your guardrails:

- Track which rails are activated
- Monitor LLM calls and responses
- Debug performance issues
- Analyze conversation flows

## Quick Start

### 1. Try the Working Example

The fastest way to see tracing in action:

```bash
# Install tracing support with SDK (needed for examples)
pip install nemoguardrails[tracing] opentelemetry-sdk

cd examples/configs/tracing/
python working_example.py
```

This will show traces printed to your console immediately.

### 2. Basic Configuration

Enable tracing in your `config.yml`:

```yaml
tracing:
  enabled: true
  adapters:
    - name: FileSystem
```

Or use OpenTelemetry (requires additional setup):

```yaml
tracing:
  enabled: true
  adapters:
    - name: OpenTelemetry
```

## Available Tracing Adapters

### FileSystem Adapter (Easiest)

Logs traces to local JSON files which is a good option for development and debugging:

```yaml
tracing:
  enabled: true
  adapters:
    - name: FileSystem
      filepath: "./logs/traces.jsonl"
```

**When to use**: Development, debugging, simple logging needs.

### OpenTelemetry Adapter

```yaml
tracing:
  enabled: true
  adapters:
    - name: OpenTelemetry
```

**When to use**: Production environments, integration with monitoring systems, distributed applications.

## OpenTelemetry Ecosystem Compatibility

**NeMo Guardrails is compatible with the entire OpenTelemetry ecosystem.** The examples below show common configurations, but you can use any OpenTelemetry compatible:

- **Exporters**: Jaeger, Zipkin, Prometheus, New Relic, Datadog, AWS X-Ray, Google Cloud Trace, and many more
- **Collectors**: OpenTelemetry Collector, Jaeger Collector, custom collectors
- **Backends**: Any system that accepts OpenTelemetry traces

For the complete list of supported exporters, see the [OpenTelemetry Registry](https://opentelemetry.io/ecosystem/registry/).

### Custom Adapter

Implement your own adapter for specific requirements:

```python
from nemoguardrails.tracing.adapters.base import InteractionLogAdapter

class MyCustomAdapter(InteractionLogAdapter):
    name = "MyCustomAdapter"

    def transform(self, interaction_log):
        # your custom logic here
        pass
```

## OpenTelemetry Setup

### Understanding the Architecture

- **NeMo Guardrails**: Uses only the OpenTelemetry API (doesn't configure anything)
- **Your Application**: Configures the OpenTelemetry SDK and exporters

This means you must configure OpenTelemetry in your application code.

### Installation

#### For Tracing Support (API only)

```bash
# minimum requirement for NeMo Guardrails tracing features
pip install nemoguardrails[tracing]
```

This installs only the OpenTelemetry API, which is sufficient if your application already configures OpenTelemetry.

#### For Running Examples and Development

```bash
# includes OpenTelemetry SDK for configuring exporters
pip install nemoguardrails[tracing] opentelemetry-sdk
```

#### For Production Deployments

```bash
# install tracing support
pip install nemoguardrails[tracing]

# install SDK and your preferred exporter
# for OTLP
pip install opentelemetry-sdk opentelemetry-exporter-otlp
# OR for Jaeger
pip install opentelemetry-sdk opentelemetry-exporter-jaeger
# OR for Zipkin
pip install opentelemetry-sdk opentelemetry-exporter-zipkin
```

### Configuration Examples

#### Common Examples

**Console Output** (Development/Testing):

Suitable for development which prints traces to your terminal:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

# configure OpenTelemetry (do this before using NeMo Guardrails)
resource = Resource.create({
    "service.name": "my-guardrails-app",
    "service.version": "1.0.0",
}, schema_url="https://opentelemetry.io/schemas/1.26.0")

tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider)

# use console exporter (prints to terminal)
console_exporter = ConsoleSpanExporter()
span_processor = BatchSpanProcessor(console_exporter)
tracer_provider.add_span_processor(span_processor)

# now configure NeMo Guardrails
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_content(
    config={
        "models": [{"type": "main", "engine": "openai", "model": "gpt-3.5-turbo-instruct"}],
        "tracing": {
            "enabled": True,
            "adapters": [{"name": "OpenTelemetry"}]
        }
    }
)

rails = LLMRails(config)
response = rails.generate(messages=[{"role": "user", "content": "Hello!"}])
```

**OTLP Exporter** (Production-ready):

For production use with observability platforms:

```bash
# install OTLP exporter
pip install opentelemetry-exporter-otlp
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# configure OpenTelemetry
resource = Resource.create({
    "service.name": "my-guardrails-app",
    "service.version": "1.0.0",
}, schema_url="https://opentelemetry.io/schemas/1.26.0")

tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider)

# configure OTLP exporter
otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",  # Your OTLP collector endpoint
    insecure=True
)

span_processor = BatchSpanProcessor(otlp_exporter)
tracer_provider.add_span_processor(span_processor)

# use with NeMo Guardrails (same as console example)
```

> **Note**: These examples show popular configurations, but OpenTelemetry supports many more exporters and backends. You can integrate with any OpenTelemetry-compatible observability platform by installing the appropriate exporter package and configuring it in your application code.

## Additional Integration Examples

These are just a few examples of the many OpenTelemetry integrations available:

### Zipkin Integration

1. Start Zipkin server:

```bash
docker run -d -p 9411:9411 openzipkin/zipkin
```

2. Install Zipkin exporter:

```bash
pip install opentelemetry-exporter-zipkin
```

3. Configure in your application:

```python
from opentelemetry.exporter.zipkin.proto.http import ZipkinExporter

zipkin_exporter = ZipkinExporter(
    endpoint="http://localhost:9411/api/v2/spans",
)
span_processor = BatchSpanProcessor(zipkin_exporter)
tracer_provider.add_span_processor(span_processor)
```

### OpenTelemetry Collector

Create a collector configuration file:

```yaml
# otel-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:

exporters:
  logging:
    loglevel: debug

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [logging]
```

Run the collector:

```bash
docker run -p 4317:4317 -p 4318:4318 \
  -v $(pwd)/otel-config.yaml:/etc/otel-collector-config.yaml \
  otel/opentelemetry-collector:latest \
  --config=/etc/otel-collector-config.yaml
```

## Migration Guide

### From Previous Versions

If you were using the old OpenTelemetry configuration:

**❌ no longer supported:**

```yaml
tracing:
  enabled: true
  adapters:
    - name: OpenTelemetry
      service_name: "my-service"
      exporter: "console"
      resource_attributes:
        env: "production"
```

**✅ supported:**

```python
# configure OpenTelemetry in your application code
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

tracer_provider = TracerProvider()
trace.set_tracer_provider(tracer_provider)

console_exporter = ConsoleSpanExporter()
span_processor = BatchSpanProcessor(console_exporter)
tracer_provider.add_span_processor(span_processor)

config = RailsConfig.from_content(
    config={
        "tracing": {
            "enabled": True,
            "adapters": [{"name": "OpenTelemetry"}]
        }
    }
)
```

### Deprecated Features

#### register_otel_exporter Function

The `register_otel_exporter` function is deprecated and will be removed in version 0.16.0:

```python
#  DEPRECATED - will be removed in 0.16.0
from nemoguardrails.tracing.adapters.opentelemetry import register_otel_exporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

register_otel_exporter("my-otlp", OTLPSpanExporter)
```

Instead, configure exporters directly in your application:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

tracer_provider = TracerProvider()
trace.set_tracer_provider(tracer_provider)

otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4318")
span_processor = BatchSpanProcessor(otlp_exporter)
tracer_provider.add_span_processor(span_processor)
```

### Why the Change?

This change follows OpenTelemetry best practices:

1. **Libraries use only the API**: No configuration conflicts
2. **Applications control observability**: You decide where traces go
3. **Better compatibility**: Works with any OpenTelemetry setup

## Troubleshooting

### Common Issues

**No traces appear:**

- Ensure OpenTelemetry is configured in your application (not just NeMo Guardrails config)
- Check that your exporter is working (try `ConsoleSpanExporter` first)
- Verify tracing is enabled in your config

**Connection errors with OTLP:**

```
WARNING: Transient error StatusCode.UNAVAILABLE encountered while exporting traces to localhost:4317
```

- Make sure your collector/endpoint is running
- Use `ConsoleSpanExporter` for testing without external dependencies

**Import errors:**

```
ImportError: No module named 'opentelemetry'
```

- Install the tracing dependencies: `pip install nemoguardrails[tracing]`
- For exporters: `pip install opentelemetry-exporter-otlp`

**Wrong service name in traces:**

- Configure the `Resource` with `SERVICE_NAME` in your application code
- The old `service_name` parameter is no longer used

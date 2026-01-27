---
title:
  page: Tracing Configuration for NeMo Guardrails
  nav: Tracing
description: Configure FileSystem and OpenTelemetry tracing adapters to monitor guardrails.
topics:
- Configuration
- Observability
tags:
- Tracing
- OpenTelemetry
- Monitoring
- YAML
- Configuration
content:
  type: reference
  difficulty: technical_intermediate
  audience:
  - engineer
  - DevOps Engineer
---

# Tracing Configuration

This section describes how to configure tracing and monitoring in the `config.yml` file.

## Overview

The NeMo Guardrails library includes tracing capabilities to monitor and debug guardrails interactions.
Tracing helps you understand rail activation, LLM call patterns, flow execution, and error conditions.

## The `tracing` Key

Configure tracing in `config.yml`:

```yaml
tracing:
  enabled: true
  adapters:
    - name: FileSystem
      filepath: "./logs/traces.jsonl"
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `enabled` | Enable or disable tracing | `false` |
| `adapters` | List of tracing adapters | `[]` |

## Tracing Adapters

### FileSystem Adapter

Log traces to local JSON files (recommended for development):

```yaml
tracing:
  enabled: true
  adapters:
    - name: FileSystem
      filepath: "./logs/traces.jsonl"
```

| Option | Description |
|--------|-------------|
| `filepath` | Path to the trace output file |

### OpenTelemetry Adapter

Integrate with observability platforms (recommended for production):

```yaml
tracing:
  enabled: true
  adapters:
    - name: OpenTelemetry
```

```{important}
To use OpenTelemetry tracing, install the tracing dependencies:
`pip install nemoguardrails[tracing]`
```

```{note}
OpenTelemetry integration requires configuring the OpenTelemetry SDK in your application code.
NeMo Guardrails follows OpenTelemetry best practices where libraries use only the API and applications configure the SDK.
```

## Adapter Comparison

| Adapter | Use Case | Configuration |
|---------|----------|---------------|
| FileSystem | Development, debugging, simple logging | `filepath: "./logs/traces.jsonl"` |
| OpenTelemetry | Production, monitoring platforms, distributed systems | Requires application-level SDK configuration |

## Multiple Adapters

Configure multiple adapters simultaneously:

```yaml
tracing:
  enabled: true
  adapters:
    - name: FileSystem
      filepath: "./logs/traces.jsonl"
    - name: OpenTelemetry
```

## Trace Information

Traces capture the following information:

| Data | Description |
|------|-------------|
| **Rail Activation** | Which rails get triggered during the conversation |
| **LLM Calls** | LLM invocations, prompts, and responses |
| **Flow Execution** | Colang flow execution paths and timing |
| **Actions** | Custom action invocations and results |
| **Errors** | Error conditions and debugging information |
| **Timing** | Duration of each operation |

## Example Configurations

### Development Configuration

```yaml
tracing:
  enabled: true
  adapters:
    - name: FileSystem
      filepath: "./logs/traces.jsonl"
```

### Production Configuration

```yaml
tracing:
  enabled: true
  adapters:
    - name: OpenTelemetry
```

### Comprehensive Configuration

```yaml
tracing:
  enabled: true
  adapters:
    # Local logs for debugging
    - name: FileSystem
      filepath: "./logs/traces.jsonl"
    # Export to observability platform
    - name: OpenTelemetry
```

## OpenTelemetry Setup

To use OpenTelemetry in production, configure the SDK in your application:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Configure the tracer provider
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Now NeMo Guardrails will export traces to your configured backend
```

## Viewing Traces

### FileSystem Traces

View JSON traces from the filesystem:

```bash
cat ./logs/traces.jsonl | jq .
```

### OpenTelemetry Traces

View traces in your configured observability platform:

- Jaeger
- Zipkin
- Grafana Tempo
- Datadog
- New Relic

## Related Topics

- [Tracing Guide](../../user-guides/tracing/index) - Detailed tracing setup and examples
- [Detailed Logging](../../user-guides/detailed-logging/README) - Additional logging options

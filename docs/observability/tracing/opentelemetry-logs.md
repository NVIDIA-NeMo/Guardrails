---
title:
  page: Exporting Guardrails Logs to OpenTelemetry
  nav: OpenTelemetry Logs
description: Forward Python logs from the NeMo Guardrails library into your OpenTelemetry backend with automatic trace correlation.
topics:
- Observability
- AI Safety
tags:
- OpenTelemetry
- Logs
- Log Correlation
- OTLP
content:
  type: how_to
  difficulty: technical_advanced
  audience:
  - DevOps Engineer
  - AI Engineer
---

# Exporting Guardrails Logs to OpenTelemetry

The NeMo Guardrails library emits operational logs through Python's standard `logging` module. When you have OpenTelemetry tracing configured, you can forward those log records into the same backend as your traces with a few lines of application code. Records emitted inside an active guardrails span automatically carry the span's `trace_id` and `span_id`, so every log line correlates to the request that produced it.

This page covers the setup. For plain Python logging (verbose mode, explain, generation options), see [](../logging/index.md).

## API and SDK Responsibilities

The NeMo Guardrails library follows the OpenTelemetry library-instrumentation pattern:

- **The library depends on the OpenTelemetry API only.** It creates spans, emits log records, and otherwise participates in whatever OTEL pipeline the host application provides.
- **The host application owns the SDK.** Configuring a `TracerProvider`, a `LoggerProvider`, exporters, and attaching handlers to Python's `logging` tree are all the application's responsibility.

This split is deliberate. It lets the NeMo Guardrails library stay decoupled from SDK-version churn, avoids the library injecting itself into a host's observability stack without opt-in, and gives applications full control over where their telemetry is exported.

The three-line recipe below is therefore a user-side setup, not something the library does for you.

## Prerequisites

Before enabling log export, install the OpenTelemetry SDK as described in [](opentelemetry-integration.md#installation). The OpenTelemetry log components live in the same `opentelemetry-sdk` package that powers trace export — no additional installation is required for in-process log forwarding.

For exporting logs to an external backend over OTLP, install the OTLP exporter:

```bash
pip install opentelemetry-exporter-otlp
```

## Minimal Setup: Attach the Logging Handler

Add three lines to your application startup to forward `nemoguardrails` log records into the OpenTelemetry log pipeline:

```python
import logging
from opentelemetry.sdk._logs import LoggingHandler

logging.getLogger("nemoguardrails").addHandler(LoggingHandler())
```

What each line does:

- `logging.getLogger("nemoguardrails")` — selects the logger namespace that catches every log record emitted by the NeMo Guardrails library (all submodules log under this prefix).
- `LoggingHandler()` — an OpenTelemetry-provided `logging.Handler` subclass that converts each Python `LogRecord` into an OTEL log record. Resolves the active `LoggerProvider` at emit time and attaches trace context automatically.
- `.addHandler(...)` — attaches the handler. From this point forward, every record the NeMo Guardrails library emits flows to both the host's existing handlers (console, files, etc.) and the OpenTelemetry pipeline.

This is **additive**: your existing Python logging configuration continues to work unchanged. OpenTelemetry export happens alongside, not instead.

## Full Example: Traces and Logs Together

This program configures a `TracerProvider` and a `LoggerProvider`, both exporting to the console, then runs a guardrails request so you can see correlated spans and log records.

```python
import logging

from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter

from nemoguardrails import LLMRails, RailsConfig

# Application-owned SDK setup
resource = Resource.create({"service.name": "guardrails-log-demo"})

# 1. Traces → console
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(tracer_provider)

# 2. Logs → console
logger_provider = LoggerProvider(resource=resource)
logger_provider.add_log_record_processor(BatchLogRecordProcessor(ConsoleLogExporter()))
set_logger_provider(logger_provider)

# 3. Forward nemoguardrails log records into the OTEL pipeline
logging.getLogger("nemoguardrails").addHandler(LoggingHandler())

# Guardrails configuration
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

response = rails.generate(messages=[{"role": "user", "content": "Hello!"}])
print(f"Response: {response}")
```

Running this script prints both the span tree and the log records to your console. Records emitted while the guardrails request is in flight carry `trace_id` and `span_id` fields that match the enclosing span.

## Exporting to a Backend

The log-record processor in the example above can target any OpenTelemetry log exporter. For an OTLP collector:

```python
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

otlp_log_exporter = OTLPLogExporter(endpoint="http://localhost:4317", insecure=True)
logger_provider.add_log_record_processor(BatchLogRecordProcessor(otlp_log_exporter))
```

The OpenTelemetry Collector then forwards the records to any compatible backend — Loki, Datadog, New Relic, Elastic, and so on. See the [OpenTelemetry Registry](https://opentelemetry.io/ecosystem/registry/) for the list.

## What the Exported Records Contain

Each forwarded `LogRecord` becomes an OTEL log record with the following fields populated automatically:

- **Body** — the formatted log message.
- **Severity** — `severity_text` (`INFO`, `DEBUG`, `ERROR`, etc.) and `severity_number`.
- **Timestamp** — the record's emit time.
- **Trace context** — `trace_id` and `span_id` of the active span when the record was emitted. Zero when no span is active.
- **Code attributes** — `code.file.path`, `code.function.name`, `code.line.number` derived from the Python `LogRecord`.

Log records emitted outside any guardrails request (startup, engine registration, teardown) still flow through, but their `trace_id` / `span_id` are zero because there is no active span.

## Considerations

- **Experimental SDK surface.** The `opentelemetry.sdk._logs` module is still under active development in the OpenTelemetry Python SDK and may change in future releases. The underscore prefix denotes a non-stable API. Pin your `opentelemetry-sdk` version in production and review release notes before upgrading.
- **Privacy.** Guardrails log messages include user inputs and rail decisions. Before exporting to a third-party backend, review whether the records may contain PII and whether your retention/redaction policies cover them.
- **Performance.** At high log volumes or DEBUG level, log export can add measurable overhead. Use `BatchLogRecordProcessor` (as shown) rather than the synchronous `SimpleLogRecordProcessor` in production, and consider filtering at the logger level (`logging.getLogger("nemoguardrails").setLevel(logging.INFO)`) to limit what crosses the bridge.
- **Interaction with `propagate=False`.** If your application calls `nemoguardrails.guardrails.configure_logging()`, that helper sets `propagate=False` on the `nemoguardrails.guardrails` logger to prevent duplicate console output. Records from submodules under `nemoguardrails.guardrails.*` will then not reach the handler attached to `nemoguardrails`. To capture them, attach the handler to `nemoguardrails.guardrails` instead of (or in addition to) `nemoguardrails`.

## Related Resources

- [](opentelemetry-integration.md) — SDK installation and trace export setup.
- [](quick-start.md) — minimal tracing setup with the OpenTelemetry SDK.
- [](../logging/index.md) — Python logging, verbose mode, and the `log` generation option for in-process debugging.

---
title:
  page: Metrics for Guardrails
  nav: Metrics
description: Emit OpenTelemetry metrics from IORails for SLO dashboards, capacity planning, and LLM cost tracking.
topics:
- Observability
- AI Safety
tags:
- Metrics
- OpenTelemetry
- Monitoring
- Prometheus
content:
  type: how_to
  difficulty: technical_intermediate
  audience:
  - engineer
  - DevOps Engineer
  - AI Engineer
---

(metrics)=

# Metrics for Guardrails

Metrics give a low-overhead, aggregate view of guardrails behavior in production.
While tracing answers *"what happened on a particular request?"*, metrics answer *"how is the NeMo Guardrails behaving over the last five minutes?"*.

The IORails engine emits OpenTelemetry metrics inline as requests flow through it.
These metrics are independent of tracing — you can enable either signal alone, or both together, as best fits your observability stack.

With metrics, you can:

- Track request volume, error rate, and latency for SLO dashboards.
- Monitor how many requests are buffered and in-flight.
- Measure downstream LLM token usage and operation latency for cost and performance analysis.
- Alert on rejected requests, blocked requests, or rising error rates.

## Engine Support

| Engine | Metrics |
|--------|---------|
| **IORails** | Supported. All metrics described on this page are emitted by `IORails`. |
| **LLMRails** | Not supported. LLMRails uses post-hoc tracing for observability; see [](../tracing/index.md). |

## Independent of Tracing

Metrics and tracing are configured separately and can be toggled independently.
A common production pattern is **metrics-only**: lightweight aggregate signals without the cost of full trace export.

```yaml
tracing:
  enabled: false

metrics:
  enabled: true
```

The two signals share the same `opentelemetry-api` dependency (installed via `pip install nemoguardrails[tracing]`), but otherwise have separate SDK configuration in your application code: a `TracerProvider` for traces and a `MeterProvider` for metrics.

## Metric Categories

Two families of metrics are emitted.

| Family | Prefix | Purpose |
|--------|--------|---------|
| Request-level | `guardrails.*` | Volume, latency, errors, blocked requests, queue and stream saturation. |
| LLM client-side | `gen_ai.client.*` | Per-LLM-call token usage, operation duration, streaming chunk timing. Follows the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/). |

For the full list of metric names, types, labels, and units, see [](reference.md).

## Important Considerations

- **Library / SDK split.**
  The NeMo Guardrails library depends on the OpenTelemetry **API** only.
  Your application configures the **SDK** — `MeterProvider`, exporters, periodic readers.
  Without a `MeterProvider`, the API hands back a no-op meter and all emissions are silently discarded.
  This is the same library-instrumentation pattern used by the tracing path.
- **Evolving GenAI standards.**
  The [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) are still under active development.
  Metric names, labels, and bucket boundaries may change as the spec matures.
  Pin your `opentelemetry-sdk` version and review release notes before upgrading.
- **Cardinality.**
  Labels on the emitted metrics are deliberately low-cardinality (rail type, error class name, model name, provider name, token type).
  Avoid adding views in your SDK that introduce high-cardinality dimensions like user IDs or request IDs.
- **Performance.**
  The hot-path overhead is bounded — counters and histograms are recorded with simple atomic operations.
  The cost is dominated by SDK-level batching and export, which the host application controls.

## Contents

- [](quick-start.md) — Minimal setup to enable metrics using the OpenTelemetry SDK with console output.
- [](opentelemetry-integration.md) — Production-ready OpenTelemetry SDK configuration with OTLP and Prometheus exporters.
- [](reference.md) — Full reference for every metric: name, instrument type, unit, labels, and emission semantics.
- [](troubleshooting.md) — Common issues and solutions.

```{toctree}
:hidden:

Quick Start <quick-start>
OpenTelemetry <opentelemetry-integration>
Metric Reference <reference>
Troubleshooting <troubleshooting>
```

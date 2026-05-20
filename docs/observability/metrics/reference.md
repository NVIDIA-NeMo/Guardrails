---
title:
  page: Metric Reference
  nav: Metric Reference
description: Reference every metric IORails emits, with instrument types, units, labels, and emission semantics.
topics:
- Observability
- AI Safety
tags:
- Metrics
- OpenTelemetry
- Reference
content:
  type: reference
  difficulty: technical_intermediate
  audience:
  - engineer
  - DevOps Engineer
---

# Metric Reference

This page lists every metric the IORails engine emits when `metrics.enabled: true` and a `MeterProvider` is configured.

Metrics fall into two families:

- **Request-level metrics** (`guardrails.*`) describe IORails request flow: volume, errors, blocks, latency, and saturation of the streaming and non-streaming admission paths.
- **LLM client-side metrics** (`gen_ai.client.*`) describe downstream LLM calls IORails issues.
  These follow the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/) and use the bucket boundaries recommended by that spec.

## Request-Level Metrics

| Metric | Instrument | Unit | Labels | Description |
| --- | --- | --- | --- | --- |
| `guardrails.requests` | Counter | `1` | — | Total IORails requests handled, incremented on entry. |
| `guardrails.requests.errors` | Counter | `1` | `error.type` | Requests that ended in an unhandled error. `error.type` is the exception class name (for example `QueueFull`, `TimeoutError`). |
| `guardrails.requests.blocked` | Counter | `1` | `rail.type` | Requests blocked by an input or output rail. `rail.type` is `input` or `output`. |
| `guardrails.request.duration` | Histogram | `s` | — | End-to-end request duration. For non-streaming requests this includes queue-wait time. |
| `guardrails.requests.active` | UpDownCounter | `1` | — | Requests currently in flight. Covers both streaming and non-streaming. |

### Bucket Boundaries: `guardrails.request.duration`

The duration histogram buckets use seconds:

```text
[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
```

## Saturation Metrics

These metrics expose the internal admission paths so you can detect overload before users encounter errors.

### Non-Streaming Path (Admission Queue)

| Metric | Instrument | Unit | Description |
| --- | --- | --- | --- |
| `guardrails.nonstream.queued` | ObservableGauge | `1` | Requests buffered in the admission queue, not yet picked up by a worker. |
| `guardrails.nonstream.active` | ObservableGauge | `1` | Requests currently executing on a worker. |
| `guardrails.nonstream.rejections` | Counter | `1` | Submissions rejected with `QueueFull` because the admission queue exceeded its depth limit. |

`queued` and `active` are read live at collection time, so dashboards always show the current state.
After `IORails.stop()` is called, both gauges return no observations rather than stale values.

### Streaming Path (Concurrency Semaphore)

| Metric | Instrument | Unit | Description |
| --- | --- | --- | --- |
| `guardrails.stream.active` | UpDownCounter | `1` | In-progress streaming requests holding a semaphore permit. |
| `guardrails.stream.rejections` | Counter | `1` | Streaming requests rejected because the streaming concurrency semaphore was fully occupied. |

### Cross-Checking Saturation Metrics

At any collection instant, the sum of the per-path saturation gauges should approximately equal `guardrails.requests.active`:

```text
guardrails.requests.active ≈ guardrails.nonstream.queued
                           + guardrails.nonstream.active
                           + guardrails.stream.active
```

A persistent drift between the two is a useful integrity check during dashboard development.

### Dual-Counted Rejections

A `QueueFull` rejection on the non-streaming path increments **both**:

- `guardrails.nonstream.rejections` (saturation signal)
- `guardrails.requests.errors{error.type=QueueFull}` (error signal)

This is intentional: dashboards built around either signal alone still reflect the rejection.

## LLM Client-Side Metrics

These metrics are recorded once per downstream LLM call, not once per IORails request, and follow the OpenTelemetry GenAI semantic conventions.

| Metric | Instrument | Unit | Labels | Description |
| --- | --- | --- | --- | --- |
| `gen_ai.client.token.usage` | Histogram | `{token}` | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.token.type` | Number of tokens consumed by an LLM call. Each call records two observations distinguished by the required `gen_ai.token.type` label (`input` or `output`). |
| `gen_ai.client.operation.duration` | Histogram | `s` | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, optionally `error.type` | Wall-clock duration of one LLM call. `error.type` is added as a conditional label only when the call raises. |
| `gen_ai.client.operation.time_to_first_chunk` | Histogram | `s` | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model` | Streaming-only. Time from request issue to the first content-bearing chunk. |
| `gen_ai.client.operation.time_per_output_chunk` | Histogram | `s` | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model` | Streaming-only. Inter-chunk gap; one observation per content-bearing chunk after the first. |

```{note}
`gen_ai.token.type` only takes the values `input` and `output` per spec.
Reasoning and cached tokens are exposed as **span attributes** (`gen_ai.usage.reasoning.output_tokens` and so on), not as additional metric label values.
```

### Bucket Boundaries

Per the OpenTelemetry GenAI spec, durations use powers-of-two boundaries up to ~82 s:

```text
[0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92]
```

Token counts use powers-of-four boundaries up to ~67M tokens:

```text
[1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]
```

Both match the spec exactly so backends auto-render the distributions correctly.

### Streaming vs. Non-Streaming Emission

| Metric | Non-streaming | Streaming |
| --- | :---: | :---: |
| `gen_ai.client.token.usage` | ✓ | ✓ (when the upstream provider returns usage) |
| `gen_ai.client.operation.duration` | ✓ | ✓ |
| `gen_ai.client.operation.time_to_first_chunk` | — | ✓ |
| `gen_ai.client.operation.time_per_output_chunk` | — | ✓ |

For streaming responses, `token.usage` is emitted only when the upstream provider returns a `usage` field. This is common when `stream_options.include_usage=true` is forwarded.
When usage is absent, no observation is recorded; "no observation" is deliberately distinct from "0 tokens".

## Common Label Reference

| Label | Used On | Values | Notes |
| --- | --- | --- | --- |
| `error.type` | `guardrails.requests.errors`, `gen_ai.client.operation.duration` (on error) | Exception class name | For example `QueueFull`, `TimeoutError`, `ValueError`. |
| `rail.type` | `guardrails.requests.blocked` | `input`, `output` | Identifies whether an input or output rail blocked the request. |
| `gen_ai.operation.name` | All `gen_ai.client.*` | For example `chat`, `completion`, `embedding` | OpenTelemetry GenAI operation name. |
| `gen_ai.provider.name` | All `gen_ai.client.*` | For example `openai`, `anthropic` | OpenTelemetry GenAI provider name. |
| `gen_ai.request.model` | All `gen_ai.client.*` | For example `gpt-4o-mini` | The model name passed in the request. |
| `gen_ai.token.type` | `gen_ai.client.token.usage` | `input`, `output` | Required label per spec. |

## Public API Stability

The metric names listed on this page are part of the library's public API, so dashboards and alerts can reference them.
The library tests assert on the raw strings for this reason.
Bucket boundaries follow the OpenTelemetry GenAI spec and can change if the spec changes.

## Related Resources

- [](enable-metrics.md) — Minimal SDK setup with console output.
- [](opentelemetry-integration.md) — Production exporters: OTLP, Prometheus.
- [OpenTelemetry GenAI metrics specification](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/) — Upstream semantic conventions.

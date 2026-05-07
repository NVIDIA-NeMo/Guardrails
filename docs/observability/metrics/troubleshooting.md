---
title:
  page: Troubleshooting
  nav: Troubleshooting
description: Resolve common metrics issues with OpenTelemetry SDK configuration and IORails wiring.
topics:
- Observability
- AI Safety
tags:
- Troubleshooting
- Metrics
- OpenTelemetry
- Debugging
content:
  type: reference
  difficulty: technical_intermediate
  audience:
  - engineer
  - DevOps Engineer
---

# Troubleshooting

| Issue | Solution |
|-------|----------|
| No metrics appear in your backend | Ensure `set_meter_provider(...)` is called **before** `IORails(config)` is constructed; verify `metrics.enabled: true` in the configuration. |
| `MeterProvider` not configured warning | The OpenTelemetry API returns a no-op meter when no provider is set. Configure a `MeterProvider` with at least one `MetricReader`. |
| `UserWarning: Metrics are enabled in config but the opentelemetry-api package is not installed` | Install the dependency: `pip install nemoguardrails[tracing]`. |
| Metrics are emitted but never reach the backend | Verify the exporter target is reachable; test with `ConsoleMetricExporter` first to confirm IORails-side emission, then swap in the production exporter. |
| `LLMRails` produces no metrics | Metrics are emitted only by `IORails`. Switch to `IORails` and use `generate_async` / `stream_async`. |
| Synchronous `IORails.generate()` produces no metrics | Telemetry is disabled for the ephemeral `IORails` constructed by the synchronous `generate()` shim. Use `generate_async` / `stream_async` for production paths. |
| `gen_ai.client.token.usage` missing for streaming requests | The upstream provider did not return a `usage` field in the streamed response. Forward `stream_options={"include_usage": true}` when calling OpenAI-compatible providers, or accept that token usage is not available for that provider. |
| Histogram buckets are wrong in the backend | The library sets bucket-boundary advisories per the OTEL spec. Verify your backend honors the SDK's `explicit_bucket_boundaries_advisory`; some Prometheus exporters override the advisory unless explicitly configured. |
| `guardrails.requests.active` drifts from the sum of saturation gauges | A small steady drift is expected because the gauge reads are not atomic with the counter increments. A persistent large drift indicates an instrumentation bug — file an issue. |
| Wrong `service.name` on metrics | Set the `Resource` with `service.name` when constructing the `MeterProvider`. Use the same `Resource` on the `TracerProvider` to keep traces and metrics correlated. |

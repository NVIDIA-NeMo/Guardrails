---
title:
  page: NVIDIA NeMo Guardrails Library Troubleshooting Guide
  nav: Troubleshooting
description:
  main: Diagnose and resolve common NVIDIA NeMo Guardrails library configuration, runtime, and observability issues.
topics:
- Troubleshooting
- AI Safety
tags:
- Troubleshooting
- Debugging
- Metrics
- OpenTelemetry
content:
  type: reference
  difficulty: technical_intermediate
  audience:
  - engineer
  - DevOps Engineer
  - AI Engineer
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# NeMo Guardrails Library Troubleshooting Guide

This page covers common issues you may encounter when configuring, running, or monitoring the NVIDIA NeMo Guardrails library, along with their resolution steps.

::::{admonition} Get Help
:class: tip

If your issue is not listed here, [open an issue](https://github.com/NVIDIA-NeMo/Guardrails/issues) on GitHub.
::::

## Runtime

### Nested AsyncIO Loop

The NVIDIA NeMo Guardrails library is async-first, so the core functionality is implemented with async functions.
To provide a blocking API, the library must invoke async functions inside synchronous code with `asyncio.run`.
The current Python implementation for `asyncio` does not allow nested event loops.
This issue is being discussed by the Python core team and will likely be supported in the future.
For more information, refer to [GitHub Issue 66435](https://github.com/python/cpython/issues/66435) and [Pull Request 93338](https://github.com/python/cpython/pull/93338).

The NVIDIA NeMo Guardrails library uses [nest_asyncio](https://github.com/erdewit/nest_asyncio) as a workaround.
The patching is applied when the `nemoguardrails` package is loaded the first time.

If the blocking API is not needed, or the `nest_asyncio` patching causes unexpected problems, disable it:

```console
$ export DISABLE_NEST_ASYNCIO=True
```

Then rerun your application.

## Guardrails Metrics

### No Metrics Appear in Your Backend

Call `set_meter_provider(...)` before constructing `IORails(config)`.
Then verify that `metrics.enabled: true` is set in the configuration.

### Metrics Are Silently Missing

When `metrics.enabled: true` but no `MeterProvider` is configured, the OpenTelemetry API returns a no-op meter and silently discards every emission.
The library does not log a warning.

Verify locally with `ConsoleMetricExporter` first.
Then ensure `set_meter_provider(...)` runs before constructing `IORails(config)`.

### Metrics Dependency Is Missing

If you see the following warning, the `opentelemetry-api` package is not installed:

```text
UserWarning: Metrics are enabled in config but the opentelemetry-api package is not installed
```

Install the dependency:

```console
$ pip install nemoguardrails[tracing]
```

### Metrics Are Emitted but Never Reach the Backend

Verify that the exporter target is reachable.
Test with `ConsoleMetricExporter` first to confirm IORails-side emission, then swap in the production exporter.

### `LLMRails` Produces No Metrics

Metrics are emitted only by `IORails`.
Switch to `IORails` and use `generate_async` or `stream_async`.

### Synchronous `IORails.generate()` Produces No Metrics

Telemetry is disabled for the ephemeral `IORails` constructed by the synchronous `generate()` shim.
Use `generate_async` or `stream_async` for production paths.

### `gen_ai.client.token.usage` Is Missing for Streaming Requests

The upstream provider did not return a `usage` field in the streamed response.
Forward `stream_options={"include_usage": true}` when calling OpenAI-compatible providers, or accept that token usage is not available for that provider.

### Histogram Buckets Are Wrong in the Backend

The library sets bucket-boundary advisories per the OpenTelemetry spec.
Verify that your backend honors the SDK's `explicit_bucket_boundaries_advisory`.
Some Prometheus exporters override the advisory unless explicitly configured.

### `guardrails.requests.active` Drifts from the Sum of Saturation Gauges

A small steady drift is expected because the gauge reads are not atomic with the counter increments.
A persistent large drift indicates an instrumentation bug.
[Open an issue](https://github.com/NVIDIA-NeMo/Guardrails/issues).

### Wrong `service.name` on Metrics

Set the `Resource` with `service.name` when constructing the `MeterProvider`.
Use the same `Resource` on the `TracerProvider` to keep traces and metrics correlated.

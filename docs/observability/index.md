---
title:
  page: Observability Overview
  nav: Overview
description: Logging, tracing, and metrics for end-to-end visibility into the behavior of the NVIDIA NeMo Guardrails library.
topics:
- Observability
- AI Safety
tags:
- Observability
- Logging
- Tracing
- Metrics
- OpenTelemetry
content:
  type: concept
  difficulty: technical_intermediate
  audience:
  - engineer
  - DevOps Engineer
  - AI Engineer
---

# Observability Overview

The NVIDIA NeMo Guardrails library exposes three observability signals so you can debug locally during development and monitor behavior in production: logs, traces, and metrics.
Each signal targets a different question and can be enabled independently of the others.

| Signal | Best For | Page |
| --- | --- | --- |
| **Logging** | Debugging a single request with verbose console output, the `explain()` method, and the `log` generation option for structured per-request data. | [](logging/index.md) |
| **Tracing** | Following a request through the rails it activated and the LLM calls it issued, with full OpenTelemetry semantic-convention support. | [](tracing/index.md) |
| **Metrics** | Tracking aggregate behavior, including request volume, latency distributions, error rates, saturation, and per-LLM-call token usage for SLO dashboards and alerting. | [](metrics/index.md) |

Tracing and metrics use the OpenTelemetry library-instrumentation pattern: the NVIDIA NeMo Guardrails library depends on the OpenTelemetry API only, and the host application configures the SDK providers, exporters, and processors.

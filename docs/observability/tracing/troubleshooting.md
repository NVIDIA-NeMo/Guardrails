---
title:
  page: Troubleshooting
  nav: Troubleshooting
description: Resolve common tracing issues with OpenTelemetry SDK configuration and connectivity.
topics:
- Observability
- AI Safety
tags:
- Troubleshooting
- Tracing
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
| No traces appear | Configure OpenTelemetry SDK in application code; verify `tracing.enabled: true` |
| Connection errors | Check collector is running; test with `ConsoleSpanExporter` first |
| Import errors | Install dependencies: `pip install nemoguardrails[tracing]` |
| Wrong service name | Set `Resource` with `service.name` in application code |

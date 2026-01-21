---
title:
  page: "Tracing Guardrails"
  nav: "Tracing"
description: "Enable distributed tracing with OpenTelemetry to monitor guardrails execution and LLM calls."
topics: ["Observability", "AI Safety"]
tags: ["Tracing", "OpenTelemetry", "Monitoring", "Jaeger"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["Developer", "DevOps Engineer", "AI Engineer"]
---

(tracing)=

# Tracing Guardrails

Tracing enhances the observability of guardrails execution. This section explains the configuration process for implementing tracing with the NeMo Guardrails library.

With tracing, you can:

- Track which rails are activated during conversations.
- Monitor LLM calls and their performance.
- Debug flow execution and identify bottlenecks.
- Analyze conversation patterns and errors.

## Span Formats

Starting with the NeMo Guardrails library v0.16.0, the tracing system has transitioned to OpenTelemetry semantic conventions for Generative AI (GenAI), moving away from the legacy span format. This change enhances observability standardization and improves monitoring capabilities for AI workloads.

**Reference Documentation:**

- [OpenTelemetry Semantic Conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) - Overview of GenAI semantic conventions
- [GenAI Events Specification](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/) - Details on capturing user inputs and model outputs
- [GenAI Spans Specification](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) - Span naming and attribute conventions

### Supported Formats

The NeMo Guardrails library supports the following formats.

- **OpenTelemetry** (`opentelemetry`) - **Recommended default format** following OpenTelemetry semantic conventions for GenAI
- **Legacy** (`legacy`) - Previous format with simple metrics dictionary (deprecated)

### Configuration

You can control the span format through the tracing configuration:

```yaml
tracing:
  enabled: true
  span_format: "opentelemetry"  # default
  enable_content_capture: false  # default, see privacy considerations below
```

### Key Differences

The following are the key differences between the supported span formats.

**OpenTelemetry Format:**

- Follows the standardized [semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- Rich-structured attributes and events (e.g., `gen_ai.request.model`, `gen_ai.usage.input_tokens`)
- Enhanced LLM call tracking with provider and model information
- Support for span events and error tracking
- Compatible with OpenTelemetry ecosystem tools

**Legacy Format:**

- Simple metrics dictionary
- Minimal overhead with limited observability
- **Deprecated** - maintained for backward compatibility only

### Important Considerations

**Development Status**: The [OpenTelemetry semantic conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) are currently in development and may undergo changes. Consider the following risks:

- **Evolving Standards**: Conventions may change as they mature, potentially affecting existing implementations
- **Data Privacy**: The `enable_content_capture` option captures user inputs and model outputs, which may include sensitive information (PII). Only enable when necessary and ensure compliance with data protection regulations. See [GenAI Events documentation](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/) for details
- **Performance Impact**: Extensive telemetry collection may impact system performance, especially with large inputs/outputs

### Migration Path

Existing configurations will continue to work. However, it is strongly recommended to migrate to the OpenTelemetry format. Migration steps are:

1. Update your configuration to use `span_format: "opentelemetry"`
2. Review your telemetry backends for compatibility with OpenTelemetry conventions
3. Test thoroughly, as span structures and attribute names have changed
4. Consider privacy implications if enabling content capture

## Contents

- [](quick-start.md) - Minimal setup to enable tracing using the OpenTelemetry SDK
- [](adapter-configurations.md) - Detailed configuration for FileSystem, OpenTelemetry, and Custom adapters
- [](opentelemetry-integration.md) - Production-ready OpenTelemetry setup and ecosystem compatibility
- [](troubleshooting.md) - Common issues and solutions

## Jupyter Notebooks

- [Tracing Guardrails Quickstart](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/docs/configure-rails/colang/colang-1/tutorials/8-tracing/1_tracing_quickstart.ipynb) - A quickstart guide to tracing Guardrails requests in sequential and parallel modes.
- [Tracing Guardrails with Jaeger](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/docs/configure-rails/colang/colang-1/tutorials/8-tracing/2_tracing_with_jaeger.ipynb) - A guide to visualize Guardrails traces in Jaeger.

```{toctree}
:hidden:

Quick Start <quick-start>
Adapters <adapter-configurations>
OpenTelemetry <opentelemetry-integration>
Troubleshooting <troubleshooting>
```

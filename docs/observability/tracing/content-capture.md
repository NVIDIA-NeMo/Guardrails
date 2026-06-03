---
title:
  page: Capturing Message Content
  nav: Content Capture
description: Capture prompts, responses, and rail inputs on guardrails spans for debugging, with privacy controls and OpenTelemetry GenAI formats.
topics:
- Observability
- AI Safety
tags:
- Tracing
- OpenTelemetry
- Content Capture
- Privacy
- GenAI
content:
  type: how_to
  difficulty: technical_intermediate
  audience:
  - engineer
  - DevOps Engineer
  - AI Engineer
---

(content-capture)=

# Capturing Message Content

By default, guardrails spans carry only metadata: durations, token counts, finish reasons, and which rails activated.
The prompts and responses themselves are not recorded.
Content capture is an opt-in feature that adds the actual user inputs, model outputs, and rail inputs to your spans, so you can debug blocked prompts, investigate false positives in safety rails, and see exactly what a model received and returned.

:::{admonition} Experimental Feature
The inline content-capture behavior described on this page is emitted by the opt-in IORails engine.
To enable IORails, set `NEMO_GUARDRAILS_IORAILS_ENGINE=1`.
IORails is an early-release feature, and the captured attribute and event names follow the OpenTelemetry GenAI semantic conventions, which are still under active development and can change.
:::

:::{warning}
Enabling content capture writes user inputs and model outputs to your telemetry backend.
This may include personally identifiable information (PII) and other sensitive data.
Only enable it when necessary, restrict access to the backend that receives the spans, and ensure compliance with your data-protection obligations.
:::

## Enabling Content Capture

Content capture is controlled by the `enable_content_capture` field in the `tracing` section of `config.yml`:

```yaml
tracing:
  enabled: true
  enable_content_capture: true  # default: false
  adapters:
    - name: OpenTelemetry
```

Content is only captured when tracing is also enabled — there is no point recording content onto spans that are never exported.

### Environment Variable Override

The `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` environment variable overrides the config field in **both** directions.
This gives operators a single OpenTelemetry-standard switch to flip capture across all services, regardless of what each deployed `config.yml` says.

| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Result |
|------------------------------------------------------|--------|
| `true` or `1` | Capture is forced **on**, even if `enable_content_capture: false`. |
| `false` or `0` | Capture is forced **off**, even if `enable_content_capture: true`. |
| Unset, empty, or any other value | Falls through to the `enable_content_capture` config field. |

Values are case-insensitive, and surrounding whitespace is ignored.

## Output Format

Captured content is emitted in one of two forms, selected by the `OTEL_SEMCONV_STABILITY_OPT_IN` environment variable.
This variable holds a comma-separated list of opt-in tokens.

| `OTEL_SEMCONV_STABILITY_OPT_IN` | Format | What is emitted |
|---------------------------------|--------|-----------------|
| Contains `gen_ai_latest_experimental` | **JSON span attributes** | Structured, JSON-encoded span attributes following the latest experimental OpenTelemetry GenAI conventions. |
| Unset, or does not contain the token (default) | **Legacy span events** | One span event per message, following the earlier GenAI event conventions. |

### JSON Span Attributes

When `OTEL_SEMCONV_STABILITY_OPT_IN` contains `gen_ai_latest_experimental`, content is recorded as JSON-encoded span attributes:

| Attribute | Contents |
|-----------|----------|
| `gen_ai.input.messages` | The non-system input messages, each as `{"role": ..., "parts": [{"type": "text", "content": ...}]}`. |
| `gen_ai.output.messages` | The assistant output, as a single role-wrapped message. |
| `gen_ai.system_instructions` | The system messages, as a flat list of `{"type": "text", "content": ...}` parts (no role wrapper, per the specification). |

Each attribute is set only when it has content, so a backend can distinguish "no system instructions" from an empty string.

### Legacy Span Events

By default — when `OTEL_SEMCONV_STABILITY_OPT_IN` is unset or does not include the token — content is recorded as span events instead:

| Event | Emitted for |
|-------|-------------|
| `gen_ai.system.message` | Each system message. |
| `gen_ai.user.message` | Each user message. |
| `gen_ai.assistant.message` | Each assistant message in the input. |
| `gen_ai.choice` | The assistant output (the response). |

Roles outside this set (for example, `tool`) are skipped; tool and function-call events are not yet captured.

```{note}
This format selection is independent of the `tracing.span_format` config field.
`span_format` (`opentelemetry` or `legacy`) selects the span structure used by the LLMRails post-hoc tracing adapter, whereas `OTEL_SEMCONV_STABILITY_OPT_IN` selects how IORails encodes captured content on its inline spans.
```

## Where Content Is Captured

When capture is active, content lands on the following spans.

| Span | Kind | Captured content |
|------|------|------------------|
| `guardrails.request` | SERVER | The request input messages, and the final output delivered to the caller, using the `gen_ai.*` attribute or event names above. |
| `gen_ai.*` | CLIENT | The input messages and output of every LLM call — both the main LLM and the per-rail-action LLM calls (for example, content-safety models). |
| `guardrails.rail` | INTERNAL | `guardrails.rail.input` — the JSON-encoded rail input (`{"messages": [...], "bot_response": ...}`). On a rail that blocks, `guardrails.rail.reason` also carries the human-readable block reason. |

Because the request span and the LLM spans share the same `gen_ai.*` attribute and event names, a backend can correlate the outer guardrails request with the inner model calls by name alone.

## Streaming

For streamed responses, output chunks are accumulated and the captured output is written once, at the end of the stream.
The recorded output is exactly what reached the consumer.
If an output rail blocks mid-stream, the captured output reflects the truncated stream plus any injected error response — not text the caller never received.
When nothing is delivered, no output is recorded.

## Engine Support

| Engine | Content capture |
|--------|-----------------|
| **IORails** | Preview support. The full behavior on this page — environment-variable resolution, JSON-attribute or legacy-event format selection, and capture on the request, LLM, and rail spans — is emitted by the opt-in `IORails` engine. |
| **LLMRails** | The `enable_content_capture` field is honored by the LLMRails post-hoc tracing adapter, but content is emitted through that adapter's own span extractors and attribute names. The `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` and `OTEL_SEMCONV_STABILITY_OPT_IN` controls described here apply to IORails. |

## Important Considerations

- **Privacy first.**
  Captured spans contain raw prompts and responses.
  Treat the receiving backend as holding sensitive data, and prefer enabling capture in development or in scoped investigations rather than broadly in production.
- **No truncation.**
  Content is captured in full; there is no size limit or truncation knob.
  Size your exporters and backend accordingly, especially for large inputs or long streamed responses.
- **Evolving GenAI standards.**
  The [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) are still under active development.
  Attribute names, event names, and structures can change as the specification matures.
- **Performance.**
  Extensive telemetry collection can affect performance, especially with large inputs and outputs.
  The hot-path cost is dominated by SDK-level batching and export, which your application controls.

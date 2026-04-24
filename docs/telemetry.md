# Telemetry

NeMo Guardrails collects anonymous telemetry to help the engineering team understand which deployment patterns and safety features are most widely used. The data is transparent and does not contain any user content or identifiers.

A subset of the data, after cleaning and aggregation, may be shared publicly with the community (for example, adoption charts showing which built-in safety features are most used).

> **Not to be confused with tracing.** This page is about anonymous instance-level usage telemetry (a census, collected once at startup and as periodic heartbeats, sent to NVIDIA). It is separate from per-request [tracing](https://docs.nvidia.com/nemo/guardrails/latest/observability/tracing/index.html), which you enable in your guardrails config to emit OpenTelemetry spans to your own observability backend.

## What is collected?

The data is collected once at startup and as periodic heartbeats. It describes the deployment, not individual requests.

| Field | Type | Example | Description |
|---|---|---|---|
| `uuid` | string | `"2b8e9879-80be-42bb-ad3f-81db8ec28e15"` | Random UUID4 per process. Not traceable to any user or machine. |
| `nemoguardrailsVersion` | string | `"0.21.0"` | Installed package version. `"unknown"` if unavailable. |
| `pythonVersion` | string | `"3.13.7"` | Python interpreter version. |
| `platform` | string | `"Linux-5.15.0-x86_64-with-glibc2.35"` | OS and architecture string. |
| `osName` | string | `"Linux"` | Operating system name (`"Darwin"`, `"Linux"`, `"Windows"`). |
| `colangVersion` | string | `"1.0"` | Colang version in use (`"1.0"` or `"2.x"`). |
| `llmProviders` | array of strings | `["nim", "openai"]` | LLM engine names, sorted. Engine identifiers, not model names. |
| `numRailsConfigured` | integer | `4` | Count of configured rail flows across all rail types. |
| `railTypesInUse` | array of strings | `["input", "output"]` | Active rail categories from `input`, `output`, `retrieval`, `tool_input`, `tool_output`, `dialog`. |
| `tracingEnabled` | boolean | `false` | Whether the tracing subsystem is enabled. |
| `context` | string | `"embedded"` | How guardrails was started: `"embedded"` (via `LLMRails`) or `"server"` (via FastAPI). |
| `railsEngine` | string | `"LLMRails"` | Which rails engine is in use: `"LLMRails"` or `"IORails"`. |
| `hasKnowledgeBase` | boolean | `false` | Whether a knowledge base is configured. |
| `streamingConfigured` | boolean | `false` | Whether streaming output is enabled. |
| `builtinFeatures` | array of strings | `["content_safety", "jailbreak_detection"]` | Active built-in library features, sorted (see list below). |
| `numCustomFlows` | integer | `0` | Count of user-defined Colang flows. Never flow names or contents. |
| `timestamp` | number | `1775716074.855979` | Unix timestamp when the event was collected. |
| `event` | string | `"startup"` | Event type: `"startup"` for the initial report, `"heartbeat"` for periodic pings. |

### Possible values for `builtinFeatures`

Each corresponds to a directory under `nemoguardrails/library/`:

`activefence`, `ai_defense`, `autoalign`, `clavata`, `cleanlab`, `content_safety`, `crowdstrike_aidr`, `factchecking`, `fiddler`, `gliner`, `guardrails_ai`, `hallucination`, `injection_detection`, `jailbreak_detection`, `llama_guard`, `pangea`, `patronusai`, `policyai`, `prompt_security`, `regex`, `self_check`, `sensitive_data_detection`, `topic_safety`, `trend_micro`.

## What is NOT collected

- Model names, model paths, or model parameters
- API keys, endpoints, URLs, or credentials
- Rail definitions, Colang source code, or YAML configuration contents
- Prompts, completions, or any user messages
- Token counts, request latency, or per-request metrics
- File paths, usernames, or IP addresses

The schema is designed to be structurally safe. Nothing in the payload is sensitive even if the entire dataset were made public.

## Sample payloads

**Startup event:**

```json
{
  "uuid": "2b8e9879-80be-42bb-ad3f-81db8ec28e15",
  "nemoguardrailsVersion": "0.21.0",
  "pythonVersion": "3.13.7",
  "platform": "Linux-5.15.0-x86_64-with-glibc2.35",
  "osName": "Linux",
  "colangVersion": "1.0",
  "llmProviders": ["nim"],
  "numRailsConfigured": 4,
  "railTypesInUse": ["input", "output"],
  "tracingEnabled": false,
  "context": "embedded",
  "railsEngine": "LLMRails",
  "hasKnowledgeBase": false,
  "streamingConfigured": false,
  "builtinFeatures": ["content_safety", "jailbreak_detection", "topic_safety"],
  "numCustomFlows": 0,
  "timestamp": 1775716074.855979,
  "event": "startup"
}
```

**Heartbeat event (every 10 minutes):**

```json
{
  "uuid": "2b8e9879-80be-42bb-ad3f-81db8ec28e15",
  "timestamp": 1775716674.123456,
  "event": "heartbeat"
}
```

Each event is wrapped in the shared NVIDIA telemetry envelope (protocol v1.6) with `nemoSource: "guardrails"` before it is transmitted.

## Inspecting what is sent

Every outgoing payload is written to a local audit file before it is sent over the network:

```bash
cat ~/.config/nemoguardrails/usage_stats.json
```

The file is JSON lines format (one event per line), bounded at 10 MB with automatic rotation.

## Opting out

Any one of the following disables telemetry:

```bash
# Product-specific:
export NEMO_GUARDRAILS_NO_USAGE_STATS=1

# Industry standard:
export DO_NOT_TRACK=1

# File-based:
mkdir -p ~/.config/nemoguardrails && touch ~/.config/nemoguardrails/do_not_track
```

When any opt-out is active, no daemon thread is spawned, no audit file is written, and no network request is made.

## Schema and source code

The authoritative machine-readable schema lives at [`schemas/telemetry.json`](../schemas/telemetry.json). It is auto-generated from the `GuardrailsUsageEvent` Pydantic class in [`nemoguardrails/telemetry.py`](../nemoguardrails/telemetry.py) via `scripts/generate_telemetry_schema.py`.

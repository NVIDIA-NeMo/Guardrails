# Telemetry

NeMo Guardrails collects anonymous telemetry to help the engineering team understand which deployment patterns and safety features are most widely used. The event payload is transparent and does not contain user content or direct user identifiers such as usernames, API keys, or IP addresses.

A subset of the data, after cleaning and aggregation, may be shared publicly with the community (for example, adoption charts showing which built-in safety features are most used).

> **Not to be confused with tracing.** This page is about anonymous usage telemetry (emitted at each `LLMRails` / `Guardrails` instantiation and as periodic heartbeats, sent to NVIDIA). It is separate from per-request [tracing](https://docs.nvidia.com/nemo/guardrails/latest/observability/tracing/index.html), which you enable in your guardrails config to emit OpenTelemetry spans to your own observability backend.

## What is collected?

The data is collected when `LLMRails` / `Guardrails` is constructed and as periodic heartbeats. It describes the deployment, not individual requests.

In this context, a **session is the lifetime of a single Python process** running NeMo Guardrails: the session ID is generated in memory when telemetry starts and is not stored for reuse across process restarts. The session ID is an optional non-sensitive prefix from `NEMO_SESSION_PREFIX` plus a random UUID4. The same session ID is included in local audit records and transmitted telemetry events so startup and heartbeat events from one process can be correlated. Two runs of guardrails by the same user produce two unrelated session IDs.

| Field | Type | Example | Description |
|---|---|---|---|
| `sessionId` | string | `"smoke-run-2b8e9879-80be-42bb-ad3f-81db8ec28e15"` | Session ID. Optional non-sensitive prefix plus a random UUID4 generated in memory when telemetry starts. Shared by all events from the same process and included in audit records and transmitted events, but not stored for reuse across restarts. |
| `nemoguardrailsVersion` | string | `"0.21.0"` | Installed package version. `"unknown"` if unavailable. |
| `pythonVersion` | string | `"3.13.7"` | Python interpreter version. |
| `platform` | string | `"Linux-5.15.0-x86_64-with-glibc2.35"` | OS and architecture string. |
| `osName` | string | `"Linux"` | Operating system name (`"Darwin"`, `"Linux"`, `"Windows"`). |
| `colangVersion` | string | `"1.0"` | Colang version in use (`"1.0"` or `"2.x"`). |
| `llmProviders` | array of strings | `["nim", "openai"]` | LLM engine names, sorted. Engine identifiers, not model names. |
| `numRailsConfigured` | integer | `4` | Count of configured rail flows for input, output, retrieval, tool input, and tool output rails. Dialog usage is represented in `railTypesInUse` but not counted as a flow. |
| `railTypesInUse` | array of strings | `["input", "output"]` | Active rail categories from `input`, `output`, `retrieval`, `tool_input`, `tool_output`, `dialog`. |
| `tracingEnabled` | boolean | `false` | Whether the tracing subsystem is enabled. |
| `deploymentType` | string | `"library"` | How guardrails was deployed: `"library"` (direct `LLMRails` use), `"api"` (FastAPI server), `"cli"` (interactive `nemoguardrails chat`). |
| `nemoSource` | string | `"guardrails"` | The NeMo product that produced the event. Always `"guardrails"` for this event type. Mirrors the shared `NemoSourceEnum` from the nemo-telemetry repo. |
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

The schema is designed to avoid user content and direct user identifiers. Review the audit file to confirm the exact event fields emitted by your local configuration.


## Sample payloads

**Startup event:**

```json
{
  "sessionId": "2b8e9879-80be-42bb-ad3f-81db8ec28e15",
  "nemoguardrailsVersion": "0.21.0",
  "pythonVersion": "3.13.7",
  "platform": "Linux-5.15.0-x86_64-with-glibc2.35",
  "osName": "Linux",
  "colangVersion": "1.0",
  "llmProviders": ["nim"],
  "numRailsConfigured": 4,
  "railTypesInUse": ["input", "output"],
  "tracingEnabled": false,
  "deploymentType": "library",
  "nemoSource": "guardrails",
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
  "sessionId": "2b8e9879-80be-42bb-ad3f-81db8ec28e15",
  "nemoSource": "guardrails",
  "timestamp": 1775716674.123456,
  "event": "heartbeat"
}
```

Each event is wrapped in the shared NVIDIA telemetry envelope (protocol v1.6) with `nemoSource: "guardrails"` before it is transmitted.

## Inspecting what is sent

Telemetry attempts to write each outgoing event payload to a local audit file before it is sent over the network:

```bash
cat ~/.config/nemoguardrails/usage_stats.json
```

The file is JSON lines format (one event per line), bounded at 10 MB with automatic rotation. It stores the inner event payload, not the full NVIDIA telemetry envelope. Audit writes are best-effort; if local audit writing fails, the telemetry send still proceeds.

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

Set the opt-out before NeMo Guardrails starts. Changing environment variables or creating `do_not_track` after telemetry has started does not stop an already-running heartbeat thread.

### Automatic suppression in test and CI environments

Telemetry is also automatically disabled, with no configuration required, when either of the following environment variables is set:

- `CI` (truthy: `1` or `true`). Set by GitHub Actions, GitLab CI, CircleCI, Travis, Buildkite, and most other CI runners. Honoring this is the same convention used by Homebrew, npm, conda, and others.
- `PYTEST_CURRENT_TEST`. Set by pytest while a test is running. Suppresses telemetry from your test suite even outside CI.

The intent is that adoption metrics reflect real deployments only, not synthetic test or CI traffic. If you genuinely want telemetry from a CI run (rare), unset `CI` for that step.

## Schema and source code

The Python source for the event lives in [`nemoguardrails/telemetry.py`](../nemoguardrails/telemetry.py). A vendored snapshot of the wire-format schema is at [`schemas/anonymous_events.snapshot.json`](../schemas/anonymous_events.snapshot.json) and is used by the conformance test in [`tests/telemetry/test_telemetry.py`](../tests/telemetry/test_telemetry.py) to validate emitted payloads against the canonical contract.

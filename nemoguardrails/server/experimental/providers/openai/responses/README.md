# OpenAI Responses boundary

This experimental integration guards foreground `POST /v1/responses` in
buffered and SSE modes. Typed Python declarations are authoritative. The
[exported contract](../../../contracts/openai/_generated/responses.guard.yaml)
documents the actual bound policy; runtime construction never reads it.

## Supported content

| Direction | Guarded subject | Restrictions |
| --- | --- | --- |
| Request | One nonempty `input` string | No input-item arrays, instructions, prompt templates, conversation references, tools, or background execution. |
| Buffered response | One assistant message's single nonempty `output_text.text` | Reasoning items may occur before or after it, but may not contain plaintext reasoning or summaries. |
| Streaming response | `response.output_text.delta.delta` | Reviewed text snapshots must equal the accumulated deltas; only one assistant target is supported. |

The `single_text.v1` and `single_text_delta.v1` profiles are framework
capabilities, not user-selected rail configurations. Replacement is unsupported
for Responses in both directions. A replacement decision fails explicitly.

Request `stream` is a strict boolean, defaulting to false; this boundary rejects
null rather than coercing it. `background` permits only the boolean false.
Disabled fields accept omission or null, not an empty container or a value such
as `tool_choice: none`. `top_logprobs` permits omission, null, or integer zero.
Response and event logprobs and annotations must be absent, null, or empty:
token alternatives and citation text are not separately inspected.

Reviewed provider metadata, including encrypted reasoning, usage, IDs, and
echoed controls, remains opaque where declared. Opaque does not mean safe or
inspected. Unknown root fields fail closed. Only nested objects explicitly
marked configurable can accept unknown members under a trusted `ALLOW` policy;
the stream classifier always uses the closed default. Case aliases cannot
override reviewed policy.

## Declaration and runtime responsibilities

- `request_projection.py` and `response_projection.py` classify fields.
- The binding modules derive coverage and text locations from those declarations.
- `ObjectPolicy.variant` declares the required discriminator and selected-array
  cardinality. Buffered output requires exactly one assistant message; lifecycle
  snapshots allow at most one, including initially empty output.
- `stream_projection.py` describes the reviewed event bodies.
- `stream_classifier.py` declares event selection, roles, and missing-text
  handling. Text paths and shape coverage are derived rather than handwritten.
- `stream_hooks.py` owns per-request protocol state and native error framing.
- `endpoint.py` composes the payload bindings and explicit `StreamBinding`.

The original payload and SSE bytes are forwarded, not reconstructed from
Pydantic models. Projection defaults are never inserted into provider traffic.

## Streaming behavior and limits

Creation, progress, queue, message, content-part, text-done, and terminal
response events are text snapshots. Initially missing message text is treated
as an empty snapshot only where the event rule explicitly permits it. Reasoning
item events are validated metadata; plaintext reasoning events and unknown
content-producing families fail closed.

A foreground stream must end with `response.completed`, `response.failed`,
`response.incomplete`, or `error`. An optional `[DONE]` may follow native
completion once; it cannot replace it. Data-less keepalives are not terminal.
Post-terminal payload and duplicate completion are rejected.

When supplied, sequence numbers must increase and text item IDs and output
indices must remain consistent, including with lifecycle snapshots. Content
index is restricted to zero. Unrelated metadata is not required, so this is
not a full validator of every provider-required field or lifecycle transition.
SSE event names, when present, must match the JSON `type` discriminator.

The existing kernel bounds event size, pending bytes, and snapshot history.
Snapshots whose history can no longer be verified fail closed. Checked windows
may already have reached the client before a later protocol failure; the proxy
cannot retract them. Blocking withholds pending text and snapshots. Proxy stream
failures use a native `event: error` body with `type: error`, not a Chat `[DONE]`
frame. Provider terminal failures retain their original bytes after any pending
assistant text has been checked.

## Application composition

Use `create_openai_integration` to select which built-in operations to guard.
Its `operations` argument accepts a nonempty sequence of unique typed names:
`"chat_completions"`, `"responses"`, or both. The fixed default is
`("chat_completions", "responses")`, not all operations supported by future
versions. Unknown names, duplicate names, empty selections, and bare strings
are rejected during construction.

`create_openai_chat_integration` remains a Chat-only wrapper, including its
existing explicit dialect endpoint customization. The broader factory selects
strict OpenAI boundaries; it does not infer a dialect from an upstream URL.

```python
from pathlib import Path

from nemoguardrails.server.experimental._application import create_proxy_application
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_integration
from nemoguardrails.server.experimental.rails import ConfigEntry

app = create_proxy_application(
    providers=(
        create_openai_integration(
            base_url="https://api.openai.com/v1",
            operations=("chat_completions", "responses"),
        ),
    ),
    rails_configs=(ConfigEntry("policy", Path("/trusted/path/to/rails")),),
)
```

This uses the existing application assembly and trusted Rails configuration
directory, not a new launcher or configuration format. Server dependencies use
the project's `server` extra. Provider generation is forwarded directly over
HTTP, not delegated to a LangChain model. The caller supplies upstream
authorization; any configured detectors have their own configuration and
credential requirements.

For Responses-only guarding, pass `operations=("responses",)`. This selects
guard bindings, not a route allowlist: Chat Completions and other unselected
routes retain the application's passthrough behavior without inspection.
Disabling those routes is a separate deployment policy.

Deployment authentication and socket-level acceptance
belong to the application/example work, not these declarations.

## Contract export and verification

```bash
uv run --locked python -m nemoguardrails.server.experimental.provider.contract_export \
  nemoguardrails.server.experimental.providers.openai.responses.endpoint:RESPONSES_ENDPOINT \
  --output nemoguardrails/server/experimental/contracts/openai/_generated/responses.guard.yaml
```

Use `--check` instead of `--output` to detect drift without writing. Only import
trusted endpoint references. The export includes event rules and array-variant
selection, but does not serialize hook state, lifecycle behavior, or transport
implementation. Generic JSON Schema validators ignore guard annotations such
as selected-variant cardinality and case-alias rejection; policy-aware checks
are needed for those rules. Python strict integer checks are also stronger than
JSON Schema's mathematical integer semantics.

The [OpenAI source pin](../../../contracts/openai/source.yaml) remains the review
baseline. The old Responses implementation provided characterization cases,
not authoritative generated models. The
[official streaming guide](https://developers.openai.com/api/docs/guides/streaming-responses)
describes semantic SSE events; this integration intentionally supports only the
reviewed subset above. Tests use synthetic API-shaped payloads, not recordings
or a claim of compatibility with every OpenAI-compatible server.

Focused tests cover projections, export parity, protocol behavior, fragmented
SSE, blocked output, and application composition with real offline regex Rails
and HTTPX ownership. Healthy cleanup is exercised, but the separate core/shared
HTTP interrupted-cleanup defects remain prerequisites for complete lifecycle
acceptance. This feature does not fix or suppress those regressions.

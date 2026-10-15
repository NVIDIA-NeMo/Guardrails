# Chat Completions projection policy

The handwritten request and response projection models are the source of truth
for the buffered field policy. Types express accepted values, annotations express
guarded/constrained/disabled/opaque policy, and assignments express defaults:

```python
n: Annotated[Literal[1], BeforeValidator(_require_int), constrained(reason="core_capability.single_text_target")] = 1
stream: Annotated[StrictBool | None, constrained()] = False
audio: Annotated[None, disabled("core_capability.audio_content")] = None
```

The shared [projection policy module](../../../provider/projection_policy.py)
provides the declaration helpers, runtime metadata derivation, and schema export.
`ObjectPolicy` records object-level provider schema names, reviewed opaque fields, and
unknown-field policy. It is not a deployment profile selector. The capability
profile remains `single_text.v1`; the exported document format remains experimental
`1.0.0-alpha.1`.

`ObjectPolicy` controls object closure. `PolicyModel` requires Pydantic
`extra="allow"` internally so its validator can inspect every unreviewed member;
overriding it with `ignore` or `forbid` is rejected at class declaration.

The request/response binding modules retain their existing runtime classes.
The provider revision pin lives once in
[`providers/openai/source.py`](../source.py), mirroring `source.yaml`. Their coverage inventories and exact text locations
are derived from the model annotations. Extraction supports one required item
at each traversed array boundary; unsupported or ambiguous boundaries fail.
No YAML is loaded to construct these bindings.
Text locations are derived directly from typed fields, not from exported schemas.
An unrelated constrained union therefore does not require contract-format support
just to construct a runtime binding.

`export_payload_schema` exports the declared schema and policy for readers.
It does not prove upstream provider compatibility or serialize arbitrary Python
validators.
Streaming classification, stateful hooks, and endpoint construction are outside
this buffered projection layer.

Buffered validation uses the same field policy: omitted disabled fields
default to null, explicit non-null disabled values fail, `stream` is a strict
boolean or null, and response annotations accept only null or an empty list. Only
`stream: true` selects streaming; false, null, or omission selects buffered output.
The original request bytes are forwarded without normalizing null to false.
`n` accepts only the
integer `1`; booleans, strings, and floats are rejected. Generic JSON Schema
and Python validation are not claimed to be interchangeable. Canonical exports
are closed by default and explicitly declare configurable unknown-field handling
and Unicode case-alias rejection. Contract-aware tools must implement those
annotations when trusted configuration selects `ALLOW`.

The HTTP route publishes the same object-policy rules in its OpenAPI request
schema: reviewed opaque members remain available, and unreviewed members are
rejected by default. This schema still does not encode arbitrary Python validators.

The [contract guide](../../../contracts/README.md) defines the document format;
the [OpenAI boundary summary](../../../contracts/openai/README.md) records its
scope and provider provenance. Neither is loaded by the models or bindings.

## Streaming declarations

[stream_projection.py](stream_projection.py) uses the same field annotations and
`ObjectPolicy` for the experimental `single_text_delta.v1` boundary. A chunk
accepts zero or one choice with a strict integer index of zero, and `logprobs`
must be null. The root, choice, and delta reject unreviewed members; reviewed
root metadata such as `id`, `model`, and `usage` passes through unchanged and
is not inspected.
Missing or null content is metadata, while empty string content is still guarded
text. Error messages are constrained fields, not guarded assistant output. The
error envelope rejects unreviewed members, and its error object rejects them
unless trusted configuration allows unknown fields; the stream classifier
always validates with the closed default.

[stream_classifier.py](stream_classifier.py) owns explicit `StreamEventRule`
declarations for content and provider errors, along with the opaque `[DONE]`
sentinel. Field coverage comes from the models; shape coverage comes from event
rules, missing-text fallbacks, and transport declarations.
Data-less SSE events have a distinct `keepalive` shape; they are not
terminal `[DONE]` events. Numeric bounds and strictness use ordinary Pydantic
constraints, independent of the inspection document's supported vocabulary.
Unsupported document representations fail at export, not runtime construction.
The shared
[stream policy helpers](../../../provider/stream_policy.py) derive a text path
from typed fields, without schema generation, when there is one subject and
arrays are bounded to at most one item or declare a reviewed, bounded
`ObjectPolicy.variant` selection. Chat uses the simple bounded-array path;
Responses uses heterogeneous variants. A null container or null text on that
path is missing text. Unannotated unions remain unsupported by typed derivation.
Buffered extraction still requires exactly one selected item; the streaming
helper does not relax it.

The typed classifier builder checks the text path, assistant subject, event roles,
and coverage against these declarations before runtime use. A mismatched path
cannot silently turn guarded text into a metadata-only event. The lower-level
classifier resolves text from the original JSON before validation and rejects a
projection that changes that text, including through container transformations.
Bytes are relayed unchanged only after their original guarded text is checked.
The lower-level classifier API remains available for handwritten bindings; those do not
automatically satisfy typed-policy validation or contract export.

[stream_hooks.py](stream_hooks.py) remains handwritten. It enforces terminal
ordering, distinguishes real `[DONE]` events from keepalives, accepts provider
errors as terminal outcomes, and frames native errors. These checks require
history and do not belong in field annotations or a stateless classifier.

The projections and classifier can be used independently of endpoint wiring.
The endpoint owns one `StreamBinding`: the classifier and a hook factory.
Runtime creates fresh hooks per request; endpoint construction and contract
export never invoke the factory. The router selects buffered or streaming dispatch
after the shared request projection and input check. Streaming requests require
an injected streaming dispatcher; without one, they fail before dispatch.

## Contract export

The shared [contract exporter](../../../provider/contract_export.py) takes the
runtime endpoint, including its declared document identity. No per-operation exporter is needed.

Before export, it checks coverage, text bindings, that replacement blockers name
reviewed fields, and any declared stream selector against the field policy. Stream rules are checked against
typed text paths and exact shape coverage. Custom extraction, mismatched bindings,
and unsupported event selectors or fallbacks are rejected rather than silently
omitted. A handwritten classifier can still run through `StreamBinding`, but the
exporter requires the standard declarative classifier.
Custom stream validators and custom response-mode properties are runtime-only,
not exported. This includes properties with no selector field. An exported
selector must be strict boolean (optionally nullable), with no default factory
or custom selector validation. Unsupported schema keywords fail at export.

## Streaming ownership and limits

Streaming dispatch owns the upstream resource until it returns
`StreamingHttpResponse`. A dispatcher using an async generator must provide
`close=upstream_response.aclose` (or an equivalent resource-owner callback),
not just the generator's `aclose`. Cancellation can permanently interrupt a
generator's awaited `finally`. An ordinary self-owning iterator may implement
idempotent `aclose` directly. If dispatch or response construction fails before
handoff, the dispatcher must close the resource itself.

The HTTP response owns cleanup afterward, including disconnect before iteration,
source or checker failure, and normal EOF. Iterator cleanup and resource closing
use the same shielded, five-second-per-attempt helper. Closing can be retried;
callbacks must be idempotent and cooperate with cancellation. Timeouts and
ordinary close failures are logged without replacing the primary stream outcome.
This cannot forcibly close uncooperative code or guarantee resources were freed.

With output inspection disabled, bytes are relayed without event classification
or provider lifecycle checks. Inspected streams check windows of nonempty text
deltas: `chunk_size` and `context_size` are counts, not tokens or bytes. Previously
released bytes cannot be retracted. Pending-byte limits do not include retained
context and are not a complete memory budget. The final pending events wait for
completion validation; upstream idle/read deadlines belong to transport policy.

## Exporting the bound contract

```python
from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT

contract = export_guard_contract(CHAT_COMPLETIONS_ENDPOINT)
```

Run the shared CLI from the repository root:

```bash
uv run --locked python -m nemoguardrails.server.experimental.provider.contract_export \
  nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint:CHAT_COMPLETIONS_ENDPOINT \
  --output nemoguardrails/server/experimental/contracts/openai/_generated/chat-completions.guard.yaml
```

Use `--check` instead of `--output` with the same path to detect drift. Without
either option the document is written to stdout. The command validates against
the existing contract format before writing. The endpoint owns `provider_operation_id` and optional
`contract_name`; callers cannot override that identity. `operation_name` separately
identifies the runtime operation.

The endpoint argument imports trusted local Python code. Do not obtain it from
requests or untrusted documents. The CLI does not scan providers or load runtime
configuration from YAML. Its YAML/JSON Schema dependencies are needed only for
serialization and format validation, not endpoint construction.

The [exported contract](../../../contracts/openai/_generated/chat-completions.guard.yaml)
gets its field policy from the Python models and endpoint labels, route, and
error codes from `CHAT_COMPLETIONS_ENDPOINT`. Do not edit the artifact by hand.
Its stream section describes the same classifier bound at runtime: event payload
policy, roles, predicates, missing-text handling, sentinel, and non-data behavior.
An endpoint without a stream binding exports only its buffered boundary.
Handwritten hooks are not serialized: ordering, completion checks, and native
error framing remain in Python. The request model recognizes the `stream` flag,
and the router supports both response modes when their dispatchers are supplied.
Replacement eligibility is declared by the models, while applying replacement
outcomes remains unsupported by this integration.
Exporting the policy does not enable runtime features.

A successful response whose assistant content is empty or null is relayed
unchanged without output checks, but only after the full response passes the
closed projection. Null content alongside refusal, reasoning, tool, or citation
content is still rejected.

Guarded requests require exactly one `Content-Type: application/json` header.
Inspected successful buffered responses require the same; inspected streams
require exactly one `Content-Type: text/event-stream` header. Media types are
case-insensitive and may have parameters. Duplicate headers are rejected even
when their values agree, rather than selecting a first or last value.

When output checks run, guarded requests ask the provider for
`accept-encoding: identity`, replacing any client value, because an encoded
response cannot be inspected. A successful response that is still encoded is
rejected, not relayed. Without output checks, the client's own preference is
forwarded and the response is relayed unchanged.

The shared exporter describes payload policy, declared stream semantics, and
endpoint labels, not arbitrary Python validation or transport behavior:
header/query API-revision bindings and alternate route ownership are not
serialized. JSON Schema also cannot distinguish the integer `0` from `0.0`;
the stream's Python validator requires an actual integer choice index. Export is
an inspection document, not proof of complete runtime value equivalence.

Nullable annotations are exported as disjoint array/null `oneOf` branches; the
array branch has `maxItems: 0` because citation content is not inspected.
No new contract format version, provider download, or runtime YAML loading is
introduced.

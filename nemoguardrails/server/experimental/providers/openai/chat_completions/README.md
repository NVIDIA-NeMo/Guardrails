# Chat Completions buffered policy

The handwritten request and response projection models are the source of truth
for the buffered field policy. Types express accepted values, annotations express
guarded/constrained/disabled/opaque policy, and assignments express defaults:

```python
n: Annotated[Literal[1], BeforeValidator(_require_int), constrained(reason="core_capability.single_text_target")] = 1
stream: Annotated[StrictBool, constrained()] = False
audio: Annotated[None, disabled("core_capability.audio_content")] = None
```

The shared [projection policy module](../../../provider/projection_policy.py)
provides the declaration helpers, runtime metadata derivation, and schema export.
`ObjectPolicy` records object-level provider schema names, reviewed opaque fields, and
unknown-field policy. It is not a deployment profile selector. The capability
profile remains `single_text.v1`; the exported document format remains experimental
`1.0.0-alpha.1`.

The request/response binding modules retain their existing runtime classes.
The provider revision pin lives once in
[`providers/openai/source.py`](../source.py), mirroring `source.yaml`. Their coverage inventories and exact text locations
are derived from the model annotations. Extraction supports one required item
at each traversed array boundary; unsupported or ambiguous boundaries fail.
No YAML is loaded to construct these bindings.

`export_payload_schema` exports the declared schema and policy for readers.
It does not prove upstream provider compatibility or serialize arbitrary Python
validators.
Streaming classification, stateful hooks, and endpoint construction are outside
this buffered projection layer.

Existing validation behavior is preserved: omitted disabled fields default to
null, explicit non-null disabled values fail, `stream` is a strict boolean,
and response annotations accept only null or an empty list. `n` accepts only the
integer `1`; booleans, strings, and floats are rejected. Generic JSON Schema
and Python validation are not claimed to be interchangeable. Canonical exports
are closed by default and explicitly declare configurable unknown-field handling
and Unicode case-alias rejection. Contract-aware tools must implement those
annotations when trusted configuration selects `ALLOW`.

The [contract guide](../../../contracts/README.md) defines the document format;
the [OpenAI boundary summary](../../../contracts/openai/README.md) records its
scope and provider provenance. Neither is loaded by the models or bindings.

## Buffered contract export

The shared [contract exporter](../../../provider/contract_export.py) takes the
runtime endpoint, including its declared document identity. No per-operation exporter is needed:

```python
from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT

contract = export_guard_contract(CHAT_COMPLETIONS_ENDPOINT)
```

Run the shared CLI from the repository root:

```bash
uv run --locked python -m nemoguardrails.server.experimental.provider.contract_export \
  nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint:CHAT_COMPLETIONS_ENDPOINT \
  --output nemoguardrails/server/experimental/contracts/openai/_generated/chat-completions.buffered.guard.yaml
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

The [exported buffered contract](../../../contracts/openai/_generated/chat-completions.buffered.guard.yaml)
gets its field policy from the Python models and endpoint labels, route, and
error codes from `CHAT_COMPLETIONS_ENDPOINT`. Do not edit the artifact by hand.
This is a buffered-only export: there is no stream section or streaming hook.
The request model recognizes the `stream` flag, but this integration still
rejects streaming requests before dispatch. Replacement eligibility is declared
by the models, while applying replacement outcomes remains unsupported by this
integration. Exporting the policy does not enable either runtime feature.

A successful response whose assistant content is empty or null is relayed
unchanged without output checks, but only after the full response passes the
closed projection. Null content alongside refusal, reasoning, tool, or citation
content is still rejected.

When output checks run, guarded requests ask the provider for
`accept-encoding: identity`, replacing any client value, because an encoded
response cannot be inspected. A successful response that is still encoded is
rejected, not relayed. Without output checks, the client's own preference is
forwarded and the response is relayed unchanged.

The shared exporter describes buffered payload policy and endpoint labels, not
arbitrary transport behavior: header/query API-revision bindings and alternate
route ownership are not serialized.

Nullable annotations are exported as disjoint array/null `oneOf` branches; the
array branch has `maxItems: 0` because citation content is not inspected.
No new contract format version, provider download, or runtime YAML loading is
introduced.

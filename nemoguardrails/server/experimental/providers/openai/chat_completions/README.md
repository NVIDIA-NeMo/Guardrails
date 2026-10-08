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
and Python validation are not claimed to be interchangeable.

The [contract guide](../../../contracts/README.md) defines the document format;
the [OpenAI boundary summary](../../../contracts/openai/README.md) records its
scope and provider provenance. Neither is loaded by the models or bindings.

## Buffered contract export

Run the exporter from the repository root:

```bash
poetry run python -m nemoguardrails.server.experimental.providers.openai.chat_completions.contract \
  --output nemoguardrails/server/experimental/contracts/openai/_generated/chat-completions.buffered.guard.yaml
```

Use `--check` instead of `--output` with the same path to detect drift. Without
either option the document is written to stdout. The command validates against
the existing contract format before writing.

The [exported buffered contract](../../../contracts/openai/_generated/chat-completions.buffered.guard.yaml)
gets its field policy from the Python models and endpoint labels, route, and
error codes from `CHAT_COMPLETIONS_ENDPOINT`. Do not edit the artifact by hand.
This is a buffered-only export: there is no stream section or streaming hook.
The request model recognizes the `stream` flag, but this integration still
rejects streaming requests before dispatch. Replacement eligibility is declared
by the models, while applying replacement outcomes remains unsupported by this
integration. Exporting the policy does not enable either runtime feature.

Nullable annotations are exported as disjoint array/null `oneOf` branches.
No new contract format version, provider download, or runtime YAML loading is
introduced.

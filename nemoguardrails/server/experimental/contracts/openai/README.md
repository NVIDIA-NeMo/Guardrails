# OpenAI Chat Completions contract

Operation `createChatCompletion` uses the `single_text.v1` capability profile
for its guarded text boundary. This page summarizes that boundary and its
provider source. Neither the summary nor the source pin enables runtime
behavior. See [Guard contracts](../README.md) for the document vocabulary.

## Guarded boundary

| Area | Single-text boundary |
| --- | --- |
| Request | One user message with non-empty string content at `messages[0].content`. |
| Buffered response | One assistant choice with non-empty string content at `choices[0].message.content`. |
| Constrained values | User/assistant roles, single-item arrays, and request `n` constrained to one. |
| Unsupported content | Tool, audio, multimodal, refusal, and separate reasoning content where explicitly disabled. |
| Opaque data | Reviewed provider-owned metadata and controls, not additional guarded subjects. |
| Closed objects | The request, its user message, the buffered response, and its choice and assistant message reject members outside OpenAI's fields. Fields specific to compatible servers belong to their own reviewed extensions. |

This request boundary does not accept conversation histories, system/developer
messages alongside the user message, or multimodal content blocks. Disabled
fields are null-only: a non-null value does not become acceptable merely because
it requests plain text or no tools. Response `tool_calls` is the one exception:
it is constrained to null or an empty list, because OpenAI's schema allows an
empty list and it carries no tool content.

Log probabilities are unsupported: they carry token text that rails do not
inspect. Request `logprobs` accepts only `false` or null, request `top_logprobs`
accepts only null, and buffered response `choices[0].logprobs` accepts only null.
The request check applies even when output inspection is off.

Text replacement eligibility is separate from endpoint support for replacement
outcomes. Response annotations must be null or empty, because citation text is
not inspected. The policy still declares that non-empty annotations would block
text replacement. Unrelated provider data must remain intact.

The Python projections, bindings, and endpoint determine exact acceptance and
runtime behavior. Their machine-readable contract belongs with the integration,
not in a separately maintained handwritten policy here. Recognizing the request
`stream` flag does not itself provide a streaming endpoint. Streaming needs its
own event classification, lifecycle handling, and implementation tests; the
buffered boundary is not a claim about accepted stream events.

## Provider provenance

[source.yaml](source.yaml) records the OpenAPI document's immutable revision,
document version, download location, and expected SHA-256 digest. The pin is a
review baseline, not a claim to cover every current OpenAI field or compatible
provider.

The offline metadata tests check pin structure and URL/revision consistency.
They do not fetch the document or verify its bytes against the digest. Reviewing
a provider update requires checking the source artifact and changes to guarded,
constrained, disabled, and opaque fields separately from format validation.

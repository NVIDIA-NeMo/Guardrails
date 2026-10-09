# OpenAI Chat Completions contract

Operation `createChatCompletion` uses the `single_text.v1` capability profile
for its guarded text boundary. This page summarizes that boundary and its
provider source. Neither the summary nor the source pin enables runtime
behavior. See [Guard contracts](../README.md) for the document vocabulary.

## Guarded boundary

| Area | Single-text boundary |
| --- | --- |
| Request | One user message with non-empty string content at `messages[0].content`. |
| Buffered response | One assistant choice with string content at `choices[0].message.content`. Empty or null content means the response has nothing to inspect, because every other reviewed field is content-free. The projection reports this through `has_text`, and the buffered integration relays such a response without output checks. |
| Streamed response | At most one choice, with integer index zero and optional assistant text at `choices[0].delta.content`. Missing or null text is metadata; an empty choices array allows usage chunks. Reviewed root metadata passes through. |
| Constrained values | User/assistant roles, single-item arrays, and request `n` constrained to one. |
| Unsupported content | Tool, audio, multimodal, refusal, participant name, and separate reasoning content where explicitly disabled. |
| Opaque data | Reviewed provider-owned metadata and controls, not additional guarded subjects. |
| Closed objects | The request, its user message, the buffered response, and its choice and assistant message reject members outside OpenAI's fields. The user message, choice, and assistant message are configurable: only trusted configuration may allow unknown members there. Fields specific to compatible servers belong to their own reviewed extensions. |

This request boundary does not accept conversation histories, system/developer
messages alongside the user message, or multimodal content blocks. Disabled
fields are null-only: a non-null value does not become acceptable merely because
it requests plain text or no tools. A few unsupported features are constrained
instead, because OpenAI's schema also allows a value that carries no content:
response `tool_calls` and `annotations` accept null or an empty list, and
request `logprobs` accepts null or `false`.

Log probabilities are unsupported: they carry token text that rails do not
inspect. Request `logprobs` accepts only `false` or null, request `top_logprobs`
accepts only null, and both buffered and stream `choices[0].logprobs` accept only null.
The request check applies even when output inspection is off.

Text replacement eligibility is separate from endpoint support for replacement
outcomes. Response annotations must be null or empty, because citation text is
not inspected. The policy still declares that non-empty annotations would block
text replacement. Unrelated provider data must remain intact.

The Python projections, bindings, and endpoint determine exact acceptance and
runtime behavior. The [exported contract](_generated/chat-completions.guard.yaml)
describes buffered payloads and the bound stream classifier, not a separately
maintained handwritten policy. The stream root, choice, delta, and error envelope
are closed. Error details are configurable, but the classifier validates with
the closed default, so unreviewed error fields are rejected too.

The stream export includes event selection, roles, missing-text handling, and
SSE sentinel/non-data declarations. Stateful hooks enforce completion and event
ordering and frame provider-native errors; they remain handwritten and are not
serialized. Exporting a contract does not enable streaming dispatch or reproduce
arbitrary Python validation. For example, the runtime rejects choice index
`0.0`, while JSON Schema treats it as the integer zero. See the integration's
[export guide](../../providers/openai/chat_completions/README.md#contract-export)
for generation, drift checks, and limitations.

## Provider provenance

[source.yaml](source.yaml) records the OpenAPI document's immutable revision,
document version, download location, and expected SHA-256 digest. The pin is a
review baseline, not a claim to cover every current OpenAI field or compatible
provider.

The offline metadata tests check pin structure and URL/revision consistency.
They do not fetch the document or verify its bytes against the digest. Reviewing
a provider update requires checking the source artifact and changes to guarded,
constrained, disabled, and opaque fields separately from format validation.

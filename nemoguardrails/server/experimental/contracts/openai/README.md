# OpenAI Provider Sources

The native bindings and Python policies define runtime coverage. Use the
[Chat boundary](../../providers/openai/chat_completions/README.md#guarded-boundary)
and [Responses boundary](../../providers/openai/responses/README.md#supported-content)
for the detailed implementation references. The published
[provider guide](https://docs.nvidia.com/nemo/guardrails/latest/run-guardrailed-inference/using-proxy/providers-and-streaming)
summarizes supported requests, output, and streaming.

## Provider provenance

[source.yaml](source.yaml) records the OpenAPI document's immutable revision,
document version, download location, and expected SHA-256 digest. The pin is a
review baseline, not a claim to cover every current OpenAI field or compatible
provider.

Compatible-server policy is selected separately. The
[reviewed LiteLLM dialect](../../providers/openai/dialects/litellm/README.md)
adds constrained buffered extension containers without changing the default
OpenAI endpoint. Its complete effective policy is exported to
[a separate contract](_generated/litellm-chat-completions.guard.yaml).

The offline metadata tests check pin structure and URL/revision consistency.
They do not fetch the document or verify its bytes against the digest. Reviewing
a provider update requires checking the source artifact and changes to guarded,
constrained, disabled, and opaque fields separately from format validation.

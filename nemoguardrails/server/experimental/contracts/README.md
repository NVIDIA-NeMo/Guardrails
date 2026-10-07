# Guarded provider contracts

A guard contract makes the guardrail boundary of a provider operation readable:
which fields contain guarded content, which values are constrained, and which
data stays opaque under provider authority. You do not need to read Python to
understand that policy. Your deployment separately decides which rails inspect
the selected content.

Status: experimental, `1.0.0-alpha.1`. The provider OpenAPI remains authoritative
for HTTP shapes. A guard contract is a custom format for the NVIDIA NeMo
Guardrails library that uses JSON Schema vocabulary with `x-nemo-guardrails`
annotations. It is neither an OpenAPI document nor an
[OpenAPI Overlay](https://spec.openapis.org/overlay/v1.1.0.html), and generic
OpenAPI or JSON Schema tools do not apply its guardrail semantics.

## A complete small contract

```yaml
version: '1.0.0-alpha.1'
operationId: createText
profile: single_text.v1
request:
  type: object
  required: [prompt]
  properties:
    prompt:
      type: string
      minLength: 1
      x-nemo-guardrails:
        classification: guarded
        subject:
          kind: text
          role: user
          replaceable: true
  x-nemo-guardrails:
    opaque_fields: [model]
response:
  type: object
  required: [text]
  properties:
    text:
      type: string
      minLength: 1
      x-nemo-guardrails:
        classification: guarded
        subject:
          kind: text
          role: assistant
          replaceable: true
    status:
      const: complete
      x-nemo-guardrails:
        classification: constrained
        reason: projection_policy.complete_text
integration:
  endpoint:
    unsupported_request_code: unsupported_text_request
    unsupported_response_code: unsupported_text_response
```

This [executable example](minimal.guard.example.yaml) is paired with a
[small provider description](minimal.openapi.example.yaml). It guards the
request's `prompt` and response's `text`, delegates `model` to the provider, and
constrains response `status` to `complete`. `replaceable: true` declares that
the selected text may be replaced while preserving unrelated data. It records
policy only: the guarded proxy does not execute replacement yet and rejects every
replacement decision.

`operationId` selects one operation in the provider OpenAPI.
`request` and `response` are required; add `stream` for streaming and
`excluded_stream_events` for deliberately unsupported event schemas.
`integration` holds runtime bindings, separate from policy.

## Reading the policy

| Declaration | Meaning |
| --- | --- |
| `classification: guarded` | Contains guarded content directly or through explicitly described children. |
| `classification: constrained` | Locally restricts structure or values without making that field a subject. |
| `classification: opaque` | Delegates the entire value to the provider; the library does not inspect its nested shape. |
| `opaque_fields: [id, model]` | Shorthand for explicit opaque properties on this object only. |
| `subject` | Identifies guarded content: its kind, its user/assistant role, and its replacement policy. |
| `reason` | Records why a restriction exists as a namespaced identifier, such as `core_capability.tool_content`. |

Each selected object accounts for every known provider property exactly once.
This version supports one subject kind, `text`, because rails currently inspect
text only.
The [reference](reference.md) defines each annotation with its semantics,
examples, and edge cases, including opaque values, unknown fields, requiredness,
nulls, defaults, replacement, and streaming.

## Version and stability

The only accepted contract version is `"1.0.0-alpha.1"`. Unsupported versions and
unknown keys are errors; there are no version aliases or alternate layouts.
Authoring and semantics may change before a stable `1.0.0`. Increment the alpha
revision when you publish an incompatible contract revision, and never
reinterpret a published revision. This follows
[Semantic Versioning](https://semver.org/#spec-item-9).

Contract versions are independent of capability profile identifiers such as
`single_text.v1`, which identify semantic boundaries rather than the document
format. Each operation has one maintained contract.

## Editor support and validation

[guard-contract.schema.json](guard-contract.schema.json) is a self-contained
JSON Schema 2020-12 schema. Associate it with `*.guard.yaml` in your editor. For
the [YAML language server](https://github.com/redhat-developer/yaml-language-server#associating-schemas),
the equivalent workspace setting is:

```json
{
  "yaml.schemas": {
    "./nemoguardrails/server/experimental/contracts/guard-contract.schema.json": "**/*.guard.yaml"
  }
}
```

The schema offers key/value completion and catches misspellings, wrong types,
unsupported schema keywords, blank exclusion reasons, and malformed integration
metadata. It does not prove provider compatibility; see the
[validation limits](reference.md#validation-and-compatibility-limits).

## Authoring and review

Start from the [minimal contract](minimal.guard.example.yaml), review it against
the selected provider operation, and validate it with the schema while you edit.
When you change a constraint, edit that field's schema and reason together. Do
not mark a newly documented content-bearing field opaque to silence a review
error.

Before you submit a new or changed operation, verify:

- The provider source is immutable, pinned, and digest-verified.
- The guard contract uses standard Schema Objects for structure.
- Every provider field at every guarded object boundary is classified once.
- Every opaque field has been reviewed as independent of guarded content.
- Every constrained field is necessary to make the current capability safe and
  unambiguous.
- Every disabled feature has a structured reason.
- Every subject has the correct kind, role, and explicit replacement policy.
- Arrays and unions identify exactly the supported subject.
- Stream event variants are disjoint and cover every accepted provider branch.
- Stateful stream behavior remains in handwritten hooks rather than the
  contract.
- Endpoint symbols live inside the provider's runtime package.
- Schema validation and the relevant runtime tests succeed.

## Provider contracts

Provider-specific contracts are introduced with their provider capabilities.
Each operation contract covers its request, buffered response, and optional
streaming boundary as one policy document. Use the [reference](reference.md)
for precise syntax and meaning.

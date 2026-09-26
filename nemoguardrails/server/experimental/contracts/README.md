# Guarded provider contracts

A guard contract makes the guardrail boundary readable: which fields contain
guarded text, which values are constrained, and which data stays opaque under
provider authority. You do not need to read Python to understand that policy.
The deployment separately decides which rails inspect the selected text.

Status: experimental, `1.0.0-alpha.1`. The provider OpenAPI remains authoritative
for HTTP shapes. This is a custom contract format using JSON Schema vocabulary,
not an OpenAPI document or Overlay. Overlay 1.1 is used for normalization and
portable export.

## A complete small contract

```yaml
version: '1.0.0-alpha.1'
operationId: createText
profile: single_text.v1
request:
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
constrains response `status` to `complete`. `replaceable: true` permits replacing
the selected text while preserving unrelated data.

`operationId` selects one operation in the normalized provider OpenAPI.
`request` and `response` are required; add `stream` for streaming and
`excluded_stream_events` for deliberately unsupported event schemas.
`integration` holds runtime bindings, separate from policy. The supported
profiles and binding restrictions are listed in the
[reference](reference.md#validation-and-compatibility-limits).

## Reading the policy

| Declaration | Meaning |
| --- | --- |
| `classification: guarded` | Contains guarded text directly or through explicitly described children. |
| `classification: constrained` | Locally restricts structure or values without making that field a text subject. |
| `classification: opaque` | Delegates the entire value to the provider; Guardrails does not inspect its nested shape. |
| `opaque_fields: [id, model]` | Shorthand for explicit opaque properties on this object only. |
| `subject` | Identifies text, its user/assistant role, and replacement policy. |
| `reason` | Records why a restriction exists, using the existing namespaced reason vocabulary. |

A guarded field can also have constraints. These are not three separate maps
or mutually exclusive sets of schema keywords. `opaque` is not a safety claim:
authors must decide whether a value can affect guarded content before delegating
it. Nested provider changes inside an opaque value do not trigger local review.

Every selected object must account for every known provider property exactly
once. Name opaque fields explicitly; wildcards, duplicates, and overlap with
`properties` are errors. Unknown runtime keys follow `additionalProperties`
and, where declared, `unknown_fields: configurable`. Allowing unknown keys does
not waive review of newly documented provider fields at a selected boundary.

The [reference](reference.md) puts each annotation beside its semantics,
examples, and edge cases, including requiredness, nulls, defaults, replacement,
and streaming.

## Version and stability

The only accepted contract version is `"1.0.0-alpha.1"`. It identifies this
experimental authoring envelope and its documented semantics. Unsupported
versions and unknown keys are errors; there are no version aliases or alternate
layouts.

A bare `version: 1` would identify a format, not inherently promise stability.
The explicit prerelease label makes the status clear: authoring and semantics
may change before a stable `1.0.0`. Increment the alpha revision when publishing
an incompatible contract revision; never silently reinterpret a published
revision. This follows [Semantic Versioning](https://semver.org/#spec-item-9).

Contract versions are independent of `overlay: 1.1.0`, provider manifest
`version: 1`, generated annotation versions, and capability profile identifiers.
Those identify different formats or semantic boundaries.

List guard contracts directly in `provider.yaml`:

```yaml
operations:
  - chat-completions.guard.yaml
```

Each operation has one maintained contract. OpenAPI Overlays are used for
evidence-backed normalization and portable export, not as operation-authoring
inputs.

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
metadata. It is not proof of provider compatibility. Provider coverage, source
reachability, text-target cardinality, hook compatibility, and runtime capability
require semantic validation against the provider source. Provider-aware
validation and generation are introduced with the compiler.

## Authoring workflow

Start from the [minimal contract](minimal.guard.example.yaml) and review it
against the selected provider operation. Classify every selected provider field,
declare accepted or explicitly excluded stream events when streaming is present,
and put runtime bindings under `integration`. List the `*.guard.yaml` filename
in the provider manifest.

Use schema validation while authoring. When changing a constraint, edit that
field's schema and reason, then review it against the pinned provider
description. Do not mark a newly documented content-bearing field opaque merely
to silence a review error.

## Author review checklist

Before submitting a new or changed operation, verify:

- The provider source is immutable, pinned, and digest-verified.
- Normalization actions cite evidence and contain provider facts, not Guardrails
  policy.
- The guard contract uses standard Schema Objects for structure.
- Every provider field at every guarded object boundary is classified once.
- Every opaque field has been reviewed as independent of guarded content.
- Every constrained field is necessary to make the current capability safe and
  unambiguous.
- Every disabled feature has a structured reason.
- Guarded text has the correct role and explicit replacement policy.
- Arrays and unions identify exactly the supported text target.
- Stream event variants are disjoint and cover every accepted provider branch.
- Stateful stream behavior remains in handwritten hooks rather than the
  contract.
- Endpoint symbols live inside the declared runtime package.
- Schema validation and the relevant runtime tests succeed.

## Standards boundary

This document format is custom, using familiar JSON Schema vocabulary and
explicit `x-nemo-guardrails` semantics. It is neither an OpenAPI document nor an
OpenAPI Overlay. Its exported transformation follows
[Overlay 1.1](https://spec.openapis.org/overlay/v1.1.0.html); ordinary Overlay
tools apply the transformation, not guardrail behavior.

See the [semantic limits](reference.md#validation-and-compatibility-limits)
for the current implementation boundary and known schema/runtime differences.

## Provider contracts

Provider-specific contracts are introduced with their provider capabilities.
Each operation contract covers its request, buffered response, and optional
streaming boundary as one policy document. Use the [reference](reference.md)
for precise syntax and meaning.

# Guard contracts

A guard contract describes a provider operation's guardrail boundary: which
content is inspected, which values are restricted, and which data remains under
provider authority. It makes that boundary reviewable without reading Python.

The Python projection models, bindings, and endpoint implementation are the
source of truth. A contract is a description of their policy, not server
configuration or a second policy to maintain by hand. Deployment configuration
selects the rails that inspect supported content.

This directory defines the experimental document format and an illustrative
example. Provider exports belong with their integrations; the format alone does
not implement an endpoint, an exporter, or runtime YAML loading.

## Document structure

Start with [minimal.guard.example.yaml](minimal.guard.example.yaml). It marks
request `prompt` and response `text` as guarded, leaves `model` opaque, and
constrains response `status` to `complete`. It is a format example, not the
output of a provider integration.

| Section | Describes |
| --- | --- |
| `version`, `operationId`, `profile` | Format revision, provider operation, and framework capability. |
| `request`, `response` | Request and buffered-response payload projections. |
| `stream` | Optional event payloads, classifications, and transport framing. |
| `excluded_stream_events` | Explicitly unsupported event schemas and the reasons for exclusion. |
| `integration.endpoint` | Endpoint metadata such as route, rejection codes, and implementation bindings. |

Payload projections use a subset of JSON Schema vocabulary: `type`,
`properties`, `required`, `items`, `oneOf`, constants, enums, and bounds.
NeMo policy is carried by `x-nemo-guardrails`.

## Field policy

| Declaration | Meaning |
| --- | --- |
| `classification: guarded` | A text subject, or a container leading to it. |
| `subject` | Identifies text, its user/assistant role, and replacement eligibility. |
| `classification: constrained` | Restricts a value or shape without making it a rail subject. |
| `gate: disabled` | Marks a constrained, null-only field for an unsupported feature. |
| `classification: opaque` | Leaves the value under provider authority, without guardrail inspection. |
| `opaque_fields` | Lists reviewed opaque names on this object without declaring each property. |
| `reason` | Explains a restriction: capability, provider integrity, or projection policy. |

Opaque does not mean safe or trusted. Its nested content is not independently
reviewed. Opaque names should not overlap declared properties or use wildcards.
`extension: true` identifies a local compatibility field outside the pinned
provider schema, not an exemption from review.

An object's `required` list controls presence; nullability is separate. Disabled
fields can be omitted when optional, or supplied as null, but cannot enable the
feature. A `default` describes an internal default, not permission to insert
fields into the forwarded provider payload.

`replaceable` records text replacement eligibility. `replacement_blocked_by`
identifies metadata that can prevent replacement, with a `replacement_reason`.
The integration determines the blocking predicate and whether replacement
outcomes can be executed. Neither marking a subject nor marking it replaceable
enables a detector or a runtime capability.

## Object and stream metadata

`model` identifies the Python model; `source` identifies a provider schema
component. `unknown_fields: configurable` marks an object whose unknown-field
handling depends on runtime validation context. Generic JSON Schema validators
do not enforce this annotation. `additionalProperties` remains the ordinary
schema-level object closure rule. Provider-required opaque fields may be left
to provider validation; a projection is not a full provider request validator.

Object `variant` metadata describes discriminator matching and selected-variant
cardinality. This is distinct from ordinary array length constraints.
`stream_selector_field` identifies a request flag, not streaming support.

A stream's `oneOf` branches describe event families: `guarded_delta`, `snapshot`,
`opaque`, or `provider_error`. Their variants identify source schemas, matching
rules, and classifier shape names. Missing-text policy is distinct from schema
nullability. Transport metadata describes SSE discriminators and sentinels.
Event exclusions and external-source reasons record scope and provenance; they
do not establish that every provider event has been reviewed.

Stateful event ordering, lifecycle checks, conditional replacement, and native
error encoding remain implementation behavior. Endpoint symbol references are
descriptive metadata, not instructions to load code from an untrusted document.

## Validation and stability

[guard-contract.schema.json](guard-contract.schema.json) validates the document
format and rejects unsupported keys in the reserved guardrail vocabulary. Other
`x-*` schema annotations are allowed. The schema can be associated with
`*.guard.yaml` in a JSON Schema-aware editor.

Format validation does not establish complete provider coverage, correct text
extraction, preservation of unrelated data, or equivalence to arbitrary Python
validators and coercion. These require implementation and source-aware tests.
Provider revision and digest checks are separate from document validation.

The format is experimental: `1.0.0-alpha.1`, not stable version 1. Its version
is separate from the capability profile `single_text.v1` and provider API
revision. A profile identifies the supported content capability; it does not
choose deployment rails or enable runtime profile selection.

This is a NeMo Guardrails document using JSON Schema vocabulary, not an OpenAPI
document or an OpenAPI Overlay. Generic schema tools do not execute its guardrail
annotations.

## Provider boundaries

- [OpenAI Chat Completions](openai/README.md)

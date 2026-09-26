# Guard contract reference

This is the syntax and semantic specification for `1.0.0-alpha.1`
(experimental). Start with the [README](README.md) for a complete small example
and the editing workflow. This reference defines what an implementation must
preserve, not just how the compiler currently generates Python.

The [authoring schema](guard-contract.schema.json) defines accepted syntax:
a limited subset of JSON Schema with local `x-nemo-guardrails` annotations.
Generic JSON Schema tools do not enforce Guardrails-specific semantics.

## Authority and scope

The pinned provider description owns HTTP paths, source schemas, and provider
facts. Evidence-backed normalization corrects that description before guard
policy is interpreted. A guard contract selects a supported boundary; it does
not redefine the whole provider API or claim to configure all detectors,
prompts, input/output rails, or deployment settings.

Request and response policy describes structural acceptance and text targets.
The deployment separately selects which rails inspect that text and whether a
rail result permits, rejects, or replaces it. A marked subject is a guardable
surface, not a guarantee that every deployment runs a particular safety check.

## Examples

Begin with the [minimal contract](minimal.guard.example.yaml), paired with
its [provider OpenAPI](minimal.openapi.example.yaml). Complete examples are
introduced with their provider capabilities.

For authoring and validation steps, use the [guard contract guide](README.md).

Root component names, projection inventories, operation bindings, and source
bindings are derived. They are not a second set of authoring decisions.

## Payload Projections

A Payload Projection is a normal OpenAPI Schema Object with local Guardrails
annotations. It describes either a request or a buffered JSON response.

### Minimal request example

```yaml
request:
  type: object
  additionalProperties: true
  title: example.chat.request.text.v1
  properties:
    messages:
      type: array
      minItems: 1
      maxItems: 1
      items:
        type: object
        additionalProperties: true
        title: ExampleUserMessageProjection
        properties:
          role:
            const: user
            x-nemo-guardrails:
              classification: constrained
          content:
            type: string
            minLength: 1
            x-nemo-guardrails:
              classification: guarded
              subject:
                kind: text
                role: user
                replaceable: true
        required: [role, content]
        x-nemo-guardrails:
          model: ExampleUserMessageProjection
          source: '#/components/schemas/ProviderUserMessage'
          unknown_fields: configurable
      x-nemo-guardrails:
        classification: guarded
    stream:
      type: boolean
      default: false
      x-nemo-guardrails:
        classification: constrained
    model:
      x-nemo-guardrails:
        classification: opaque
  required: [messages]
  x-nemo-guardrails:
    model: ExampleChatGuardedRequestProjection
    stream_selector_field: stream
```

The schema describes the local request policy. The frontend derives the root
provider source, direction, and capability metadata from the contract and
selected operation. Optional `title` and `x-nemo-guardrails.model` give stable
projection and class-name overrides. `stream_selector_field` identifies the
boolean request field that selects streaming.

### Complete field accounting

Every field declared by a bound provider object must be classified exactly
once:

| Classification | Meaning |
| --- | --- |
| `guarded` | Contains or leads to content inspected by Guardrails. |
| `constrained` | Narrows the provider shape so guarded extraction, replacement, or response handling is safe and unambiguous. |
| `opaque` | Remains provider-owned and is forwarded without Guardrails interpretation. |
| local extension | Does not exist in the provider source; it is an explicitly capability-gated compatibility field. |

For example:

```yaml
properties:
  input:
    type: string
    minLength: 1
    x-nemo-guardrails:
      classification: guarded
      subject:
        kind: text
        role: user
        replaceable: true

  response_format:
    type: 'null'
    default: null
    x-nemo-guardrails:
      classification: constrained
      gate: disabled
      reason: projection_policy.plain_text_output

  temperature:
    x-nemo-guardrails:
      classification: opaque
```

Compilation fails if a field from the provider source reaches this object and
is not reviewed. This is the provider-drift boundary: a new provider field
cannot silently acquire guard semantics.

`opaque` does not mean “unknown” or “ignored.” It records a deliberate decision
that the field does not affect the guarded capability and can remain under
provider authority. Opaque does not mean trusted or safe: the entire value is
delegated to the provider and must be preserved without local assertions.

`opaque_fields` is shorthand for individually annotated opaque properties on
this object. Names must be explicit and unique, cannot overlap `properties`,
and cannot use wildcards. It does not review nested fields inside an opaque
value. New known fields at a selected boundary require review even when
runtime unknown fields are allowed.

### Guarded text subjects

The `single_text.v1` profile identifies one user or assistant text subject.
Selected native representations do not permit multiple independent subjects.
Replacement may affect only the selected text; unrelated metadata and declared
replacement restrictions must be preserved.

A direct guarded string declares its provider-neutral meaning:

```yaml
content:
  type: string
  minLength: 1
  x-nemo-guardrails:
    classification: guarded
    subject:
      kind: text
      role: assistant
      replaceable: true
```

Subject keys are:

| Key | Meaning |
| --- | --- |
| `role` | `user` for input or `assistant` for generated output. |
| `replaceable` | Whether a buffered Guardrails decision may replace this provider field. Required by `single_text.v1`. |
| `replacement_blocked_by` | Related provider field that may make replacement unsafe. |
| `replacement_reason` | Structured reason for a restriction on replacement. |

For example, replacing cited text may invalidate its citations:

```yaml
subject:
  kind: text
  role: assistant
  replaceable: true
  replacement_blocked_by: citations
  replacement_reason: provider_integrity.cited_text
```

The runtime can replace the text only when the related field is absent. If
`replaceable` is false, or replacement is conditional, a
`replacement_reason` is required.

### Constraints and capability gates

Standard JSON Schema keywords describe the narrowed provider value. The
semantic compiler currently recognizes these selection and validation
keywords:

```text
type, const, default, enum,
minItems, maxItems, minLength, maxLength, pattern
```

The compiler rejects unsupported schema keywords and unknown Guardrails keys
with the component and nested property location. It accepts descriptive
annotations such as `description` and `examples` without treating them as
constraints. Opaque properties cannot carry validation assertions: classify a
property as constrained when the guard projection should validate it. A disabled
gate must agree with its null-only schema.

Examples include selecting one role:

```yaml
role:
  const: user
  x-nemo-guardrails:
    classification: constrained
```

and restricting generation to one output:

```yaml
n:
  const: 1
  default: 1
  x-nemo-guardrails:
    classification: constrained
    reason: core_capability.single_text_target
```

A disabled capability uses a null-only projection:

```yaml
tools:
  type: 'null'
  default: null
  x-nemo-guardrails:
    classification: constrained
    gate: disabled
    reason: core_capability.tool_content
```

`gate: disabled` means the guarded operation does not support that capability;
it does not mean the provider lacks it. A gated field must be constrained and
must provide a reason. A disabled field may be absent or null but must not
enable the unsupported feature. Normalization must not disguise this local
restriction as a provider fact.

Reason identifiers use one of these namespaces:

| Prefix | Use |
| --- | --- |
| `core_capability.*` | The shared runtime capability does not yet support the provider feature. |
| `provider_integrity.*` | Accepting or changing the value could invalidate related provider data or protocol guarantees. |
| `projection_policy.*` | The chosen guarded projection deliberately narrows otherwise supported provider behavior. |

Use a stable lower-case identifier after the prefix, for example
`core_capability.tool_content`. Reasons are reviewable policy identifiers, not
free-form explanations.

### Defaults

`default` is an annotation, not permission to insert an absent property into
the forwarded provider payload. Generated models may use defaults internally;
that does not authorize rewriting provider input. See the
[JSON Schema annotation guidance](https://json-schema.org/understanding-json-schema/reference/annotations).

### Requiredness

The standard object-level `required` array determines which properties must be
present in the guarded projection. Properties omitted from this array are
optional, even when the provider requires them. Keep a policy `reason` on a
property when deliberately changing provider requiredness.

```yaml
required: [messages]
```

Field-level `x-nemo-guardrails.required` is not accepted. Optionality does not
imply nullability in JSON Schema: the value schema must explicitly permit null.
Provider-required opaque values remain under provider authority rather than
being promoted to local validation.

### Nested objects and arrays

Attach object metadata to every nested object that the compiler traverses:

```yaml
message:
  type: object
  properties:
    content: {}
  x-nemo-guardrails:
    classification: guarded
    model: ExampleMessageProjection
    source: '#/components/schemas/ProviderMessage'
    unknown_fields: configurable
```

For an array, put the nested object metadata on `items` and classify the array
field itself:

```yaml
messages:
  type: array
  minItems: 1
  maxItems: 1
  items:
    type: object
    title: ExampleMessageProjection
    properties: {}
    x-nemo-guardrails:
      model: ExampleMessageProjection
      source: '#/components/schemas/ProviderMessage'
      unknown_fields: configurable
  x-nemo-guardrails:
    classification: guarded
```

Author a direct guarded array with `minItems: 1` and `maxItems: 1` when the
guarded target is its first item. For union variants, the current
`single_text.v1` profile requires an `exactly_one` matching selection.
Cardinality is therefore part of safety, not merely validation convenience.

### Union variants

Arrays and alternatives must identify the supported text unambiguously.
Explicit match selectors must agree with declared constant fields. Authored
alternative order must not supply identity. The `single_text.v1` profile also
requires nonempty text and an unambiguous traversal.

Use variants when an array contains a provider union and only selected object
variants participate in the guarded capability:

```yaml
content:
  type: array
  minItems: 1
  maxItems: 1
  items:
    oneOf:
      - type: object
        title: ExampleTextBlockProjection
        properties:
          type:
            const: text
            x-nemo-guardrails:
              classification: constrained
          text:
            type: string
            minLength: 1
            x-nemo-guardrails:
              classification: guarded
              subject:
                kind: text
                role: assistant
                replaceable: true
        required: [type, text]
        x-nemo-guardrails:
          model: ExampleTextBlockProjection
          source: '#/components/schemas/ProviderTextBlock'
          unknown_fields: configurable
          variant:
            match:
              type: text
            cardinality: exactly_one
  x-nemo-guardrails:
    classification: guarded
```

`variant.match` can be omitted when required properties with `const` values
identify the variant. An explicit match must correspond to those selected
constants. `cardinality` is one of `any`, `at_most_one`, or `exactly_one`, although
the current payload profile requires `exactly_one` for the guarded target.

### Alternative text representations

Use a field-level `oneOf` when the same semantic text may use different native
encodings, such as a string or a one-element text-block array:

```yaml
content:
  oneOf:
    - type: string
      minLength: 1
      x-nemo-guardrails:
        subject:
          kind: text
          role: user
          replaceable: true
    - type: array
      minItems: 1
      maxItems: 1
      items:
        type: object
        properties:
          text:
            type: string
            minLength: 1
            x-nemo-guardrails:
              classification: guarded
              subject:
                kind: text
                role: user
                replaceable: true
        x-nemo-guardrails:
          model: ExampleInputTextBlockProjection
          source: '#/components/schemas/ProviderInputTextBlock'
          unknown_fields: configurable
      x-nemo-guardrails: {}
  x-nemo-guardrails:
    classification: guarded
```

Each representation must lead to exactly one text subject with the same
provider-neutral meaning. The compiler derives multiple native bindings for
one semantic subject.

### Unknown-field policies

Use standard `additionalProperties` for object closure. Add
`x-nemo-guardrails.unknown_fields: configurable` only for a content boundary
whose closure the runtime may override:

| Value | Generated behavior |
| --- | --- |
| `additionalProperties: true` or omitted | Accept additional provider fields. |
| `additionalProperties: false` | Reject additional fields, while retaining declared opaque properties. |
| `unknown_fields: configurable` with `additionalProperties: true` | Accept additional fields by default and allow the runtime's closed-content policy to reject unreviewed extras. |

Use `additionalProperties` for allow/forbid behavior rather than duplicating it
in the extension.

These policies do not replace field accounting against the pinned provider
contract. Known provider fields at a guarded boundary must still be marked
guarded, constrained, or opaque during compilation.

### Local extension fields

Occasionally a compatibility layer adds a field that is not claimed by the
provider source. Mark it explicitly:

```yaml
reasoning_content:
  type: 'null'
  default: null
  x-nemo-guardrails:
    classification: constrained
    extension: true
    gate: disabled
    reason: core_capability.reasoning_content
```

A local extension must be a disabled constrained capability. Do not use
`extension: true` to avoid proving that a real provider field exists.

## Stream Projections

Stream policy describes decoded provider data events and declared SSE framing.
It does not implicitly upgrade the source to OpenAPI 3.2 transport `itemSchema`.

A Stream Projection describes complete SSE event payloads. The compiler uses it
to generate typed event models and a stateless per-event classifier.

The projection does not describe cross-event ordering. Stateful sequence
validation and provider-native error framing remain handwritten stream hooks.

### Stream root

```yaml
stream:
  title: example.chat.stream.text.v1
  oneOf:
    - {}
    - {}
  x-nemo-guardrails:
    event_schema_binding: operation_response
    transport:
      require_sse_event: false
      non_data_shape: '[DONE]'
      sentinels:
        '[DONE]': '[DONE]'
```

Root stream keys are:

| Key | Meaning |
| --- | --- |
| `event_schema_binding` | `operation_response` when the SSE response reaches the event schema; `component` only when justified by external evidence. |
| `event_schema_reason` | Required reason for a component-only binding and forbidden for an operation-bound one. |
| `transport` | SSE discriminator, event-field, non-data, and sentinel behavior. |

Transport keys are:

| Key | Meaning |
| --- | --- |
| `data_discriminator` | JSON property used by the provider stream union discriminator, when present. |
| `require_sse_event` | Require the SSE `event:` field to match the JSON discriminator. |
| `non_data_shape` | Shape name for SSE events without a `data:` field. |
| `sentinels` | Map exact non-JSON `data:` values to stable shape names. |

When `require_sse_event` is true, `data_discriminator` is required.

### Stream event review

Every schema in the selected provider stream union must be accepted or named
in `excluded_stream_events` with a nonblank author reason. Unknown, duplicate,
stale, or simultaneously accepted/excluded entries are errors. A new event
schema must require a new author decision; implicit rejection is insufficient
as source-update review. External provider-error variants retain the existing
evidence-rationale requirement. Nested provider union coverage remains governed
by the existing selection checks; the exclusion list covers top-level event
schemas, not every future nested union alternative.

### Event families

Each root `oneOf` entry is an event family: one projection model and one
Guardrails role shared by one or more native provider variants.

```yaml
- type: object
  title: ExampleStreamPayloadProjection
  properties:
    object:
      const: chat.chunk
      x-nemo-guardrails:
        classification: constrained
    choices:
      type: array
      maxItems: 1
      items:
        type: object
        properties:
          delta:
            type: object
            properties:
              content:
                type: string
                x-nemo-guardrails:
                  classification: guarded
                  subject:
                    kind: text
                    role: assistant
            x-nemo-guardrails:
              classification: guarded
              model: ExampleStreamDeltaProjection
              unknown_fields: configurable
              source: '#/components/schemas/ProviderStreamDelta'
        x-nemo-guardrails:
          model: ExampleStreamChoiceProjection
          unknown_fields: configurable
          source: '#/components/schemas/ProviderStreamChoice'
      x-nemo-guardrails:
        classification: guarded
  required: [object, choices]
  x-nemo-guardrails:
    model: ExampleStreamPayloadProjection
    source: '#/components/schemas/ProviderChatStreamEvent'
    event:
      classification: guarded_delta
      variants:
        - source_schema: ProviderChatStreamEvent
          shape: chat.chunk:content
          match:
            object: chat.chunk
          required_fields: []
      missing_text: opaque
      missing_text_shape: chat.chunk:metadata
```

Event classifications are:

| Classification | Runtime role | Meaning |
| --- | --- | --- |
| `guarded_delta` | `GUARDED_TEXT` | Carries a new text delta inspected by Guardrails. |
| `snapshot` | `TEXT_SNAPSHOT` | Carries accumulated text checked against previously observed deltas. |
| `opaque` | `OPAQUE_METADATA` | Contains no guarded text. |
| `provider_error` | `PROVIDER_ERROR` | Provider-native in-stream failure. |

Each event variant has:

| Key | Meaning |
| --- | --- |
| `source_schema` | Provider component represented by this variant. |
| `shape` | Stable Guardrails name returned after classification. |
| `match` | Dotted JSON selectors and scalar values that identify the event. |
| `required_fields` | Top-level fields that must be required to distinguish the event. |
| `external_reason` | Evidence reason for a provider-error schema accepted outside the primary stream union. Other external variants are not supported. |

A variant must have at least one `match` entry or required field. The compiler
proves selectors against the provider schema and rejects duplicate selectors.
Rules must also be semantically disjoint: the runtime rejects an event if more
than one rule matches it.

### Text paths are derived

Authors do not write a `text_path`. They mark the nested text property with a
guarded subject. From the preceding example, the compiler derives something
equivalent to:

```python
StreamEventRule(
    shape="chat.chunk:content",
    role=StreamEventRole.GUARDED_TEXT,
    match=(("object", "chat.chunk"),),
    text_path=("choices", 0, "delta", "content"),
)
```

The generated path means
`payload["choices"][0]["delta"]["content"]`. Keeping the path derived prevents
the extraction logic from drifting away from the schema that validates it.

### Events without text

An opaque event family is not an opaque property: the family still has a
selected schema and constrained fields. Those constraints remain enforced even
though the family has no guarded text subject.

A structurally valid event in a guarded family may omit its text field. Choose
that behavior explicitly:

| `missing_text` | Behavior |
| --- | --- |
| `reject` | Reject the event. This is the default. |
| `opaque` | Reclassify it as metadata; `missing_text_shape` may give it a distinct shape, otherwise the original shape is reused. |
| `empty` | Treat it as an empty text snapshot. |

Only `guarded_delta` families may use `opaque`; only `snapshot` families may
use `empty`.

For example, Chat Completions chunks may carry role, usage, or finish metadata
without a content delta:

```yaml
event:
  classification: guarded_delta
  variants: [...]
  missing_text: opaque
  missing_text_shape: chat.completion.chunk:metadata
```

### Classifier versus hooks

The generated classifier answers questions about one event in isolation:

- Is its JSON shape valid?
- Which declared variant matched?
- Does it carry guarded text, a snapshot, metadata, or an error?
- What text should the shared runner inspect?

Handwritten hooks answer questions that require history or executable provider
behavior:

- Was a block started before receiving its delta?
- Is this event legal after the preceding event?
- Did the stream end with unfinished provider state?
- How should a Guardrails failure be framed as provider-native SSE?

An operation with streaming declares a hooks class under its endpoint metadata.
The class must be constructible without arguments and implement
`observe_event`, `validate_end_of_stream`, and `encode_error`. A fresh hooks
instance is created for every upstream stream; the stateless generated
classifier may be shared.

## Integration and stable identities

`integration.endpoint` holds error codes and optional route, label, stream-hook,
and API-revision bindings. When `route_path` or `operation_label` is omitted,
the route defaults to the selected provider path and the label is derived from
the operation name. Stateful stream behavior is
still implemented by the declared hook; changing a hook is a behavior change.

`integration.name` optionally preserves an existing artifact identity. Otherwise
the name is derived from `operationId`, for example `createText` becomes
`create_text`. Root component names, projection references, and operation binding
actions are derived. Existing `title` and `x-nemo-guardrails.model` overrides
remain supported to preserve names; renaming them may change generated imports.
Unnamed object alternatives need an explicit source or one event variant so
their derived identity does not depend on list position.

Types may be inferred only where unambiguous. A scalar whose provider type is
nullable, a union, or absent needs an explicit type. Selecting an object union
may require `x-nemo-guardrails.source`. No classification or subject is inferred.

## Endpoint binding

Runtime metadata belongs in the contract's integration section:

```yaml
integration:
  name: chat
  endpoint:
    route_path: /v1/chat
    operation_label: Example Chat
    unsupported_request_code: unsupported_example_chat_shape
    unsupported_response_code: unsupported_example_chat_response_shape
    stream_hooks:
      module: example_guardrails_provider.chat.stream_hooks
      name: ExampleChatStreamHooks
    api_revision:
      module: example_guardrails_provider.chat.api_revision
      name: EXAMPLE_API_REVISION
```

The compiler derives projection references and the operation declaration.
`integration.name` is optional and otherwise derived from `operationId`.

Endpoint keys are:

| Key | Meaning |
| --- | --- |
| `route_path` | Runtime route; it must match the provider operation, optionally with a concrete leading API-version segment. |
| `operation_label` | Human-readable name used in diagnostics. |
| `unsupported_request_code` | Provider-facing error code for unsupported request shapes. |
| `unsupported_response_code` | Provider-facing error code for unsupported response or stream shapes. |
| `stream_hooks` | Optional provider-package Python symbol for stateful streaming behavior. |
| `api_revision` | Optional provider-package `ProviderApiRevisionBinding` symbol. |

Python symbols must live inside the runtime package declared by
`provider.yaml`. Stream hooks are required when a stream projection is bound
and forbidden when no stream projection exists.

## `x-nemo-guardrails` quick reference

The meaning of a key depends on where the extension is attached. The same
extension name is intentionally local rather than one large embedded document.

### Projection object metadata

These keys appear on a root projection or nested object Schema Object:

| Key | Applies to | Meaning |
| --- | --- | --- |
| `model` | payload and stream objects | Optional Python model-name override. An identifier-valued `title` supplies the default. |
| `source` | payload and stream objects | Provider component reference. Root bindings are derived; use it on nested objects that select a named provider schema and omit it for inline objects. |
| `unknown_fields` | payload and stream objects | Optional runtime configurability; ordinary closure comes from `additionalProperties`. |
| `stream_selector_field` | request root | Field whose boolean value selects a streaming response. |
| `event_schema_binding` | stream root | `operation_response` or `component`. |
| `event_schema_reason` | stream root | Required structured reason for a component-only stream binding. |
| `transport` | stream root | Stream framing and discriminator metadata. |

### Field metadata

These keys appear on a property Schema Object:

| Key | Meaning |
| --- | --- |
| `classification` | `guarded`, `constrained`, or `opaque`. |
| `subject` | Direct guarded-text meaning and replacement policy. |
| `reason` | Structured capability, integrity, or projection-policy rationale. |
| `gate` | Currently only `disabled`; requires a constrained field and reason. |
| `extension` | Marks a local compatibility field absent from the provider source. Only valid for a disabled constrained field. |
| `model` | Optional generated name for a nested object, overriding its `title`. |
| `source` | Binds that nested object to a provider component. |
| `unknown_fields` | Sets that nested object's unknown-field policy. |

Array item objects carry `title`, optional `model`, `source`, and `unknown_fields` on the
`items` Schema Object. A field-level representation branch carries either a
direct `subject` or an annotated `items` object, not an ordinary field
classification.

### Variant, event, and endpoint metadata

| Location | Key | Meaning |
| --- | --- | --- |
| object union branch | `variant.match` | Optional selector override; defaults to required properties with `const` values. |
| object union branch | `variant.cardinality` | `any`, `at_most_one`, or `exactly_one`. |
| stream event family | `event.classification` | `guarded_delta`, `snapshot`, `opaque`, or `provider_error`. |
| stream event family | `event.variants` | Provider schemas and selectors represented by the family. |
| stream event family | `event.missing_text` | `reject`, `opaque`, or `empty`. |
| stream event family | `event.missing_text_shape` | Optional alternate shape used when guarded text is absent. |

## Evidence-backed normalization

A guard contract narrows a correct provider contract. It should not silently
repair an incomplete one.

When the pinned provider OpenAPI omits or misstates a fact required for
compilation, record authoritative evidence in `evidence.yaml` and apply a
separate `normalization.overlay.yaml` before guard contracts.

```yaml
version: 1
provider: example
primary:
  id: example_openapi
  kind: openapi
  origin: https://example.com/openapi.yaml
  revision: provider-revision
  sha256: <source digest>
supplemental:
  - id: example_streaming_docs
    kind: reviewed_documentation
    origin: https://example.com/docs/streaming
    revision: reviewed-2026-09-22
    claims:
      - POST /chat returns an SSE stream when stream is true
    claims_sha256: <digest of the canonical claims array>
```

Each normalization action cites one declared evidence source:

```yaml
overlay: 1.1.0
info:
  title: Complete the example streaming contract
  version: 1.0.0
actions:
  - target: $.paths['/chat'].post.responses
    description: Add the documented SSE response.
    x-guard-evidence: example_streaming_docs
    update:
      '200':
        content:
          text/event-stream:
            schema:
              $ref: '#/components/schemas/ProviderChatStreamEvent'
            x-guard-source-evidence: reviewed-2026-09-22
```

Evidence kinds are `openapi`, `sdk_types`, and `reviewed_documentation`.
Machine-readable evidence pins the downloaded artifact with `sha256`.
Reviewed documentation pins a canonical list of claims with `claims_sha256`
because a mutable web page cannot be treated as an immutable artifact.
The digest is SHA-256 over the compact, ASCII JSON representation of the claims
array. For example:

```bash
uv run --locked python -c 'import hashlib,json; claims=["POST /chat returns SSE"]; print(hashlib.sha256(json.dumps(claims,separators=(",",":"),ensure_ascii=True).encode()).hexdigest())'
```

Normalization and guard policy have different responsibilities and formats:

```text
normalization Overlay: make the provider contract factually complete
guard contract:        define the supported guarded subset
```

## Reading compiler failures

Semantic failures report three pieces of information:

```text
compile <projection identifier>
<semantic path>
<explanation>
```

Treat the semantic path as the authoring location to inspect. Common failures
mean:

| Failure | Likely cause |
| --- | --- |
| `provider contract has unreviewed fields` | A bound provider object contains fields not classified by the projection. |
| `field is absent from provider contract` | The projection claims a field that the pinned provider schema does not expose; use normalization only with evidence. |
| `does not identify one provider schema branch` | A `type`, `const`, or representation selection is ambiguous or unsupported by the provider union. |
| `requires exactly one matching guarded array variant` | The current capability profile cannot identify one text target. Narrow array cardinality or variant selection. |
| `stream event selection is duplicated` | Two stream variants declare the same selector. Give each native shape a unique selector. |
| runtime reports an ambiguous stream event | Different selectors can still match the same payload. Make the authored rules semantically disjoint. |
| `stream text paths require explicit array cardinality` | A generated stream extraction path crosses an unconstrained array. |
| `must bind its declared stream projection` | Endpoint metadata, stream projection, and handwritten hooks are inconsistent. |

Do not respond to a compiler proof failure by weakening coverage or marking an
unknown content-bearing field opaque without review. The failure is the desired
signal that the guarded boundary changed.

## Validation and compatibility limits

There are three different checks: authoring syntax, provider-aware semantic
validation, and runtime enforcement. A generic JSON Schema validator knows only
standard assertions; it does not enforce subjects, configurable policies,
variant cardinality, review decisions, or hook behavior.

The current frontend requires component-reference operation roots, JSON
requests, HTTP 200 JSON responses, and optional HTTP 200 SSE responses. Only
the documented single-text profiles are supported. Tools, images, audio,
multiple independent guarded messages, structured output, and other semantic
content kinds require explicit shared runtime and compiler capability designs.

Public JSON Schema and runtime acceptance are not identical in every case.
Selected-variant cardinality, optional/null/default handling, provider-owned
requiredness, and disabled gates need separate conformance work. Characterization
tests record concrete cases; artifact reproducibility is not proof of complete
schema/runtime equivalence.

If implementation and this semantic specification disagree, record the mismatch
and add a conformance case; do not silently weaken either definition.

Human authoring trials are still needed to assess readability and editing effort.
The alpha label makes no stable compatibility promise; see the
[version policy](README.md#version-and-stability).

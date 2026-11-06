# Offline provider conformance

These tests compare independent provider specifications and recordings with
typed field policies and observable proxy behavior. Export agreement checks
consistency; it does not establish that a field classification is correct.

## Coverage

| Operation | Evidence and checks |
| --- | --- |
| [Chat Completions](openai/chat_completions/README.md) | Pinned OpenAPI field inventories, source-valid synthetic HTTP cases, and reviewed LiteLLM extensions. |
| [Responses](openai/responses/README.md) | Pinned field inventories, provider examples, native/gateway recordings, and synthetic mutations of recorded bodies. |

Tests use deterministic checker doubles and mocked network I/O. They do not
exercise real Rails configurations, socket delivery, or live-provider behavior.
Guard models are partial content policies, not complete provider validators.
Guarded-request validation still applies when output inspection is disabled.

## Reviewer probes

When extending coverage, check the evidence independently of runtime declarations:

1. Compare source properties and example/recorded keys at each object boundary.
   Account for omissions and native/dialect differences explicitly.
2. Trace additional input, tools, reasoning, refusals, citations, and extension
   fields to inspection or rejection. Opaque values, including nested strings,
   remain uninspected.
3. Check unknown fields, case aliases, repeated snapshots, and request admission
   with output inspection disabled. Reuse recorded/kernel tests for blocking
   and protocol lifecycle checks.
4. Keep provider recordings distinct from synthetic payload mutations and checker
   verdicts. Report a field path, evidence reference, and minimal reproduction.

Pin source provenance and expected outcomes independently. Updating a source or
recording requires reviewing new fields, not deriving expected policies from
the implementation being tested.

## Running tests

```bash
make test WORKERS=0 \
  TEST='tests/server/experimental/conformance' \
  ARGS='--block-network -q'
```

To include related kernel, export, dialect, transport, and recorded tests:

```bash
make test WORKERS=0 \
  TEST='tests/server/experimental tests/recorded/server/experimental' \
  ARGS='--block-network --record-mode=none -q'
```

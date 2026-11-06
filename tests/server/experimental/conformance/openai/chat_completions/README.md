# Chat Completions conformance

This suite audits native Chat field policies and exercises native and explicitly
selected LiteLLM endpoints through the router and HTTP dispatcher.

## Evidence and coverage

- [Pinned OpenAI extract](fixtures/README.md): six Chat operations and their
  reachable schemas, with source provenance and a reproducible extractor.
  `test_schema.py` checks every bound native policy model against the reviewed
  field inventory, including explicit local extensions.
- `test_http.py`: independently source-valid synthetic requests and responses
  covering inspected text, byte preservation, unsupported content, provider
  errors, and buffered/streaming behavior.
- `dialects/test_litellm.py`: synthetic cases based on the
  [reviewed LiteLLM source](../../../../../../nemoguardrails/server/experimental/providers/openai/dialects/litellm/README.md).
  These check explicit selection, closed extension bags, empty-only content
  fields, and usage-only streaming events. Buffered extension permissions do
  not automatically apply to streaming events.

The separate [recorded Chat suite](../../../../../recorded/server/experimental/openai/test_chat_completions.py)
replays gateway exchanges through the LiteLLM endpoint. Its
[inventory tests](../../../../../recorded/server/experimental/openai/test_litellm_inventory.py)
check observed extension fields. Those recordings and the synthetic conformance
cases provide different evidence; neither establishes every backend's behavior.

## Source and routing limits

The source declares OpenAPI 3.1 but uses legacy `nullable` annotations, including
beside enums and references. `support.py` interprets these constructs for
validation without changing fixture bytes or runtime policy. Review that
interpretation when updating the source pin.

The router guards `POST /v1/chat/completions`. Collection `GET` and the four item
operations on `/v1/chat/completions/{id}` and its `/messages` child are unguarded
passthrough. Their tests check transport preservation, not content inspection or
full provider-schema validation.

## Run and extend

```bash
make test WORKERS=0 \
  TEST='tests/server/experimental/conformance/openai/chat_completions' \
  ARGS='--block-network -q'
```

Follow the [shared reviewer probes](../../README.md#reviewer-probes). Keep Chat
error envelopes, `[DONE]`, usage-only chunks, and buffered/streaming extension
rules distinct from other protocols.

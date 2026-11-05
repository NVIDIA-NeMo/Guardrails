# Pinned OpenAI Chat Completions specification

`pinned.openapi.yaml` contains the six Chat operations and their reachable
schemas from the source identified in
`nemoguardrails/server/experimental/contracts/openai/source.yaml`.
It provides independent evidence for offline tests, not runtime policy.
The extractor removes documentation annotations while preserving named
properties and literal schema values.

Source: [OpenAI OpenAPI revision 3e288ba](https://github.com/openai/openai-openapi/blob/3e288ba501d136b81181919a12a88495a0244488/openapi.yaml),
licensed MIT. Original source SHA-256:
`cf91be897ab55e4aefcfc44ec1eb7611eea1227509b67178be583244c2680ab3`.
Extract SHA-256:
`fe22de8fe8bb4a92680c02df3fbbbd1b865706acc33aa7fd6ef03209269cce71`.

See the [Chat conformance guide](../README.md) for test scope and the source's
legacy `nullable` interpretation.

To reproduce, download the descriptor's exact `download_url`, then run:

```bash
uv run --locked python tests/server/experimental/conformance/openai/chat_completions/fixtures/extract_source.py \
  /path/to/openapi.yaml \
  nemoguardrails/server/experimental/contracts/openai/source.yaml \
  tests/server/experimental/conformance/openai/chat_completions/fixtures/pinned.openapi.yaml
```

The extractor verifies the source digest and rejects non-component references.
After a reviewed source or extraction change, update the extract hash here and
in `../test_schema.py`, then run the conformance suite.
New fields require an explicit policy review.

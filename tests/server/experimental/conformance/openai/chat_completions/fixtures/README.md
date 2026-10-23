# Pinned OpenAI Chat Completions specification

`pinned.openapi.yaml` contains the six Chat operations and their reachable
schemas from the source identified in
`nemoguardrails/server/experimental/contracts/openai/source.yaml`.
It provides independent evidence for offline tests, not runtime policy.
The extractor removes documentation annotations while preserving named
properties and literal schema values.

Source: [OpenAI OpenAPI revision df63773](https://github.com/openai/openai-openapi/blob/df63773f69f542ef875b9f00c3837c25ba5f4f2a/openapi.yaml),
licensed MIT. Original source SHA-256:
`f2dae1a9aced09b91310db89edda51bf1e36ecbfb05230c3a50b239c07708469`.
Extract SHA-256:
`a84917786d64f8a04998da6f166b9b293eba20a1f093dd49ac69a1498f78454f`.

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

# Pinned Responses field inventory

`source_inventory.json` contains the named properties of 21 reviewed Responses
components, including inherited `allOf` properties, and the official text-input
response example. It is independent of our typed declarations and generated guard
contracts. It is a field inventory, not a complete provider validation schema.

Source: [OpenAI OpenAPI revision df63773](https://github.com/openai/openai-openapi/blob/df63773f69f542ef875b9f00c3837c25ba5f4f2a/openapi.yaml),
licensed MIT. Original source SHA-256:
`f2dae1a9aced09b91310db89edda51bf1e36ecbfb05230c3a50b239c07708469`.
Inventory SHA-256:
`3166e95194393c2180bca713a440368b4760610fa422692765134401affcbb19`.

The schema omits `store` from `Response`, while its example contains it. Native
recordings additionally establish six response metadata names and delta
`obfuscation` beyond the pinned component fields. Gateway recordings and the
reviewed LiteLLM source establish root SSE `model` as a dialect extension.
These distinctions are explicit test expectations, never generated from runtime
policy. Presence alone does not establish that a field should be opaque.

## Regeneration

From the repository root, download the pinned source and run the extractor:

```bash
curl --fail --location \
  https://raw.githubusercontent.com/openai/openai-openapi/df63773f69f542ef875b9f00c3837c25ba5f4f2a/openapi.yaml \
  --output /tmp/openai-responses-source.yaml
uv run --locked python tests/server/experimental/conformance/openai/responses/fixtures/extract_source.py \
  /tmp/openai-responses-source.yaml \
  nemoguardrails/server/experimental/contracts/openai/source.yaml \
  tests/server/experimental/conformance/openai/responses/fixtures/source_inventory.json
make test TEST=tests/server/experimental/conformance/openai/responses/test_schema.py
```

Extraction verifies the source digest and accepts only local schema references.
The fixture uses LF endings for a stable digest on Windows. Updating the pin
requires reviewing new fields and classification decisions, regenerating the
inventory, and updating its digest here and in `../test_schema.py`.

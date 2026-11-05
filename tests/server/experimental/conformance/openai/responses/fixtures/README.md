# Pinned Responses field inventory

`source_inventory.json` contains the named properties of 22 reviewed Responses
components, including inherited `allOf` properties, and the official text-input
response example. It is independent of our typed declarations and generated guard
contracts. It is a field inventory, not a complete provider validation schema.

Source: [OpenAI OpenAPI revision 3e288ba](https://github.com/openai/openai-openapi/blob/3e288ba501d136b81181919a12a88495a0244488/openapi.yaml),
licensed MIT. Original source SHA-256:
`cf91be897ab55e4aefcfc44ec1eb7611eea1227509b67178be583244c2680ab3`.
Inventory SHA-256:
`2b5e5a4967f843baf7807bee8795ecb4e836056ef569894dade378c55a40ad5c`.

The schema omits `store` from `Response`, while its example contains it. Native
recordings additionally establish `billing`, `frequency_penalty`,
`presence_penalty`, and delta `obfuscation` beyond the pinned component fields.
`image`, `access_programs`, and `tool_usage` are now declared in the source.
The image object's fields are inventoried through `ResolvedImageOptions`.
Gateway recordings and the
reviewed LiteLLM source establish root SSE `model` as a dialect extension.
These distinctions are explicit test expectations, never generated from runtime
policy. Presence alone does not establish that a field should be opaque.

The source-defined request controls `image` and `access_programs` remain
unsupported. The coverage audit accounts for them explicitly, and HTTP tests
verify rejection before dispatch in every inspection mode. The source document
still reports version `2.3.0`; the revision and digest identify this refresh.

## Regeneration

From the repository root, download the pinned source and run the extractor:

```bash
curl --fail --location \
  https://raw.githubusercontent.com/openai/openai-openapi/3e288ba501d136b81181919a12a88495a0244488/openapi.yaml \
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

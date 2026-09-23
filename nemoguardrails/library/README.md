# NeMo Guardrails Library

The library contains a set of pre-built rails that can be activated in any config.

## Contributing A New Rail

When adding a new rail under `nemoguardrails/library/<feature>/`, you must also
add recorded e2e coverage so the rail's public contract is pinned and replayable
without external services. See
[`tests/recorded/rails/library/README.md`](../../tests/recorded/rails/library/README.md)
for the required cases, cassettes, and configs.

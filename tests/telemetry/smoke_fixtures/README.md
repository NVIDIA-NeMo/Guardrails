# Smoke-test fixture configs

Minimal `RailsConfig` directories consumed by [`scripts/telemetry_smoke.py`](../../../scripts/telemetry_smoke.py) as a stable, version-controlled config root for the server and multi-config scenarios.

| Dir | Rails | Why |
|---|---|---|
| `cfg1` | input only, `self check input` | distinguishable `railTypesInUse=["input"]` |
| `cfg2` | output only, `self check output` | distinguishable `railTypesInUse=["output"]` |
| `cfg3` | input + output, both `self check` flows | distinguishable `railTypesInUse=["input","output"]` |
| `rich` | input + output, tracing, streaming, kb, custom flow | exercises telemetry fields the minimal fixtures intentionally leave false or zero |
| `feature_aliases` | config-only fact-checking, Patronus, and regex settings | proves config-derived `builtinFeatures` use documented IDs (`factchecking`, `patronusai`, `regex`) |
| `v2_custom_flow` | Colang 2.x config importing `core` plus one user flow | proves bundled v2 library flows are excluded from `numCustomFlows` |

All fixtures use the same main model declaration (`openai / gpt-4o-mini`) so multi-config merges do not trip the model-conflict guard in `_join_rails_configs`. The configs are valid enough for `RailsConfig.from_path()` to parse and for `LLMRails(...)` to construct, but the smoke driver never makes a real LLM call: `LLMRails.__init__` emits telemetry before any `generate_async`, so the wire shape is exercised without needing an OpenAI key.

If you tweak these, keep them minimal and self-contained. They are not production configs; if you want example bots, see `examples/bots/`.

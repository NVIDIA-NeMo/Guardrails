# Experimental proxy tests

- Exercise payload validation, inspection, and streaming under both `allow` and
  `forbid`. Let shared proxy/client fixtures consume the `unknown_fields` fixture
  and pass it into construction; tests need the policy directly only when their
  setup or assertions depend on it. Do not override production defaults.
- Keep unrelated configuration, routing, and resource-lifecycle checks single-run.
- Keep focused unknown-field cases with explicit expected outcomes for each
  policy, and separate tests that omit the setting to verify production defaults.
  Contract inventory and schema conformance checks remain strict.
- For recorded coverage, capture identical upstream traffic once and replay it
  under both policies. Follow the recorded-tests skill and
  [recorded-test guidance](../../recorded/AGENTS.md). Strict cassette auditing
  remains independent of the runtime policy.

# AGENTS.md

Guidance for agents editing files under `tests/recorded/`.

- Placement rule: a test belongs in this suite only if the scenario sends at
  least one real provider request during the test. If nothing crosses the
  wire (injected `generator=`, local test actions, input validation), write
  a unit test in `tests/` instead. Do not pattern-match on neighboring
  non-vcr tests; they are documented exceptions.
- Every suite test must carry `@pytest.mark.vcr` or, for the rare public-API
  contract exception, `@pytest.mark.pure_runtime(reason="...")`; this is
  enforced at test setup. If the marker feels forced, the test belongs in
  `tests/`.
- Recorded tests are dialect-single: do not duplicate a test for Colang 2.
- A vcr test without a committed cassette is incomplete: record with
  `make record-cassettes`, fill snapshots offline, verify replay with
  `--block-network`.
- `README.md` in this directory owns the mechanics (markers, refresh,
  cassettes, snapshots). The `recorded-tests` skill under `.agents/skills/`
  owns the placement decision procedure. Do not restate either here.

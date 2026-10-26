# AGENTS.md

Guidance for agents editing files under `tests/recorded/`.

- `README.md` in this directory owns the mechanics (markers, refresh,
  cassettes, snapshots). The `recorded-tests` skill under `.agents/skills/`
  owns the placement decision procedure, including the real-provider-request
  criterion. Do not restate either here.
- The suite intentionally mixes vcr and non-vcr tests. Follow the README's
  marker rules and the skill's exception criteria; suite-infrastructure tests
  are not templates for provider replay tests.
- Nothing enforces that split. `conftest.py` has no marker-inspecting
  fixture and `pytest.ini` registers no marker for the exception, so a
  misplaced deterministic test collects and passes here silently. Apply the
  skill's criteria yourself, and do not pattern-match on the non-vcr
  neighbours.
- LLMRails library recordings are dialect-single: do not duplicate identical
  provider traffic for Colang 2. Public-API v2 recordings are valid when the
  different runtime produces different provider traffic. IORails parity may
  reuse sibling cassettes.
- A test that actually sends or replays provider traffic without a committed
  cassette is incomplete. A module-level `vcr` mark can also cover a deliberate
  deterministic no-cassette parity test, so inspect the test's cassette
  routing. For a new empty snapshot, follow the README's explicit record,
  create, and offline-replay sequence.

# AGENTS.md

Guidance for agents editing files under `tests/recorded/`.

- `README.md` in this directory owns the mechanics (markers, refresh,
  cassettes, snapshots). The `recorded-tests` skill under `.agents/skills/`
  owns the placement decision procedure, including the real-provider-request
  criterion. Do not restate either here.
- Most suite tests carry `@pytest.mark.vcr`. A few do not: `test_regex.py`
  and `test_injection.py` under `rails/library/`, plus the cassette
  meta-tests in this directory. Those carry only the module-level
  `pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]`.
- Nothing enforces that split. `conftest.py` has no marker-inspecting
  fixture and `pytest.ini` registers no marker for the exception, so a
  misplaced deterministic test collects and passes here silently. Apply the
  skill's criteria yourself, and do not pattern-match on the non-vcr
  neighbours.
- Recorded tests are dialect-single: do not duplicate a test for Colang 2.
- A vcr test without a committed cassette is incomplete: record with
  `make record-cassettes`, fill snapshots offline, verify replay with
  `--block-network`.

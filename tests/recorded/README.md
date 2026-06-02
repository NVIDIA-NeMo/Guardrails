# Recorded Tests

Recorded tests replay provider traffic through pytest-recording cassettes and must run without live network access by default.

## Adding a test

Markers are applied once per module via `pytestmark`; do not stack `@pytest.mark.recorded` / `vcr` / `asyncio` on each test. Use a module-level list, and fold in `vcr`/`asyncio` only when every test in the module needs them:

```python
pytestmark = [pytest.mark.recorded, pytest.mark.vcr, pytest.mark.asyncio]


async def test_my_case(openai_api_key):
    rails = LLMRails(load_config(OPENAI_BASELINE_CONFIG), verbose=False)
    result = await rails.generate_async(prompt="...")
    assert result == snapshot()
```

Request credentials as fixture parameters (`openai_api_key`, `nvidia_api_key`) rather than calling `request.getfixturevalue(...)`. In modules that mix sync/async tests or vcr/non-vcr tests, keep only `recorded` in `pytestmark` and apply `@pytest.mark.vcr` / `@pytest.mark.asyncio` per test. Then record once with credentials, fill the snapshot offline, and verify the replay:

```bash
OPENAI_API_KEY=... poetry run pytest path::test_my_case --record-mode=all
poetry run pytest path::test_my_case --block-network --inline-snapshot=create
poetry run pytest path::test_my_case --block-network
```

## Negative paths

This suite owns **pipeline-level** failures (how `LLMRails` behaves when a model call
fails) and **public-API input validation**. Client/wire-level conditions (status code to
exception mapping, retries, SSE, malformed bodies) belong in `tests/llm/clients/`, which
covers them with `httpx.MockTransport` + JSON fixtures; do not duplicate them here.

Prefer mechanisms in this order:

1. **Recordable real error** (refreshable cassette). A nonexistent model name yields a real,
   deterministic 404, so error paths record and refresh like any happy path. Use a config
   whose model is invalid (see `OPENAI_INVALID_MODEL_CONFIG`,
   `CONTENT_SAFETY_INVALID_MODEL_CONFIG`).
2. **Pure runtime** `pytest.raises` for input validation (no cassette, no transport).
3. **Fake cassette** (`@pytest.mark.fake_cassette`) only as a last resort, for a synthetic
   response that must flow through the full pipeline and cannot be reproduced by 1 or 2.

Observed behavior these tests pin: a failing model call (main *or* a rail's own model)
propagates as `LLMCallException` — a safety-model failure does not let content through
silently. Name negative tests `test_<surface>_<failure>_<behavior>` with the suffixes
`_raises` / `_fails_closed` / `_invalid_*` so they are greppable
(`pytest -k "raises or invalid"`), and co-locate each with its happy-path sibling module.

## Replay

```bash
poetry run pytest tests/recorded --block-network -v --durations=10
```

Focused rails replay:

```bash
poetry run pytest tests/recorded/rails/public_api --block-network -v
poetry run pytest tests/recorded/rails/library --block-network -v
```

Replay mode installs dummy API keys from `tests/recorded/utils.py`. A cassette miss with `--block-network` is a test failure.

## Refresh

Refresh only in a trusted environment with real provider credentials. The full
record -> fill-snapshots -> verify loop is wrapped in a make target:

```bash
OPENAI_API_KEY=... NVIDIA_API_KEY=... make record-tests
```

Or run the recording step alone:

```bash
poetry run pytest tests/recorded --record-mode=all -m "not fake_cassette" -v
```

For a focused rewrite:

```bash
poetry run pytest tests/recorded/rails/public_api/test_generate.py::test_openai_generate_async_public_contract --record-mode=rewrite -v
```

The refresh workflow uploads cassettes as artifacts for review and does not commit them.

## Cassettes

Rails tests use pytest-recording's default names:

```text
tests/recorded/rails/<suite>/cassettes/<test_module>/<test_name>.yaml
```

Parameterized tests include the parameter id in the cassette filename. Every test (rails and clients) uses this default naming; do not add `@pytest.mark.default_cassette(...)`.

JSON request and response bodies are stored as `parsed_body` and rehydrated by `ReadableYamlSerializer` during replay. SSE responses also use parseable `parsed_body` events.

At serialize time the bodies are normalized to ASCII (smart quotes, en/em dashes, the hyphen family, and ellipsis are folded, then NFKC) so cassettes and inline snapshots stay stable and portable. Response headers are dropped by exact name and by prefix (`x-`, `cf-`, `openai-`); `tests/recorded/sanitization.py` holds the `ALLOWED_HEADERS` exceptions that must survive the prefix sweep (currently `content-type`).

Inspect a cassette:

```bash
poetry run python -m tests.recorded.inspect_cassette tests/recorded/rails/public_api/cassettes/test_stream/test_openai_stream_async_public_contract.yaml
```

## Snapshots

Rails replay outputs are pinned with inline snapshots after normalization. Create or fix snapshots with:

```bash
poetry run pytest tests/recorded/rails --block-network --inline-snapshot=create
poetry run pytest tests/recorded/rails --block-network --inline-snapshot=fix
poetry run pytest tests/recorded/rails --block-network --inline-snapshot=review
```

Snapshot formatting uses `ruff format` through `[tool.inline-snapshot]` in `pyproject.toml`.

Volatile response fields (ids, timestamps, fingerprints) are scrubbed to fixed sentinels in the cassette, so snapshots assert them directly without needing loose matchers.

## Fake Outputs

Prefer `FakeLLMModel` when a test needs the main model to emit a specific output and provider-backed rail/task calls can still replay from VCR. This keeps the test refreshable.

Use a fake cassette only when runtime injection cannot model the behavior clearly, such as a provider stream/error path. Fake cassettes must:

- live under a `cassettes/**/fake/` directory,
- use `@pytest.mark.fake_cassette`,
- be excluded from refresh with `-m "not fake_cassette"`,
- include YAML header metadata with `reason`, `frozen_fields`, and `fake_llm_model_considered`.

The fake-cassette metadata validator is in `tests/recorded/fake_cassettes.py`.

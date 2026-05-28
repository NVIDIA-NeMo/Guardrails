# Recorded Tests

Recorded tests replay provider traffic through pytest-recording cassettes and must run without live network access by default.

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

Refresh only in a trusted environment with real provider credentials:

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

Parameterized tests include the parameter id in the cassette filename. Client adapter tests may use `@pytest.mark.default_cassette(...)` for stable cassette stems, but still keep cassettes under the module-specific directory.

JSON request and response bodies are stored as `parsed_body` and rehydrated by `ReadableYamlSerializer` during replay. SSE responses also use parseable `parsed_body` events.

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

## Fake Outputs

Prefer `FakeLLMModel` when a test needs the main model to emit a specific output and provider-backed rail/task calls can still replay from VCR. This keeps the test refreshable.

Use a fake cassette only when runtime injection cannot model the behavior clearly, such as a provider stream/error path. Fake cassettes must:

- live under a `cassettes/**/fake/` directory,
- use `@pytest.mark.fake_cassette`,
- be excluded from refresh with `-m "not fake_cassette"`,
- include YAML header metadata with `reason`, `frozen_fields`, and `fake_llm_model_considered`.

The fake-cassette metadata validator is in `tests/recorded/fake_cassettes.py`.

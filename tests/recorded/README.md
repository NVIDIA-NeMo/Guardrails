# Recorded Tests

Recorded tests replay provider traffic through pytest-recording cassettes and must run without live network access by default.

## Adding a test

Markers are applied once per module via `pytestmark`; do not stack `@pytest.mark.recorded` / `vcr` / `asyncio` on each test. Use a module-level list, and fold in `vcr`/`asyncio` only when every test in the module needs them:

```python
import pytest

from nemoguardrails import LLMRails
from tests.recorded.rails.public_api.configs import OPENAI_BASELINE_CONFIG
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.vcr, pytest.mark.asyncio]


async def test_my_case(openai_api_key):
    rails = LLMRails(load_config(OPENAI_BASELINE_CONFIG), verbose=False)
    result = await rails.generate_async(prompt="...")
    assert result == snapshot()
```

Config constants (``OPENAI_BASELINE_CONFIG`` and friends) live in the suite-local
``configs.py`` next to the tests; ``snapshot`` is the suite-local re-export in
``tests/recorded/snapshots.py``, not ``inline_snapshot`` directly.

Request credentials as fixture parameters (`openai_api_key`, `nvidia_api_key`,
or `nvidia_inference_key`) rather than calling
`request.getfixturevalue(...)`. In modules that mix sync/async tests or
vcr/non-vcr tests, keep only `recorded` in `pytestmark` and apply
`@pytest.mark.vcr` / `@pytest.mark.asyncio` per test. Then use the Makefile
refresh workflow to record once with credentials, fill the snapshot offline,
and verify the replay:

```bash
OPENAI_API_KEY=... make record-cassettes \
  RECORDED_TESTS=path::test_my_case \
  RECORDED_REQUIRED_KEYS=OPENAI_API_KEY
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

### Tool-calling coverage

The tool-calling modules under `rails/public_api/` and `server/` record the default OpenAI-compatible framework with Colang 1.0 and no user-intent definitions. Tool conversations use `passthrough: false`; an additional ordinary conversation checks sanitized input history with `passthrough: true`. These recordings establish the OpenAI path; they do not establish equivalent behavior for every provider or for streaming.

| Scenario | Recorded contract |
| --- | --- |
| Default text prompt, one tool | Native assistant/tool roles, matching IDs, result used in the final answer |
| Custom chat prompt through HTTP | OpenAI argument serialization, tool finish reason, final answer |
| Parallel tools with input, result, and output rails | Input rewrite, two calls, reversed result order, sanitization, final output processing, later-turn history; `single_call` enabled without intents |
| Sequential dependent tools | A second tool uses a token returned by the first; both results reach the final answer |
| Tool-call rejection | Provider actually emits a call; the public response contains only the refusal |
| Server thread history | A result-only follow-up and later user turn recover stored tool calls and results |
| Provider error on a tool follow-up | Library raises `LLMCallException`; server preserves the current OpenAI-compatible error envelope and HTTP status |
| Passthrough input sanitization across turns | The provider receives the sanitized earlier user message on both turns, while the caller's history remains unchanged |

Deterministic checks in `tests/test_tool_calling_config_matrix.py`, `tests/test_tool_calling_passthrough_integration.py`, and `tests/server/test_api.py` cover both passthrough modes, text/chat prompts, allow/sanitize/block/disabled result rails, every result in successive batches, caller-message immutability, cache reuse across generation options and wire formats, approved call history, prompt-length boundaries, and unsupported server configurations. LangChain tests also require rejected calls to be omitted. Local validation and rail decisions do not need fabricated provider recordings.

Reused tool-call IDs across turns are covered by deterministic tests: each occurrence retains its own sanitized or blocked result. Filter tests also cover unchecked occurrences and message suffixes, so an earlier rail outcome cannot overwrite a later unvalidated result or vice versa.

`tests/test_history_cache_regressions.py` also checks evolving context, explicit context updates, isolation between enabled and bypassed input rails, and sanitized user history across ordinary, tool, and blocked turns. Cache keys distinguish rail policies and multimodal inputs even when their text matches.

### Commands

```bash
uv run pytest tests/recorded --block-network -v --durations=10
```

Focused rails replay:

```bash
uv run pytest tests/recorded/rails/public_api --block-network -v
uv run pytest tests/recorded/rails/library --block-network -v
```

Replay mode installs dummy API keys from `tests/recorded/utils.py`. A cassette miss with `--block-network` is a test failure.

## Refresh

Refresh only in a trusted environment with real provider credentials. The
record -> fill-snapshots -> verify loop is wrapped in make targets.

```bash
OPENAI_API_KEY=... NVIDIA_API_KEY=... make record-cassettes
```

`record-cassettes` defaults to `RECORDED_RECORD_MODE=once` and
`RECORDED_SNAPSHOT_MODE=create`, which records missing cassettes, fills empty
snapshots, and replays existing cassettes. This is the safest mode when adding
new tests because it will not rewrite unrelated existing cassettes selected by
the same path.

For a focused new cassette that only touches one provider, pass the test node and
override the preflight list:

```bash
OPENAI_API_KEY=... make record-cassettes \
  RECORDED_TESTS=tests/recorded/rails/public_api/test_generate.py::test_new_case \
  RECORDED_REQUIRED_KEYS=OPENAI_API_KEY
```

The OpenAI Chat proxy recordings use the OpenAI protocol through NVIDIA's
inference endpoint and explicitly bind the
[LiteLLM dialect](../../nemoguardrails/server/experimental/providers/openai/dialects/litellm/README.md).
The default OpenAI endpoint is not silently widened to accept this gateway.
Refresh them with the dedicated credential and no OpenAI key:

```bash
NVIDIA_INFERENCE_KEY=... make record-cassettes \
  RECORDED_TESTS=tests/recorded/server/experimental/openai/test_chat_completions.py \
  RECORDED_REQUIRED_KEYS=NVIDIA_INFERENCE_KEY
```

Replay the Chat pipeline and the archived-field inventory together:

```bash
make replay-cassettes RECORDED_TESTS=tests/recorded/server/experimental/openai
```

The [proxy integration guide](server/experimental/README.md) describes the
allow/block scenarios, reusable checker helper, single-use replay assertions,
and the pattern for another operation. These tests combine recorded provider
exchanges with synthetic checker decisions, not real Rails configuration.
Replayed-body equality is not a claim of original provider-wire byte fidelity
or live streaming delivery. Real-config and socket coverage is separate.

These six cassettes predate the dialect: their provider payloads are unchanged.
Two buffered successes contain message `provider_specific_fields: {refusal: null}`
and empty choice extensions. Two streaming successes contain no extension bags;
the other two responses are provider errors. The inventory tests validate every
successful payload against runtime declarations and export and pin those evidence
limits. The recordings do not establish the serving LiteLLM version or support
for arbitrary backend fields. Native finish metadata and empty stream containers
are covered by separately identified, source-backed unit fixtures, not invented
recordings. A refresh must review new fields rather than modifying payloads to
fit existing expectations.

`RECORDED_TESTS` is passed directly to pytest, so it can be a single test, a
test class, several files, or a directory:

```bash
OPENAI_API_KEY=... make record-cassettes \
  RECORDED_TESTS="tests/recorded/rails/public_api/test_generate.py tests/recorded/clients/test_openai_chat.py" \
  RECORDED_REQUIRED_KEYS=OPENAI_API_KEY
```

For an intentional rewrite of existing cassettes, use `rewrite-cassettes`:

```bash
OPENAI_API_KEY=... make rewrite-cassettes \
  RECORDED_TESTS=tests/recorded/rails/public_api/test_generate.py::test_openai_generate_async_public_contract \
  RECORDED_REQUIRED_KEYS=OPENAI_API_KEY
```

`rewrite-cassettes` uses `RECORDED_RECORD_MODE=rewrite` and
`RECORDED_SNAPSHOT_MODE=fix`, so changed recorded outputs update existing inline
snapshots before the final offline replay verification.

For a full trusted refresh, set the record mode explicitly:

```bash
OPENAI_API_KEY=... NVIDIA_API_KEY=... make record-cassettes RECORDED_RECORD_MODE=all RECORDED_SNAPSHOT_MODE=fix
```

Replay and snapshot-only workflows do not need real provider credentials:

```bash
make replay-cassettes RECORDED_TESTS=tests/recorded/rails/public_api/test_generate.py::test_openai_generate_async_public_contract
make snapshot-cassettes RECORDED_TESTS=tests/recorded/rails/public_api/test_generate.py::test_openai_generate_async_public_contract
```

## Cassettes

Rails tests use pytest-recording's default names:

```text
tests/recorded/rails/<suite>/cassettes/<test_module>/<test_name>.yaml
```

Experimental server proxy recordings are grouped by provider:

```text
tests/recorded/server/experimental/<provider>/cassettes/<test_module>/<test_name>.yaml
```

Parameterized tests include the parameter id in the cassette filename. Every test (rails and clients) uses this default naming; do not add `@pytest.mark.default_cassette(...)`.

JSON request and response bodies are stored as `parsed_body` and rehydrated by `ReadableYamlSerializer` during replay. SSE responses also use parseable `parsed_body` events.

Cassettes preserve scrubbed JSON text without smart-character normalization so provider payloads stay inspectable. Request matching and snapshot helpers normalize smart quotes, dash variants, ellipses, and NFKC at comparison time. Response headers are dropped by exact name and by prefix (`x-`, `cf-`, `openai-`); `tests/recorded/sanitization.py` holds the `ALLOWED_HEADERS` exceptions that must survive the prefix sweep (currently `content-type`).

Inspect a cassette:

```bash
uv run python -m tests.recorded.inspect_cassette tests/recorded/rails/public_api/cassettes/test_stream/test_openai_stream_async_public_contract.yaml
```

## Snapshots

Rails replay outputs are pinned with inline snapshots after normalization. Create or fix snapshots with:

```bash
uv run pytest tests/recorded/rails --block-network --inline-snapshot=create
uv run pytest tests/recorded/rails --block-network --inline-snapshot=fix
uv run pytest tests/recorded/rails --block-network --inline-snapshot=review
```

Snapshot formatting uses `ruff format` through `[tool.inline-snapshot]` in `pyproject.toml`.
Snapshot create/fix/review runs must be serial. Use `make record-cassettes` or a
direct `uv run pytest ... --inline-snapshot=<mode>` command; the default
`make test` path uses xdist, where inline-snapshot disables update and report
modes.

Volatile response fields (ids, timestamps, fingerprints) are scrubbed to fixed sentinels in the cassette, so snapshots assert them directly without needing loose matchers.

## Fake Outputs

Prefer `FakeLLMModel` when a test needs the main model to emit a specific output and provider-backed rail/task calls can still replay from VCR. This keeps the test refreshable.

Use a fake cassette only when runtime injection cannot model the behavior clearly, such as a provider stream/error path. Fake cassettes must:

- live under a `cassettes/**/fake/` directory,
- use `@pytest.mark.fake_cassette`,
- be excluded from refresh with `-m "not fake_cassette"`,
- include YAML header metadata with `reason`, `frozen_fields`, and `fake_llm_model_considered`.

The fake-cassette metadata validator is in `tests/recorded/fake_cassettes.py`.

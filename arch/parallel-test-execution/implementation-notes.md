# Implementation Notes

## Command Surface

`Makefile` exposes:

- `warm_fastembed_cache`
- `test_parallel`

`test_parallel` depends on `warm_fastembed_cache` so xdist workers do not race
on first-time FastEmbed cache creation.

The local command also unsets `OPENAI_API_KEY` and `NVIDIA_API_KEY`. This keeps
normal local tests from changing behavior when real provider credentials happen
to be present in the shell.

## Isolation Fixes

The xdist rollout exposed several existing order and process-local assumptions.
The base branch fixes these before making parallel execution the recommended
local path:

- `tests/test_content_safety_actions.py` uses sorted parametrization for
  `SUPPORTED_LANGUAGES`, so xdist workers collect identical node ids.
- `tests/conftest.py` resets `explain_info_var` around each test, preventing
  worker-local explain-log leakage.
- Explain-log assertions in affected tests are relative to calls made by the
  current test rather than absolute process-local positions.
- `tests/guardrails/test_request_id.py` resets request-id context around each
  test.
- `tests/guardrails/test_iorails.py` uses an async fixture that stops
  `IORails` before the event loop is torn down.
- Server fixture configs register their `custom_llm` provider directly instead
  of relying on unrelated tests to do it first.
- `tests/test_testing_chat_harness.py` verifies that constructing a second
  `TestChat` does not replace another `TestChat` instance's active explain
  context.

## Offline Cache Fixes

Two model-cache paths needed explicit handling:

- FastEmbed: warm once before xdist workers start and use
  `.cache/fastembed`.
- fast-langdetect: use the bundled `lite` model for content-safety language
  detection, avoiding a cold network download of the full model during tests.

## Slow-Test Profile

The full suite still has intentional timing tests. These are useful behavioral
tests, but they cap full-suite speedup:

- `tests/test_parallel_rails.py` verifies delayed input/output rails execute in
  parallel.
- `tests/test_parallel_streaming_output_rails.py` compares sequential and
  parallel streaming output rails.
- `tests/guardrails/test_async_work_queue.py` verifies queue behavior,
  cancellation, and worker lifecycle transitions.
- `tests/test_cache_lfu.py` verifies cache stats timing and thread-safety.

The slow-test cleanup follow-up reduced several artificial sleeps while
preserving the behavior under test:

- parallel rail delay: `1.0s` to `0.2s`
- streaming performance action delay: `100ms` to `75ms`
- LFU stats logging: real sleeps replaced with mocked time
- selected async queue waits: one-second sleeps replaced with event-based
  synchronization

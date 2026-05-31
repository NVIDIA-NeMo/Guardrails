# Follow-Up Plan

These follow-ups are intentionally separate from the base local xdist PR.

## 1. Fast Local Loop

Branch:

```text
feat/parallel-tests-fast
```

Commit:

```text
test: add fast parallel test target
```

Scope:

- Add `make test_fast`.
- Run a curated subset with xdist.
- Exclude `slow` and `live`.
- Mark timing-heavy suites as `slow`.

Validation:

```text
make test_fast
528 passed, 6 skipped in 7.88s
```

## 2. CI Non-Coverage Parallelization

Branch:

```text
ci/parallel-non-coverage-tests
```

Commit:

```text
ci: run non-coverage tests with xdist
```

Scope:

- Use xdist for reusable workflow jobs where `with-coverage == false`.
- Keep coverage jobs unchanged.
- Warm FastEmbed before worker startup.
- Keep provider keys sanitized.

## 3. CI Coverage Parallelization

Branch:

```text
ci/parallel-coverage-tests
```

Commit:

```text
ci: run coverage tests with xdist
```

Scope:

- Run coverage jobs with xdist after the non-coverage CI change is stable.
- Keep coverage XML output at `coverage.xml`.
- Verify `pytest-cov` combines worker coverage correctly.

Local smoke validation:

```text
pytest -n 2 --dist worksteal --cov=nemoguardrails tests/test_utils.py
17 passed in 5.59s
Coverage XML written
```

## 4. Slow-Test Runtime Cleanup

Branch:

```text
test/reduce-slow-test-runtime
```

Commit:

```text
test: reduce timing-heavy test runtime
```

Scope:

- Reduce artificial parallel rail delays.
- Use mocked time for LFU cache stats timing.
- Replace selected fixed async queue sleeps with event-based synchronization.
- Reduce streaming performance sleeps while keeping a measurable
  sequential-versus-parallel signal.

Validation:

```text
make test_parallel PYTEST_ARGS='--tb=short -q --durations=25'
3909 passed, 164 skipped in 34.76s
```

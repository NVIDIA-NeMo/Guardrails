# Decision Log

## Local Parallelism

Decision: introduce `pytest-xdist` as an opt-in local command before changing
CI defaults.

Rationale:

- The suite already uses pytest.
- Process workers isolate module state better than thread-based test
  parallelism.
- The first rollout needed to surface and fix order dependencies before CI
  could depend on parallel execution.

## Default Worker Count

Decision: default local `test_parallel` to `8` workers with `--dist worksteal`.

Rationale:

- `worksteal` balances the long tail better than fixed file ordering for this
  suite.
- On the local machine, all detected cores were not necessarily the best
  wall-clock choice because collection, imports, CPU contention, and process
  overhead are material.
- `8` workers gave a stable full-suite result around `35s` to `40s`.

## Cache Warmup

Decision: make `test_parallel` warm FastEmbed before starting xdist workers.

Rationale:

- Multiple workers can race when the FastEmbed model is first created in a cold
  cache.
- A single pre-worker warmup is simpler and more deterministic than letting
  workers discover and initialize the cache independently.

## Live Provider Keys

Decision: unset `OPENAI_API_KEY` and `NVIDIA_API_KEY` for the default local
parallel command.

Rationale:

- The normal local test suite should not change behavior based on real
  provider credentials in a developer shell.
- Live/provider-backed behavior should be isolated behind explicit test modes
  or separate jobs.

## Fast Test Target

Decision: keep `make test_fast` as a follow-up rather than include it in the
base xdist PR.

Rationale:

- `test_parallel` is full-suite validation.
- `test_fast` is a curated feedback loop and should be reviewed as a separate
  scope.
- The validated follow-up target completed in `7.88s` with `528 passed` and
  `6 skipped`.

## CI Rollout

Decision: split CI adoption into non-coverage first, then coverage.

Rationale:

- Non-coverage CI can validate xdist scheduling without also changing coverage
  report generation.
- Coverage should move only after verifying `pytest-cov` combines reports
  correctly under xdist.

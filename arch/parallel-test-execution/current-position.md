# Current Position

The test suite now has an opt-in local pytest worker-parallel path through
`make test_parallel`.

The command runs the normal test tree with `pytest-xdist`, sanitizes live
provider keys, and warms the repo-local FastEmbed cache before workers start:

```bash
make test_parallel PYTEST_ARGS='--tb=short -q'
```

The current default shape is:

```text
pytest -n 8 --dist worksteal tests/
```

with:

- `PYTEST_WORKERS ?= 8`
- `PYTEST_DIST ?= worksteal`
- `FASTEMBED_CACHE_PATH ?= .cache/fastembed`
- `OPENAI_API_KEY` and `NVIDIA_API_KEY` unset for normal local test runs

The latest full-suite validation on the base branch passed:

```text
3909 passed, 164 skipped in 39.48s
```

After the slow-test cleanup follow-up, the full parallel suite passed in:

```text
3909 passed, 164 skipped in 34.76s
```

## Performance Bound

Using the measured serial baseline of `168.29s`, the best observed full-suite
parallel runtime of `35.30s` is:

```text
168.29 / 35.30 = 4.77x faster
```

That is about a `79.0%` wall-clock reduction:

```text
(168.29 - 35.30) / 168.29 = 0.790
```

With 8 workers, the perfect lower bound is:

```text
168.29 / 8 = 21.04s
```

That bound assumes perfect balancing and zero overhead, which xdist cannot
achieve for this suite. Workers still pay for startup, imports, pytest
collection, fixture setup, shared I/O, cache work, and uneven test durations.

The best measured worker efficiency is about `60%`:

```text
168.29 / (8 * 35.30) = 0.596
```

The slow-test cleanup validation at `34.76s` is also about `60%` efficient:

```text
168.29 / (8 * 34.76) = 0.605
```

The base branch validation at `39.48s` is about `53%` efficient:

```text
168.29 / (8 * 39.48) = 0.533
```

So the realistic full-suite local result is roughly `4.3x` to `4.8x` faster.
A sub-10-second complete suite is not achievable through xdist tuning alone
with 8 local workers. It would require reduced scope, many more effective
workers, or CI sharding across machines.

## Recommended Commands

Full local validation:

```bash
make test_parallel PYTEST_ARGS='--tb=short -q'
```

Specific file or subset:

```bash
make test_parallel TEST_FILE=tests/test_utils.py
make test_parallel TEST_FILE='tests/test_utils.py tests/test_cache_utils.py'
```

Worker and distribution tuning:

```bash
make test_parallel PYTEST_WORKERS=4
make test_parallel PYTEST_DIST=loadscope
```

Fast local loop follow-up:

```bash
make test_fast
```

`test_fast` is intentionally a curated subset, not a full-suite replacement.
The validated follow-up result was:

```text
528 passed, 6 skipped in 7.88s
```

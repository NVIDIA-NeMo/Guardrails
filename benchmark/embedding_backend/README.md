# Embedding-backend benchmark

`bench_embedding_backend.py` compares embedding nearest-neighbor backends for the
default `BasicEmbeddingsIndex`:

- **`numpy`** — exact cosine search over an L2-normalized matrix (the current default).
- **`annoy`** — the previous default (approximate, `metric=angular`, `n_trees=10`),
  benchmarked only if `annoy` is installed (it is no longer a project dependency).

It provides reproducible benchmark evidence for replacing Annoy with exact NumPy
search as the default backend. It is not a user migration guide: existing default
knowledge-base caches are regenerated automatically, and threshold semantics are
preserved by the implementation.

## What it measures

For each `N` (number of indexed vectors) and each backend:

| Metric | Meaning |
|---|---|
| `build (ms)` | time to construct the index from precomputed embeddings |
| `search p50 / p95 (ms)` | per-query latency percentiles over `--queries` random queries |
| `recall@1`, `recall@k` | fraction of the **exact** cosine top-k that the backend returns |
| `mem (MB)` | best-effort resident-set delta during build (Linux `/proc`) |

**Methodology.** Both backends receive *identical* L2-normalized float32 vectors, so the
embedding model is held constant and only the index varies. Ground truth is the exact
cosine top-k computed with a full matrix product. `recall@k` therefore measures a backend's
*approximation loss* against the exact answer — the NumPy backend is exact, so its recall is
1.0 by construction; Annoy's recall reflects what its `n_trees=10` configuration gives up.

The vectors are deterministic (fixed `--seed`) and lightly clustered so neighborhoods are
non-trivial. Absolute recall depends on the data distribution; the exact backend is 1.0
regardless.

## Prerequisites

Run inside the project's Poetry environment:

```bash
poetry install --with dev
```

> **Note (Annoy native build).** Annoy is a C++ extension. On a machine without a prebuilt
> wheel, installing it requires a compiler and the Python headers, e.g. on Debian/Ubuntu:
>
> ```bash
> sudo apt-get install -y g++ python3-dev
> ```
>
> This native build requirement is one of the reasons Annoy was dropped. The NumPy backend
> has no such requirement.

## Running

```bash
# Default sweep: N = 100, 1k, 10k, 100k
poetry run python benchmark/embedding_backend/bench_embedding_backend.py

# Quick subset
poetry run python benchmark/embedding_backend/bench_embedding_backend.py --sizes 100 1000 --queries 100

# Flags
#   --sizes    list of index sizes (default: 100 1000 10000 100000)
#   --dim      embedding dimension (default: 384, = all-MiniLM-L6-v2)
#   --k        neighbors to retrieve (default: 20, = BasicEmbeddingsIndex.search default)
#   --queries  number of query vectors timed (default: 200)
#   --seed     RNG seed for reproducibility (default: 1234)
```

### Reproducing the Annoy baseline comparison

Annoy is no longer installed by default. To reproduce the head-to-head numbers below,
install the previous dependency version into the environment first (needs the
compiler/headers noted above):

```bash
ANNOY_COMPILER_ARGS="-DANNOYLIB_MULTITHREADED_BUILD" poetry run python -m pip install --no-cache-dir --no-binary=:all: "annoy==1.17.3"
poetry run python benchmark/embedding_backend/bench_embedding_backend.py
```

If `annoy` is not importable, the script automatically runs the NumPy backend only and
prints a note.

## Results

Representative run captured with `dim=384, k=20, queries=200, seed=1234`; Annoy
`metric=angular, n_trees=10` (matching `nemoguardrails/embeddings/basic.py` before
the change).

| N | backend | build (ms) | search p50 (ms) | search p95 (ms) | recall@1 | recall@20 | mem (MB) |
|---|---|---|---|---|---|---|---|
| 100 | annoy | 1.8980 | 0.0381 | 0.0434 | 1.000 | 1.000 | 0.0 |
| 100 | **numpy** | 0.0033 | 0.0069 | 0.0076 | **1.000** | **1.000** | 0.0 |
| 1 000 | annoy | 18.7925 | 0.0699 | 0.0824 | 0.480 | 0.526 | 0.0 |
| 1 000 | **numpy** | 0.0028 | 0.0215 | 0.0257 | **1.000** | **1.000** | 0.0 |
| 10 000 | annoy | 197.1430 | 0.0787 | 0.1308 | 0.510 | 0.445 | 0.0 |
| 10 000 | **numpy** | 0.0022 | 0.1466 | 0.1962 | **1.000** | **1.000** | 0.0 |
| 100 000 | annoy | 2206.5025 | 0.1196 | 0.1886 | 0.060 | 0.066 | 0.0 |
| 100 000 | **numpy** | 0.0021 | 2.2650 | 2.7545 | **1.000** | **1.000** | 0.0 |

(`build` for NumPy measures the index-construction step after embeddings are already
available and normalized. In this harness that is a contiguous float32 matrix handoff;
the production `BasicEmbeddingsIndex.build()` also converts and normalizes the embeddings
before storing the matrix. `mem` is a best-effort Linux RSS delta; on platforms without
`/proc`, including macOS, the script reports `0.0`. The NumPy matrix size is analytically
`N · dim · 4` bytes ≈ 146 MB at N=100k and is freed when the index is dropped.)

### Conclusion

- **Accuracy.** With the previous `n_trees=10`, Annoy's `recall@1` falls from 1.0 (N=100) to
  **0.06 at N=100k** — it returns the true nearest neighbor only ~6% of the time at scale.
  Exact NumPy is 1.0 everywhere. For a component that decides *which guardrail fires*, exact
  retrieval is the safer default.
- **Latency.** NumPy is faster at N ≤ 1 000, sub-millisecond and competitive at 10k, and only
  slower at 100k — where Annoy's "win" is hollow because its answers are mostly wrong at that
  recall.
- **Build.** NumPy index construction over precomputed vectors is effectively free; Annoy
  grows past 2 s at 100k in this run.
- **Crossover.** Exact search wins up to roughly the 10k-100k range; NeMo Guardrails'
  default indexes sit well below that. For genuinely large indexes, a dedicated opt-in ANN
  provider would be the right path instead of adding Annoy back to the default install.

> Numbers vary with hardware and data distribution; re-run locally to get figures for your
> environment. The qualitative result — exact NumPy keeps or improves quality and is
> competitive-to-faster below the crossover — is robust.

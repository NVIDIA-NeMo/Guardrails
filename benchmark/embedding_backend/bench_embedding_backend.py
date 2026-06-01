# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark harness: Annoy (previous default) vs. exact NumPy cosine search.

This measures the *index* layer in isolation (build / search / recall), feeding
both backends identical pre-computed, L2-normalized vectors so the embedding
model is held constant and only the nearest-neighbor backend varies.

It records the Annoy baseline and reports:
  - retrieval recall@k vs. exact-cosine ground truth
  - search latency p50 / p95
  - build time
  - index memory (best-effort RSS delta)

Run:
    poetry run python benchmark/embedding_backend/bench_embedding_backend.py
    poetry run python benchmark/embedding_backend/bench_embedding_backend.py --sizes 100 1000 --queries 100

Notes on equivalence:
  Annoy "angular" distance is monotonic with cosine similarity for normalized vectors.
  Therefore Annoy's exact ranking would match cosine ranking; recall@k below measures
  Annoy's approximation loss against the exact answer. The NumPy backend computes exact
  cosine, so its recall is 1.0 by construction.
"""

import argparse
import gc
import statistics
import time
from typing import Callable, List, Tuple

import numpy as np

SearchFn = Callable[[np.ndarray, int], List[int]]
BuildResult = Tuple[float, SearchFn, float]

# Parameters matching the previous default in nemoguardrails/embeddings/basic.py
ANNOY_METRIC = "angular"
ANNOY_N_TREES = 10
DEFAULT_DIM = 384  # all-MiniLM-L6-v2
DEFAULT_K = 20  # BasicEmbeddingsIndex.search default max_results


def _rss_mb() -> float:
    """Best-effort resident-set-size in MB (Linux /proc, else 0)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])  # resident pages
        return pages * 4096 / (1024 * 1024)
    except Exception:
        return 0.0


def make_normalized_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    """Deterministic, L2-normalized float32 vectors (clustered, embedding-like)."""
    rng = np.random.default_rng(seed)
    # A handful of cluster centers + noise so neighborhoods are non-trivial
    n_clusters = max(1, min(50, n // 20))
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    assign = rng.integers(0, n_clusters, size=n)
    vecs = centers[assign] + 0.35 * rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


def exact_topk(matrix: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Ground-truth exact cosine top-k indices, shape (Q, k), best-first."""
    sims = queries @ matrix.T  # (Q, N) cosine, inputs already normalized
    # argpartition for top-k, then sort those k by score desc
    part = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[rows, part], axis=1)
    return part[rows, order]


def recall_at_k(approx: List[List[int]], truth: np.ndarray, k: int) -> float:
    """Mean fraction of the exact top-k recovered by the approximate backend."""
    if not approx:
        return 0.0
    total = 0.0
    for i, got in enumerate(approx):
        gold = set(truth[i, :k].tolist())
        if not gold:
            continue
        total += len(gold.intersection(got[:k])) / len(gold)
    return total / len(approx)


def annoy_available() -> bool:
    try:
        import annoy  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def build_annoy(matrix: np.ndarray) -> BuildResult:
    """Build an Annoy index and return build time, search callback, and memory delta."""
    from annoy import AnnoyIndex  # type: ignore

    n, dim = matrix.shape
    gc.collect()
    mem_before = _rss_mb()
    t0 = time.perf_counter()
    index = AnnoyIndex(dim, ANNOY_METRIC)
    for i in range(n):
        index.add_item(i, matrix[i])
    index.build(ANNOY_N_TREES)
    build_s = time.perf_counter() - t0
    mem_delta = _rss_mb() - mem_before

    def search(q: np.ndarray, k: int) -> List[int]:
        idxs, _dists = index.get_nns_by_vector(q, n=k, include_distances=True)
        return idxs

    return build_s, search, mem_delta


def build_numpy(matrix: np.ndarray) -> BuildResult:
    """Build a NumPy exact-search index and return build time, search callback, and memory delta."""
    gc.collect()
    mem_before = _rss_mb()
    t0 = time.perf_counter()
    # Vectors are already normalized; store a contiguous float32 matrix.
    mat = np.ascontiguousarray(matrix, dtype=np.float32)
    build_s = time.perf_counter() - t0
    mem_delta = _rss_mb() - mem_before

    def search(q: np.ndarray, k: int) -> List[int]:
        sims = mat @ q  # (N,) cosine similarity
        kk = min(k, sims.shape[0])
        part = np.argpartition(-sims, kth=kk - 1)[:kk]
        return part[np.argsort(-sims[part])].tolist()

    return build_s, search, mem_delta


# Annoy is no longer a dependency. Install it explicitly to reproduce the
# baseline comparison; otherwise only the NumPy backend runs.
BACKENDS = {"numpy": build_numpy}
if annoy_available():
    BACKENDS = {"annoy": build_annoy, "numpy": build_numpy}


def time_searches(search_fn: SearchFn, queries: np.ndarray, k: int) -> Tuple[List[List[int]], float, float, float]:
    """Run each query individually; return (results, p50_ms, p95_ms, mean_ms)."""
    latencies: List[float] = []
    results: List[List[int]] = []
    for q in queries:
        t0 = time.perf_counter()
        res = search_fn(q, k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        results.append(res)
    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))]
    mean = statistics.fmean(latencies)
    return results, p50, p95, mean


def run(sizes: List[int], dim: int, k: int, n_queries: int, seed: int) -> List[dict]:
    rows: List[dict] = []
    for n in sizes:
        matrix = make_normalized_vectors(n, dim, seed=seed)
        queries = make_normalized_vectors(n_queries, dim, seed=seed + 1)
        truth = exact_topk(matrix, queries, k)

        for name, builder in BACKENDS.items():
            build_s, search_fn, mem_mb = builder(matrix)
            # warm-up
            search_fn(queries[0], k)
            results, p50, p95, mean = time_searches(search_fn, queries, k)
            rec1 = recall_at_k(results, truth, 1)
            reck = recall_at_k(results, truth, k)
            rows.append(
                dict(
                    n=n,
                    backend=name,
                    build_ms=build_s * 1000.0,
                    p50_ms=p50,
                    p95_ms=p95,
                    mean_ms=mean,
                    recall_1=rec1,
                    recall_k=reck,
                    mem_mb=mem_mb,
                )
            )
            print(
                f"  N={n:<7} {name:<6} build={build_s * 1000:11.4f}ms "
                f"p50={p50:9.4f}ms p95={p95:9.4f}ms "
                f"recall@1={rec1:.3f} recall@{k}={reck:.3f} mem~{mem_mb:6.1f}MB"
            )
    return rows


def print_markdown(rows: List[dict], k: int, dim: int = DEFAULT_DIM) -> None:
    print("\n### Results (dim={}, k={})\n".format(dim, k))
    print(f"| N | backend | build (ms) | search p50 (ms) | search p95 (ms) | recall@1 | recall@{k} | mem (MB) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['n']} | {r['backend']} | {r['build_ms']:.4f} | {r['p50_ms']:.4f} | "
            f"{r['p95_ms']:.4f} | {r['recall_1']:.3f} | {r['recall_k']:.3f} | {r['mem_mb']:.1f} |"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 10000, 100000])
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    print(
        f"Embedding backend benchmark | dim={args.dim} k={args.k} queries={args.queries} "
        f"seed={args.seed}\nbackends: {', '.join(BACKENDS)}\n"
        + (
            f"Annoy: metric={ANNOY_METRIC} n_trees={ANNOY_N_TREES}\n"
            if "annoy" in BACKENDS
            else "Annoy not installed -- run `poetry run python -m pip install annoy` to reproduce the baseline comparison.\n"
        )
    )
    rows = run(args.sizes, args.dim, args.k, args.queries, args.seed)
    print_markdown(rows, args.k, args.dim)


if __name__ == "__main__":
    main()

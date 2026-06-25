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

"""Unit tests for the exact NumPy backend of BasicEmbeddingsIndex.

These exercise the index layer directly with injected embeddings (no embedding
model), covering: exact cosine ranking, threshold filtering, and save/load
round-trips.
"""

import re

import numpy as np
import pytest

from nemoguardrails.embeddings.basic import BasicEmbeddingsIndex
from nemoguardrails.embeddings.index import IndexItem
from nemoguardrails.kb import kb as kb_module
from nemoguardrails.kb.kb import KnowledgeBase
from nemoguardrails.rails.llm.config import KnowledgeBaseConfig
from nemoguardrails.utils import compute_hash


def _make_index(embeddings, query_embedding):
    """Build an index from raw embeddings and stub out the query embedding."""
    idx = BasicEmbeddingsIndex()
    idx._items = [IndexItem(text=f"item{i}", meta={"i": i}) for i in range(len(embeddings))]
    idx._embeddings = [list(map(float, e)) for e in embeddings]

    async def _fake_get_embeddings(texts):
        return [list(map(float, query_embedding))]

    idx._get_embeddings = _fake_get_embeddings  # type: ignore[assignment]
    return idx


@pytest.mark.asyncio
async def test_build_normalizes_and_sets_size():
    idx = _make_index([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], [1.0, 0.0, 0.0])
    await idx.build()

    matrix = idx.embeddings_index
    assert matrix.shape == (2, 3)
    assert idx.embedding_size == 3
    # Rows must be unit vectors after build.
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1.0, 1.0], atol=1e-6)


@pytest.mark.asyncio
async def test_search_returns_exact_nearest_order():
    # Query is closest to item0, then item1, then item3.
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.6, 0.6, 0.0],
    ]
    idx = _make_index(embeddings, [1.0, 0.05, 0.0])
    await idx.build()

    results = await idx.search("q", max_results=3)
    assert [r.text for r in results] == ["item0", "item1", "item3"]


@pytest.mark.asyncio
async def test_threshold_filters_by_annoy_parity_score():
    embeddings = [
        [1.0, 0.0, 0.0],  # cos ~1.0 with query
        [0.0, 1.0, 0.0],  # cos ~0.0 with query
    ]
    idx = _make_index(embeddings, [1.0, 0.0, 0.0])
    await idx.build()

    # High threshold keeps only the near-collinear item.
    results = await idx.search("q", max_results=5, threshold=0.9)
    assert [r.text for r in results] == ["item0"]

    # Default (inf) threshold returns everything, best-first.
    results_all = await idx.search("q", max_results=5)
    assert {r.text for r in results_all} == {"item0", "item1"}


@pytest.mark.asyncio
async def test_score_matches_annoy_angular_formula_not_raw_cosine():
    """Lock the scoring contract: score == 1 - sqrt(2 - 2*cos)/2, NOT raw cosine.

    The discriminating item has cosine 0.8 with the query, whose Annoy-parity
    score is 1 - sqrt(2 - 1.6)/2 = 0.6838. A threshold of 0.70 must EXCLUDE it
    (raw-cosine scoring, score=0.8, would wrongly include it); 0.68 must keep it.
    """
    # Unit vectors: query=[1,0]; item=[0.8, 0.6] -> cosine = 0.8.
    idx = _make_index([[0.8, 0.6]], [1.0, 0.0])
    await idx.build()

    parity_score = 1.0 - (2.0 - 2.0 * 0.8) ** 0.5 / 2.0  # ~0.6838
    assert 0.68 < parity_score < 0.70  # guards the test's own constants

    assert await idx.search("q", max_results=1, threshold=0.70) == []
    assert len(await idx.search("q", max_results=1, threshold=0.68)) == 1


@pytest.mark.asyncio
async def test_search_caps_at_index_size():
    idx = _make_index([[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0])
    await idx.build()
    results = await idx.search("q", max_results=20)
    assert len(results) == 2  # never more than the number of items


@pytest.mark.asyncio
async def test_save_load_roundtrip(tmp_path):
    embeddings = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]]
    query = [1.0, 0.05, 0.0]
    idx = _make_index(embeddings, query)
    await idx.build()
    expected = [r.text for r in await idx.search("q", max_results=3)]

    path = str(tmp_path / "index.npy")
    idx.save(path)

    # Fresh index: load the matrix, attach the same items, search again.
    reloaded = _make_index(embeddings, query)
    reloaded._embeddings = []  # ensure search uses the loaded matrix, not rebuilt
    reloaded.load(path)
    assert reloaded.embedding_size == 3
    got = [r.text for r in await reloaded.search("q", max_results=3)]
    assert got == expected


def test_save_load_without_npy_suffix(tmp_path):
    idx = BasicEmbeddingsIndex(index=np.zeros((2, 3), dtype=np.float32))
    path = tmp_path / "index"

    idx.save(str(path))
    assert not path.exists()
    assert path.with_suffix(".npy").exists()

    reloaded = BasicEmbeddingsIndex()
    reloaded.load(str(path))
    assert reloaded.embedding_size == 3


def test_embeddings_index_setter_updates_embedding_size():
    idx = BasicEmbeddingsIndex()
    idx.embeddings_index = np.zeros((2, 4), dtype=np.float32)
    assert idx.embedding_size == 4

    idx.embeddings_index = None
    assert idx.embedding_size == 0


@pytest.mark.parametrize(
    "bad_index",
    [
        np.zeros((4,), dtype=np.float32),
        np.zeros((2, 0), dtype=np.float32),
    ],
)
def test_init_and_setter_reject_invalid_index_shape(bad_index):
    with pytest.raises(ValueError, match="Embedding index is not a valid embeddings index"):
        BasicEmbeddingsIndex(index=bad_index)

    idx = BasicEmbeddingsIndex()
    with pytest.raises(ValueError, match="Embedding index is not a valid embeddings index"):
        idx.embeddings_index = bad_index


@pytest.mark.parametrize(
    "bad_index",
    [
        np.zeros((4,), dtype=np.float32),
        np.zeros((2, 0), dtype=np.float32),
    ],
)
def test_load_rejects_invalid_index_shape(tmp_path, bad_index):
    path = tmp_path / "index.npy"
    np.save(path, bad_index)

    idx = BasicEmbeddingsIndex()
    with pytest.raises(ValueError, match=f"{re.escape(str(path))} is not a valid embeddings index"):
        idx.load(str(path))


@pytest.mark.asyncio
async def test_kb_cache_load_rejects_index_item_count_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(kb_module, "CACHE_FOLDER", str(tmp_path))

    config = KnowledgeBaseConfig()
    kb = KnowledgeBase([], config, lambda _: BasicEmbeddingsIndex())
    kb.chunks = [
        {"title": "First", "body": "alpha"},
        {"title": "Second", "body": "beta"},
    ]

    all_text_items = [f"# {chunk['title']}\n\n{chunk['body'].strip()}" for chunk in kb.chunks]
    hash_prefix = config.embedding_search_provider.parameters.get(
        "embedding_engine", ""
    ) + config.embedding_search_provider.parameters.get("embedding_model", "")
    cache_file = tmp_path / f"{compute_hash(hash_prefix + ''.join(all_text_items))}.npy"
    np.save(cache_file, np.zeros((1, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="Expected 2 rows, got 1"):
        await kb.build()


@pytest.mark.asyncio
async def test_search_before_build_raises():
    idx = _make_index([[1.0, 0.0]], [1.0, 0.0])
    with pytest.raises(ValueError):
        await idx.search("q")


@pytest.mark.asyncio
async def test_build_without_items_raises():
    idx = BasicEmbeddingsIndex()
    with pytest.raises(ValueError):
        await idx.build()

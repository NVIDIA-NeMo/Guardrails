# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The Annoy-backed embeddings index must rank deterministically.

``BasicEmbeddingsIndex.build`` seeds the Annoy RNG so the approximate-NN trees are built
reproducibly. Without the seed, nearest-neighbor ordering can vary between builds (even
for identical embeddings), which makes embedding retrieval — and anything derived from it,
such as retrieved examples baked into LLM prompts — non-reproducible across processes.
"""

import pytest

from nemoguardrails.embeddings.basic import BasicEmbeddingsIndex
from nemoguardrails.embeddings.index import IndexItem

# Fixed embeddings so the test does not depend on a real model. "query" is closest to
# "apricot", then "apple"; "banana"/"cherry" are far.
_VECTORS = {
    "apple": [1.0, 0.05, 0.0],
    "apricot": [0.95, 0.1, 0.0],
    "banana": [0.0, 1.0, 0.0],
    "cherry": [0.0, 0.0, 1.0],
    "query": [0.9, 0.2, 0.0],
}


class _FixedEmbeddingModel:
    async def encode_async(self, texts):
        return [_VECTORS[text] for text in texts]

    def encode(self, texts):
        return [_VECTORS[text] for text in texts]


async def _ranked_texts():
    index = BasicEmbeddingsIndex(embedding_model="fixed", embedding_engine="fixed")
    index._model = _FixedEmbeddingModel()
    await index.add_items([IndexItem(text=text) for text in ("apple", "apricot", "banana", "cherry")])
    await index.build()
    return [item.text for item in await index.search("query", max_results=4)]


@pytest.mark.asyncio
async def test_basic_embeddings_index_build_is_deterministic():
    runs = [await _ranked_texts() for _ in range(3)]

    # Identical ordering across independent builds (the seed makes the Annoy build reproducible).
    assert runs[0] == runs[1] == runs[2]
    # And the ordering is the expected nearest-first ranking.
    assert runs[0] == ["apricot", "apple", "banana", "cherry"]

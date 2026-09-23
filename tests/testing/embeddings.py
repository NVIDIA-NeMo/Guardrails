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

import hashlib
import math
import re
from typing import List, Optional

from nemoguardrails.embeddings.index import EmbeddingsIndex, IndexItem
from nemoguardrails.rails.llm.config import EmbeddingsCacheConfig

TEST_EMBEDDING_SEARCH_PROVIDER = "__test_hash_embedding_search_provider__"


class DeterministicEmbeddingSearchProvider(EmbeddingsIndex):
    """Deterministic test provider that ranks results without default score filtering.

    The default ``search_threshold=float("inf")`` mirrors ``BasicEmbeddingsIndex``:
    results are sorted by similarity and capped by ``max_results``, but low-scoring
    items are not dropped unless a finite threshold is passed explicitly.
    """

    def __init__(self, search_threshold: float = float("inf"), **kwargs):
        self.items: List[IndexItem] = []
        self.embeddings: List[List[float]] = []
        self.search_threshold = search_threshold
        self._cache_config = EmbeddingsCacheConfig()

    @property
    def embedding_size(self):
        return 64

    @property
    def cache_config(self):
        return self._cache_config

    async def _get_embeddings(self, texts: List[str]):
        return [_vectorize(text) for text in texts]

    async def add_item(self, item: IndexItem):
        self.items.append(item)
        self.embeddings.append(_vectorize(item.text))

    async def add_items(self, items: List[IndexItem]):
        self.items.extend(items)
        self.embeddings.extend(_vectorize(item.text) for item in items)

    async def search(self, text: str, max_results: int = 20, threshold: Optional[float] = None) -> List[IndexItem]:
        if threshold is None:
            threshold = self.search_threshold

        text_embedding = _vectorize(text)
        scored_items = []

        for index, item in enumerate(self.items):
            score = _similarity(text, item.text, text_embedding, self.embeddings[index])
            if threshold != float("inf") and not _exact_match(text, item.text):
                score = min(score, threshold - 1.0)
            if threshold == float("inf") or score >= threshold:
                scored_items.append((score, index, item))

        scored_items.sort(key=lambda result: (-result[0], result[1]))
        return [item for _, _, item in scored_items[:max_results]]


def _similarity(text: str, item_text: str, text_embedding: List[float], item_embedding: List[float]) -> float:
    normalized_text = _normalized_text(text)
    normalized_item_text = _normalized_text(item_text)

    if normalized_text and (
        normalized_text == normalized_item_text
        or normalized_text in normalized_item_text
        or normalized_item_text in normalized_text
    ):
        return 1.0

    return sum(a * b for a, b in zip(text_embedding, item_embedding, strict=True))


def _exact_match(text: str, item_text: str) -> bool:
    return text.strip().casefold() == item_text.strip().casefold()


def _normalized_text(text: str) -> str:
    return " ".join(_tokens(text))


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _vectorize(text: str) -> List[float]:
    vector = [0.0] * 64
    tokens = _tokens(text)

    if not tokens:
        vector[0] = 1.0
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        vector[digest[0] % len(vector)] += 1.0 if digest[1] % 2 == 0 else -1.0

    norm = math.sqrt(sum(value * value for value in vector))

    if norm == 0:
        vector[0] = 1.0
        return vector

    return [value / norm for value in vector]

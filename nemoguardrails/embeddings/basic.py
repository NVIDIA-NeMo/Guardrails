# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, cast

from annoy import AnnoyIndex  # type: ignore

from nemoguardrails.embeddings.cache import cache_embeddings
from nemoguardrails.embeddings.index import EmbeddingsIndex, IndexItem
from nemoguardrails.embeddings.providers import EmbeddingModel, init_embedding_model
from nemoguardrails.rails.llm.config import EmbeddingsCacheConfig

log = logging.getLogger(__name__)


class BasicEmbeddingsIndex(EmbeddingsIndex):
    """Basic implementation of an embeddings index.

    It uses the `sentence-transformers/all-MiniLM-L6-v2` model to compute embeddings.
    Annoy is employed for efficient nearest-neighbor search.

    Attributes:
        embedding_model (str): The model for computing embeddings.
        embedding_engine (str): The engine for computing embeddings.
        index (AnnoyIndex): The current embedding index.
        embedding_size (int): The size of the embeddings.
        cache_config (EmbeddingsCacheConfig): The cache configuration.
        embeddings (List[List[float]]): The computed embeddings.
        use_batching: Whether to batch requests when computing the embeddings.
        max_batch_size: The maximum size of a batch.
        max_batch_hold: The maximum time a batch is held before being processed
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_engine: str = "SentenceTransformers",
        embedding_params: Optional[Dict[str, Any]] = None,
        index: Optional[AnnoyIndex] = None,
        cache_config: Optional[Union[EmbeddingsCacheConfig, Dict[str, Any]]] = None,
        search_threshold: float = float("inf"),
        use_batching: bool = False,
        max_batch_size: int = 10,
        max_batch_hold: float = 0.01,
    ):
        """Initialize the BasicEmbeddingsIndex.

        Args:
            embedding_model: The model for computing embeddings.
            embedding_engine: The engine for computing embeddings.
            index: The pre-existing index.
            cache_config: The cache configuration.
            search_threshold: The threshold for filtering search results.
            use_batching: Whether to batch requests when computing the embeddings.
            max_batch_size: The maximum size of a batch.
            max_batch_hold: The maximum time a batch is held before being processed
        """
        self._model: Optional[EmbeddingModel] = None
        self._items: List[IndexItem] = []
        self._embeddings: List[List[float]] = []
        self.embedding_model = embedding_model
        self.embedding_engine = embedding_engine
        self.embedding_params = embedding_params or {}
        self._embedding_size = 0
        self.search_threshold = search_threshold
        if isinstance(cache_config, Dict):
            self._cache_config = EmbeddingsCacheConfig(**cache_config)
        else:
            self._cache_config = cache_config or EmbeddingsCacheConfig()
        self._index = index

        # Data structures for batching embedding requests
        self._req_queue: Dict[int, str] = {}
        self._req_results: Dict[int, List[float]] = {}
        self._req_idx: int = 0
        self._current_batch_finished_event: Optional[asyncio.Event] = None
        self._current_batch_full_event: Optional[asyncio.Event] = None
        self._current_batch_submitted: asyncio.Event = asyncio.Event()

        # Initialize the batching configuration
        self.use_batching = use_batching
        self.max_batch_size = max_batch_size
        self.max_batch_hold = max_batch_hold

    @property
    def embeddings_index(self):
        """Get the current embedding index"""
        return self._index

    @embeddings_index.setter
    def embeddings_index(self, index):
        """Setter to allow replacing the index dynamically."""
        self._index = index

    @property
    def cache_config(self):
        """Get the cache configuration."""
        return self._cache_config

    @property
    def embedding_size(self):
        """Get the size of the embeddings."""
        return self._embedding_size

    @property
    def embeddings(self):
        """Get the computed embeddings."""
        return self._embeddings

    def _init_model(self):
        """Initialize the model used for computing the embeddings."""
        model = self.embedding_model
        engine = self.embedding_engine

        self._model = init_embedding_model(
            embedding_model=model,
            embedding_engine=engine,
            embedding_params=self.embedding_params,
        )

    @cache_embeddings
    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of texts.

        Args:
            texts (List[str]): The list of texts to compute embeddings for.

        Returns:
            List[List[float]]: The computed embeddings.
        """
        if self._model is None:
            self._init_model()

        # self._model can't be None here, or self._init_model() would throw a ValueError
        model: EmbeddingModel = cast(EmbeddingModel, self._model)
        embeddings = await model.encode_async(texts)
        return embeddings

    async def add_item(self, item: IndexItem):
        """Add a single item to the index.

        Args:
            item (IndexItem): The item to add to the index.
        """
        self._items.append(item)

        # If the index is already built, we skip this
        if self._index is None:
            self._embeddings.append((await self._get_embeddings([item.text]))[0])

            # Update the embedding if it was not computed up to this point
            self._embedding_size = len(self._embeddings[0])

    async def add_items(self, items: List[IndexItem]):
        """Add multiple items to the index at once.

        Args:
            items (List[IndexItem]): The list of items to add to the index.
        """
        self._items.extend(items)

        # If the index is already built, we skip this
        if self._index is None:
            self._embeddings.extend(await self._get_embeddings([item.text for item in items]))

            # Update the embedding if it was not computed up to this point
            self._embedding_size = len(self._embeddings[0])

    async def build(self):
        """Builds the Annoy index."""
        self._index = AnnoyIndex(len(self._embeddings[0]), "angular")
        for i in range(len(self._embeddings)):
            self._index.add_item(i, self._embeddings[i])
        self._index.build(10)

    async def _run_batch(self):
        """Runs the current batch of embeddings."""

        # Wait up to `max_batch_hold` time or until `max_batch_size` is reached.
        if self._current_batch_full_event is None or self._current_batch_finished_event is None:
            raise RuntimeError("Batch events not initialized. This should not happen.")

        done, pending = await asyncio.wait(
            [
                asyncio.create_task(asyncio.sleep(self.max_batch_hold)),
                asyncio.create_task(self._current_batch_full_event.wait()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

        # Reset the batch event
        batch_event: asyncio.Event = self._current_batch_finished_event
        self._current_batch_finished_event = None

        # Create the actual batch to be send for computing
        batch = []
        batch_ids = list(self._req_queue.keys())
        for req_id in batch_ids:
            batch.append(self._req_queue[req_id])

        # Empty the queue up to this point
        self._req_queue = {}

        # We allow other batches to start
        self._current_batch_submitted.set()

        # print(f"Running batch of length {len(batch)}")

        # Compute the embeddings
        embeddings = await self._get_embeddings(batch)
        for i in range(len(embeddings)):
            self._req_results[batch_ids[i]] = embeddings[i]

        # Signal that the batch has finished processing
        batch_event.set()

    async def _batch_get_embeddings(self, text: str) -> List[float]:
        # As long as the queue is full, we wait for the next batch
        while len(self._req_queue) >= self.max_batch_size:
            await self._current_batch_submitted.wait()

        req_id = self._req_idx
        self._req_idx += 1
        self._req_queue[req_id] = text

        if self._current_batch_finished_event is None or self._current_batch_full_event is None:
            self._current_batch_finished_event = asyncio.Event()
            self._current_batch_full_event = asyncio.Event()
            self._current_batch_submitted.clear()
            asyncio.ensure_future(self._run_batch())

        # We check if we reached the max batch size
        if len(self._req_queue) >= self.max_batch_size:
            self._current_batch_full_event.set()

        # Wait for the batch to finish
        await self._current_batch_finished_event.wait()

        # Remove the result and return it
        result = self._req_results[req_id]
        del self._req_results[req_id]

        return result

    async def search(self, text: str, max_results: int = 20, threshold: Optional[float] = None) -> List[IndexItem]:
        """Search the closest `max_results` items.

        Args:
            text (str): The text to search for.
            max_results (int, optional): The maximum number of results to return. Defaults to 20.

        Returns:
            List[IndexItem]: The closest items found.
        """
        if threshold is None:
            threshold = self.search_threshold

        if self.use_batching:
            _embedding = await self._batch_get_embeddings(text)
        else:
            _embedding = (await self._get_embeddings([text]))[0]

        if self._index is None:
            raise ValueError("Index is not built yet. Ensure to call `build` before searching.")

        results = self._index.get_nns_by_vector(
            _embedding,
            max_results,
            include_distances=True,
        )

        # In verbose mode, we show detailed info about the scores
        if threshold != float("inf"):
            log_items = []
            for i in range(len(results[0])):
                score = 1 - results[1][i] / 2
                log_items.append((score, self._items[results[0][i]].text))
            log.info("Similarity scores :: %s", str(log_items))

        filtered_results = self._filter_results(results[0], results[1], threshold)

        return [self._items[i] for i in filtered_results]

    @staticmethod
    def _filter_results(indices: List[int], distances: List[float], threshold: float) -> List[int]:
        if threshold == float("inf"):
            return indices
        else:
            return [index for index, distance in zip(indices, distances) if (1 - distance / 2) >= threshold]

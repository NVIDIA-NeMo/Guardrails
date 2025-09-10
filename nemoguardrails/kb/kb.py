# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os
from time import time
from typing import Callable, List, Optional, cast

from nemoguardrails.embeddings.index import EmbeddingsIndex, IndexItem
from nemoguardrails.kb.utils import split_markdown_in_topic_chunks
from nemoguardrails.rails.llm.config import EmbeddingSearchProvider, KnowledgeBaseConfig
from nemoguardrails.utils import compute_hash

log = logging.getLogger(__name__)

CACHE_FOLDER = os.path.join(os.getcwd(), ".cache")


class KnowledgeBase:
    """
    Basic implementation of a knowledge base.

    This class represents a knowledge base that can store and index documents for efficient retrieval.
    It utilizes an embedding search provider to build and search an index for relevant information.

    Parameters:
    - documents (List[str]): A list of documents to initialize the knowledge base.
    - config (KnowledgeBaseConfig): Configuration for the knowledge base.
    - get_embedding_search_provider_instance (Callable[[Optional[EmbeddingSearchProvider]], EmbeddingsIndex]):
      A callable function to get an instance of the embedding search provider.

    Methods:
    - init(): Initializes the knowledge base by splitting documents into topic chunks.
    - build(): Builds the knowledge base index, utilizing the configured embedding search provider.
    - search_relevant_chunks(text: str, max_results: int = 3): Searches the index for the most relevant chunks.

    Attributes:
    - documents (List[str]): The list of documents provided during initialization.
    - chunks (List[dict]): A list of topic chunks extracted from the documents.
    - index (EmbeddingsIndex): The knowledge base index used for searching.
    - config (KnowledgeBaseConfig): Configuration for the knowledge base.

    Example:
    ```python
    # Creating a KnowledgeBase instance
    kb = KnowledgeBase(documents=["Document 1", "Document 2"], config=my_config, get_embedding_search_provider_instance=my_provider)

    # Initializing and building the knowledge base
    kb.init()
    await kb.build()

    # Searching for relevant chunks
    results = await kb.search_relevant_chunks("query text", max_results=5)
    ```

    Note:
    - The knowledge base supports markdown format documents.
    - The index is built using an embedding search provider, and the result is cached for future use.
    """

    def __init__(
        self,
        documents: List[str],
        config: KnowledgeBaseConfig,
        get_embedding_search_provider_instance: Callable[
            [Optional[EmbeddingSearchProvider]], EmbeddingsIndex
        ],
    ):
        self.documents = documents
        self.chunks = []
        self.index = None
        self.config = config
        self._get_embeddings_search_instance = get_embedding_search_provider_instance

    def init(self):
        """Initialize the knowledge base.

        The initial data is loaded from the `$kb_docs` context key. The key is populated when
        the model is loaded. Currently, only markdown format is supported.
        """
        if not self.documents:
            return

        # Start splitting every doc into topic chunks

        for doc in self.documents:
            chunks = split_markdown_in_topic_chunks(doc)
            self.chunks.extend(chunks)

    async def build(self):
        """Builds the knowledge base index."""
        t0 = time()
        index_items = []
        all_text_items = []
        for chunk in self.chunks:
            text = f"# {chunk['title']}\n\n{chunk['body'].strip()}"
            all_text_items.append(text)

            index_items.append(IndexItem(text=text, meta=chunk))

        # Stop if there are no items
        if not index_items:
            return

        # We compute the hash using default hash algorithm
        # As part of the hash, we also include the embedding engine and the model
        # to prevent the cache being used incorrectly when the embedding model changes.
        hash_prefix = self.config.embedding_search_provider.parameters.get(
            "embedding_engine", ""
        ) + self.config.embedding_search_provider.parameters.get("embedding_model", "")

        hash_value = compute_hash(hash_prefix + "".join(all_text_items))
        cache_file = os.path.join(CACHE_FOLDER, f"{hash_value}.ann")
        embedding_size_file = os.path.join(CACHE_FOLDER, f"{hash_value}.esize")

        # If we have already computed this before, we use it
        if (
            self.config.embedding_search_provider.name == "default"
            and os.path.exists(cache_file)
            and os.path.exists(embedding_size_file)
        ):
            from annoy import AnnoyIndex

            from nemoguardrails.embeddings.basic import BasicEmbeddingsIndex

            log.info(cache_file)
            self.index = cast(
                BasicEmbeddingsIndex,
                self._get_embeddings_search_instance(
                    self.config.embedding_search_provider
                ),
            )

            with open(embedding_size_file, "r") as f:
                embedding_size = int(f.read())

            ann_index = AnnoyIndex(embedding_size, "angular")
            ann_index.load(cache_file)

            self.index.embeddings_index = ann_index

            await self.index.add_items(index_items)
        else:
            self.index = self._get_embeddings_search_instance(
                self.config.embedding_search_provider
            )
            await self.index.add_items(index_items)
            await self.index.build()

            # For the default Embedding Search provider, which uses annoy, we also
            # persist the index after it's computed.
            if self.config.embedding_search_provider.name == "default":
                from nemoguardrails.embeddings.basic import BasicEmbeddingsIndex

                # We also save the file for future use
                os.makedirs(CACHE_FOLDER, exist_ok=True)
                basic_index = cast(BasicEmbeddingsIndex, self.index)
                if not basic_index.embeddings_index:
                    raise Exception("Couldn't create basic embeddings index")
                basic_index.embeddings_index.save(cache_file)

                # And, explicitly save the size as we need it when we reload
                with open(embedding_size_file, "w") as f:
                    f.write(str(basic_index.embedding_size))

        log.info(f"Building the Knowledge Base index took {time() - t0} seconds.")

    async def search_relevant_chunks(self, text, max_results: int = 3):
        """Search the index for the most relevant chunks."""
        if self.index is None:
            return []

        results = await self.index.search(text, max_results=max_results)

        # Return the chunks directly
        return [result.meta for result in results]

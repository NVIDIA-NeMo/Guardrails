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
from typing import List

from nemoguardrails import LLMRails
from nemoguardrails.embeddings.index import EmbeddingsIndex, IndexItem


class SimpleEmbeddingSearchProvider(EmbeddingsIndex):
    """A very simple implementation of an embeddings search provider.

    It actually does not use any embeddings, just plain string search through all items.
    """

    @property
    def embedding_size(self):
        return 0

    def __init__(self):
        self.items: List[IndexItem] = []

    async def add_item(self, item: IndexItem):
        """Adds a new item to the index."""
        self.items.append(item)

    async def add_items(self, items: List[IndexItem]):
        """Adds multiple items to the index."""
        self.items.extend(items)

    async def search(self, text: str, max_results: int) -> List[IndexItem]:
        """Searches the index for the closes matches to the provided text."""
        results = []
        for item in self.items:
            if text in item.text:
                results.append(item)

        return results


def init(app: LLMRails):
    app.register_embedding_search_provider("simple", SimpleEmbeddingSearchProvider)

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

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class IndexItem:
    text: str
    meta: Dict = field(default_factory=dict)


class EmbeddingsIndex:
    """The embeddings index is responsible for computing and searching a set of embeddings.

    Attributes:
        embedding_size (int): The size of the embeddings.
        cache_config (EmbeddingsCacheConfig): The cache configuration for the index.

    Methods:
        add_item(item: IndexItem) -> None: Adds a new item to the index.
        add_items(items: List[IndexItem]) -> None: Adds multiple items to the index.
        build() -> None: Builds the index after the items are added. This is optional and might not be needed for all implementations.
        search(text: str, max_results: int) -> List[IndexItem]: Searches the index for the closest matches to the provided text.
    """

    @property
    def embedding_size(self):
        raise NotImplementedError

    @property
    def cache_config(self):
        raise NotImplementedError

    async def _get_embeddings(self, texts: List[str]):
        raise NotImplementedError

    async def add_item(self, item: IndexItem):
        """Adds a new item to the index."""
        raise NotImplementedError()

    async def add_items(self, items: List[IndexItem]):
        """Adds multiple items to the index."""
        raise NotImplementedError()

    async def build(self):
        """Build the index, after the items are added.

        This is optional, might not be needed for all implementations."""
        pass

    async def search(
        self, text: str, max_results: int, threshold: Optional[float]
    ) -> List[IndexItem]:
        """Searches the index for the closest matches to the provided text."""
        raise NotImplementedError()

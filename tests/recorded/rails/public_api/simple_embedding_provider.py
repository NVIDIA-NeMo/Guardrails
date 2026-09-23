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

from typing import List

from nemoguardrails.embeddings.index import EmbeddingsIndex, IndexItem


class SimpleEmbeddingSearchProvider(EmbeddingsIndex):
    @property
    def embedding_size(self):
        return 0

    def __init__(self):
        self.items: List[IndexItem] = []

    async def add_item(self, item: IndexItem):
        self.items.append(item)

    async def add_items(self, items: List[IndexItem]):
        self.items.extend(items)

    async def build(self):
        return None

    async def search(self, text: str, max_results: int, threshold=None):
        normalized = text.lower()
        matches = [item for item in self.items if item.text.lower() in normalized or normalized in item.text.lower()]
        return matches[:max_results] or self.items[:max_results]

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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemoguardrails.kb.kb import KnowledgeBase
from nemoguardrails.rails.llm.config import EmbeddingSearchProvider, KnowledgeBaseConfig


@pytest.mark.asyncio
async def test_cache_key_includes_embedding_provider_parameters():
    config = KnowledgeBaseConfig(
        embedding_search_provider=EmbeddingSearchProvider(
            name="default",
            parameters={
                "embedding_engine": "google",
                "embedding_model": "models/text-embedding-004",
                "dimensionality": 768,
            },
        )
    )
    index = MagicMock()
    index.add_items = AsyncMock()
    index.build = AsyncMock()
    kb = KnowledgeBase(
        documents=["# Heading\n\nBody"],
        config=config,
        get_embedding_search_provider_instance=MagicMock(return_value=index),
    )
    kb.init()

    with patch("nemoguardrails.kb.kb.compute_hash", return_value="cache-key") as mock_hash:
        await kb.build()

    hash_input = mock_hash.call_args.args[0]
    assert '"dimensionality": 768' in hash_input
    assert '"embedding_engine": "google"' in hash_input
    assert '"embedding_model": "models/text-embedding-004"' in hash_input

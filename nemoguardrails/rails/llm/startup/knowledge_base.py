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

"""Knowledge base setup for LLMRails."""

import asyncio
import threading
from typing import Callable, Optional

from nemoguardrails.embeddings.index import EmbeddingsIndex
from nemoguardrails.kb.kb import KnowledgeBase
from nemoguardrails.patch_asyncio import check_sync_call_from_async_loop
from nemoguardrails.rails.llm.config import EmbeddingSearchProvider, RailsConfig
from nemoguardrails.utils import get_or_create_event_loop

__all__ = ["build_knowledge_base_for_docs", "init_knowledge_base"]


async def build_knowledge_base_for_docs(
    config: RailsConfig,
    get_embedding_search_provider_instance: Callable[[Optional[EmbeddingSearchProvider]], EmbeddingsIndex],
) -> Optional[KnowledgeBase]:
    """Build the docs-backed knowledge base configured for LLMRails."""
    if not config.docs:
        return None

    documents = [doc.content for doc in config.docs]
    kb = KnowledgeBase(
        documents=documents,
        config=config.knowledge_base,
        get_embedding_search_provider_instance=get_embedding_search_provider_instance,
    )
    kb.init()
    await kb.build()
    return kb


def init_knowledge_base(rails) -> None:
    """Initialize and register the LLMRails knowledge base."""
    rails.kb = None

    async def _init_kb():
        rails.kb = await build_knowledge_base_for_docs(
            config=rails.config,
            get_embedding_search_provider_instance=rails.embedding_search.get_provider_instance,
        )

    # There are still some edge cases not covered by nest_asyncio.
    # Using a separate thread always for now.
    loop = get_or_create_event_loop()
    if True or check_sync_call_from_async_loop():
        t = threading.Thread(target=asyncio.run, args=(_init_kb(),))
        t.start()
        t.join()
    else:
        loop.run_until_complete(_init_kb())

    rails.runtime.register_action_param("kb", rails.kb)

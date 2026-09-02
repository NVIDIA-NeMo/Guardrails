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

import hashlib
import json
import logging
from typing import Dict, List, Optional

import numpy as np

from nemoguardrails.actions import action
from nemoguardrails.embeddings.providers import init_embedding_model

log = logging.getLogger(__name__)

_detector_cache: Dict[str, "EmbeddingTopicDetector"] = {}


class EmbeddingTopicDetector:
    def __init__(
        self,
        embedding_model: str,
        embedding_engine: str,
        examples: Dict[str, List[str]],
        threshold: float,
        top_k: int,
    ):
        self.threshold = threshold
        self.top_k = top_k
        self.model = init_embedding_model(embedding_model, embedding_engine)
        self.embeddings = {
            cat: [np.array(e) for e in self.model.encode(queries)]
            for cat, queries in examples.items()
            if queries
        }

    async def detect(self, query: str) -> Dict:
        query_emb = np.array((await self.model.encode_async([query]))[0])

        sims = sorted(
            [
                (
                    cat,
                    np.dot(query_emb, emb)
                    / ((np.linalg.norm(query_emb) * np.linalg.norm(emb)) or 1e-10),
                )
                for cat, embs in self.embeddings.items()
                for emb in embs
            ],
            key=lambda x: x[1],
            reverse=True,
        )[: self.top_k]

        scores = {
            cat: float(np.mean([s for c, s in sims if c == cat]) or 0.0)
            for cat in self.embeddings
        }
        max_score = max(scores.values(), default=0.0)

        return {
            "on_topic": max_score >= self.threshold,
            "confidence": max_score,
            "top_category": max(scores, key=scores.get, default=None),
            "category_scores": scores,
        }


async def _check(context: Optional[dict], llm_task_manager, message_key: str) -> dict:
    config = llm_task_manager.config.rails.config.embedding_topic_detector
    examples_hash = hashlib.sha256(
        json.dumps(config["examples"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_key = f"{config['embedding_model']}_{config['embedding_engine']}_{config.get('threshold', 0.75)}_{config.get('top_k', 3)}_{examples_hash}"

    if cache_key not in _detector_cache:
        _detector_cache[cache_key] = EmbeddingTopicDetector(
            config["embedding_model"],
            config["embedding_engine"],
            config["examples"],
            config.get("threshold", 0.75),
            config.get("top_k", 3),
        )

    query = context.get(message_key) if context else None
    if not query:
        return {
            "on_topic": True,
            "confidence": 0.0,
            "top_category": None,
            "category_scores": {},
        }

    return await _detector_cache[cache_key].detect(query)


@action(is_system_action=True)
async def embedding_topic_check(
    context: Optional[dict] = None, llm_task_manager=None
) -> dict:
    return await _check(context, llm_task_manager, "user_message")


@action(is_system_action=True)
async def embedding_topic_check_output(
    context: Optional[dict] = None, llm_task_manager=None
) -> dict:
    return await _check(context, llm_task_manager, "bot_message")

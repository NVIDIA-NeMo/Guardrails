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
from types import SimpleNamespace

from nemoguardrails import RailsConfig
from tests.utils import TestChat

YAML_CONFIG = """
models:
  - type: main
    engine: fake
    model: test

rails:
  config:
    embedding_topic_detector:
      embedding_model: "BAAI/bge-small-en-v1.5"
      embedding_engine: "FastEmbed"
      threshold: 0.5
      top_k: 3
      examples:
        coffee:
          - "how to brew v60"
          - "best light-roast espresso beans"
          - "is soup an espresso type?"

  input:
    flows:
      - embedding topic check
"""

COLANG_CONFIG = """
define bot refuse to respond
  "I'm sorry, I can't respond to that."
"""


def test_off_topic_blocked():
    """Test that off-topic queries are blocked by the embedding detector."""
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_CONFIG
    )
    chat = TestChat(config, llm_completions=[])

    chat >> "Who won the Super Bowl?"
    chat << "I'm sorry, I can't respond to that."


def test_detector_logic():
    """Test the core embedding similarity detection logic."""
    from nemoguardrails.library.embedding_topic_detector.actions import (
        EmbeddingTopicDetector,
    )

    detector = EmbeddingTopicDetector(
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_engine="FastEmbed",
        examples={"coffee": ["how to brew coffee", "best espresso beans"]},
        threshold=0.5,
        top_k=3,
    )

    on_topic = asyncio.run(detector.detect("How do I make espresso?"))
    assert on_topic["on_topic"] is True
    assert on_topic["confidence"] > 0.5
    assert on_topic["top_category"] == "coffee"

    off_topic = asyncio.run(detector.detect("Who won the Super Bowl?"))
    assert off_topic["on_topic"] is False
    assert off_topic["confidence"] < 0.5


def test_empty_query_handling():
    """Test that empty queries are handled gracefully."""
    from nemoguardrails.library.embedding_topic_detector.actions import _check

    llm_task_manager = SimpleNamespace(
        config=SimpleNamespace(
            rails=SimpleNamespace(
                config=SimpleNamespace(
                    embedding_topic_detector={
                        "embedding_model": "BAAI/bge-small-en-v1.5",
                        "embedding_engine": "FastEmbed",
                        "examples": {"coffee": ["espresso"]},
                        "threshold": 0.5,
                        "top_k": 3,
                    }
                )
            )
        )
    )

    # Test with None context
    result = asyncio.run(_check(None, llm_task_manager, "user_message"))
    assert result == {
        "on_topic": True,
        "confidence": 0.0,
        "top_category": None,
        "category_scores": {},
    }

    # Test with empty message in context
    result = asyncio.run(_check({}, llm_task_manager, "user_message"))
    assert result["on_topic"] is True
    assert result["confidence"] == 0.0


def test_output_rail():
    """Test that output rail (bot message checking) works."""
    yaml_with_output = """
models:
  - type: main
    engine: fake
    model: test

rails:
  config:
    embedding_topic_detector:
      embedding_model: "BAAI/bge-small-en-v1.5"
      embedding_engine: "FastEmbed"
      threshold: 0.5
      top_k: 3
      examples:
        coffee:
          - "how to brew coffee"
          - "espresso tips"

  output:
    flows:
      - embedding topic check output
"""

    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=yaml_with_output
    )
    chat = TestChat(config, llm_completions=["Who won the Super Bowl yesterday?"])

    chat >> "Hello"
    chat << "I'm sorry, I can't respond to that."

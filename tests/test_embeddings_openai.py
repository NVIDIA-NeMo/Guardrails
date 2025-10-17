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

import os

import pytest

from nemoguardrails import LLMRails, RailsConfig

try:
    from nemoguardrails.embeddings.providers.openai import OpenAIEmbeddingModel
except ImportError:
    # Ignore this if running in test environment when openai not installed.
    OpenAIEmbeddingModel = None

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")

LIVE_TEST_MODE = os.environ.get("LIVE_TEST")


@pytest.fixture
def app():
    """Load the configuration where we replace FastEmbed with OpenAI."""
    config = RailsConfig.from_path(
        os.path.join(CONFIGS_FOLDER, "with_openai_embeddings")
    )

    return LLMRails(config)


@pytest.mark.skipif(not LIVE_TEST_MODE, reason="Not in live mode.")
def test_custom_llm_registration(app):
    assert isinstance(
        app.llm_generation_actions.flows_index._model, OpenAIEmbeddingModel
    )


@pytest.mark.skipif(not LIVE_TEST_MODE, reason="Not in live mode.")
@pytest.mark.asyncio
async def test_live_query():
    config = RailsConfig.from_path(
        os.path.join(CONFIGS_FOLDER, "with_openai_embeddings")
    )
    app = LLMRails(config)

    result = await app.generate_async(
        messages=[{"role": "user", "content": "tell me what you can do"}]
    )

    assert result == {
        "role": "assistant",
        "content": "I am an AI assistant that helps answer questions.",
    }


@pytest.mark.skipif(not LIVE_TEST_MODE, reason="Not in live mode.")
@pytest.mark.asyncio
def test_live_query(app):
    result = app.generate(
        messages=[{"role": "user", "content": "tell me what you can do"}]
    )

    assert result == {
        "role": "assistant",
        "content": "I am an AI assistant that helps answer questions.",
    }


@pytest.mark.skipif(not LIVE_TEST_MODE, reason="Not in live mode.")
def test_sync_embeddings():
    model = OpenAIEmbeddingModel("text-embedding-3-small")

    result = model.encode(["test"])

    assert len(result[0]) == 1536


@pytest.mark.skipif(not LIVE_TEST_MODE, reason="Not in live mode.")
@pytest.mark.asyncio
async def test_async_embeddings():
    model = OpenAIEmbeddingModel("text-embedding-3-small")

    result = await model.encode_async(["test"])

    assert len(result[0]) == 1536

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

import os

import pytest

from nemoguardrails import LLMRails, RailsConfig

try:
    from nemoguardrails.embeddings.providers.bedrock import BedrockEmbeddingModel
except ImportError:
    # Ignore this if running in a test environment without boto3 installed.
    BedrockEmbeddingModel = None

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")

LIVE_TEST_MODE = os.environ.get("LIVE_TEST")
BEDROCK_AVAILABLE = BedrockEmbeddingModel is not None


@pytest.fixture
def app():
    """Load the configuration where we replace FastEmbed with AWS Bedrock."""
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "with_bedrock_embeddings"))

    return LLMRails(config)


@pytest.mark.skipif(not LIVE_TEST_MODE or not BEDROCK_AVAILABLE, reason="Not in live mode or boto3 not installed.")
def test_custom_llm_registration(app):
    assert isinstance(app.llm_generation_actions.flows_index._model, BedrockEmbeddingModel)


@pytest.mark.skipif(not LIVE_TEST_MODE or not BEDROCK_AVAILABLE, reason="Not in live mode or boto3 not installed.")
def test_sync_embeddings():
    model = BedrockEmbeddingModel("amazon.titan-embed-text-v2:0")

    result = model.encode(["test"])

    assert len(result) == 1
    assert len(result[0]) == 1024


@pytest.mark.skipif(not LIVE_TEST_MODE or not BEDROCK_AVAILABLE, reason="Not in live mode or boto3 not installed.")
@pytest.mark.asyncio
async def test_async_embeddings():
    model = BedrockEmbeddingModel("amazon.titan-embed-text-v2:0")

    result = await model.encode_async(["test"])

    assert len(result) == 1
    assert len(result[0]) == 1024


@pytest.mark.skipif(not LIVE_TEST_MODE or not BEDROCK_AVAILABLE, reason="Not in live mode or boto3 not installed.")
def test_sync_embeddings_cohere_batched():
    model = BedrockEmbeddingModel("cohere.embed-english-v3")

    result = model.encode(["first document", "second document"])

    assert len(result) == 2
    assert len(result[0]) == 1024

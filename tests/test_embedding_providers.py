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
from typing import List

import pytest

from nemoguardrails.embeddings.providers import (
    init_embedding_model,
    register_embedding_provider,
)
from nemoguardrails.embeddings.providers.base import EmbeddingModel

SUPPORTED_PARAMS = {"param1", "param2"}


class MockEmbeddingModel(EmbeddingModel):
    """Mock embedding model for testing purposes.

        Supported embedding models:
            - mock-embedding-small: Embedding size of 128.
            - mock-embedding-large: Embedding size of 256.
        Supported parameters:
            - param1
            - param2

    Args:
        embedding_model (str): The name of the embedding model.

    Attributes:
        model (str): The name of the embedding model.
        embedding_size (int): The size of the embeddings.

    Methods:
        encode: Encode a list of documents into embeddings.
    """

    engine_name = "mock_engine"

    def __init__(self, embedding_model: str, **kwargs):
        self.model = embedding_model
        self.embedding_size_dict = {
            "mock-embedding-small": 128,
            "mock-embedding-large": 256,
        }

        self.embedding_params = kwargs

        if self.model not in self.embedding_size_dict:
            raise ValueError(f"Invalid embedding model: {self.model}")

        supported_params = SUPPORTED_PARAMS

        for param in self.embedding_params:
            if param not in supported_params:
                raise ValueError(f"Unsupported parameter: {param}")

        self.embedding_size = self.embedding_size_dict[self.model]

    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        """Encode a list of documents into embeddings asynchronously.

        Args:
            documents (List[str]): The list of documents to be encoded.

        Returns:
            List[List[float]]: The encoded embeddings.
        """
        return await asyncio.get_running_loop().run_in_executor(None, self.encode, documents)

    def encode(self, documents: List[str]) -> List[List[float]]:
        """Encode a list of documents into embeddings.

        Args:
            documents (List[str]): The list of documents to be encoded.

        Returns:
            List[List[float]]: The encoded embeddings.
        """
        return [[float(i) for i in range(self.embedding_size)] for _ in documents]


register_embedding_provider(MockEmbeddingModel)


def test_init_embedding_model_with_params():
    embedding_model = "mock-embedding-small"
    embedding_engine = "mock_engine"
    supported_param = next(iter(SUPPORTED_PARAMS))
    embedding_params = {supported_param: "value1"}
    model = init_embedding_model(embedding_model, embedding_engine, embedding_params)
    assert isinstance(model, MockEmbeddingModel)
    assert model.model == embedding_model
    assert model.embedding_size == 128
    assert model.engine_name == embedding_engine
    assert model.embedding_params == embedding_params


def test_init_embedding_model_without_params():
    embedding_model = "mock-embedding-large"
    embedding_engine = "mock_engine"
    model = init_embedding_model(embedding_model, embedding_engine)
    assert isinstance(model, MockEmbeddingModel)
    assert model.model == embedding_model
    assert model.embedding_size == 256
    assert model.engine_name == embedding_engine
    assert model.embedding_params == {}


def test_init_embedding_model_with_unsupported_params():
    embedding_model = "mock-embedding-small"
    embedding_engine = "mock_engine"
    embedding_params = {"unsupported_param": "value"}
    with pytest.raises(ValueError, match="Unsupported parameter: unsupported_param"):
        init_embedding_model(embedding_model, embedding_engine, embedding_params)


def test_init_embedding_model_with_invalid_model():
    embedding_model = "invalid_model"
    embedding_engine = "mock_engine"
    embedding_params = {"param1": "value1"}
    with pytest.raises(ValueError, match="Invalid embedding model: invalid_model"):
        init_embedding_model(embedding_model, embedding_engine, embedding_params)


def test_encode_method():
    embedding_model = "mock-embedding-small"
    embedding_engine = "mock_engine"
    model = init_embedding_model(embedding_model, embedding_engine)
    assert isinstance(model, MockEmbeddingModel)
    documents = ["doc1", "doc2", "doc3"]
    embeddings = model.encode(documents)
    assert len(embeddings) == len(documents)
    assert len(embeddings[0]) == model.embedding_size


@pytest.mark.asyncio
async def test_encode_async_method():
    embedding_model = "mock-embedding-large"
    embedding_engine = "mock_engine"
    model = init_embedding_model(embedding_model, embedding_engine)
    assert isinstance(model, MockEmbeddingModel)
    documents = ["doc1", "doc2", "doc3"]
    embeddings = await model.encode_async(documents)
    assert len(embeddings) == len(documents)
    assert len(embeddings[0]) == model.embedding_size

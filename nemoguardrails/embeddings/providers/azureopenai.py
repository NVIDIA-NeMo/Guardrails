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
import os
from typing import List

from .base import EmbeddingModel


def get_executor():
    from . import embeddings_executor

    return embeddings_executor


class AzureEmbeddingModel(EmbeddingModel):
    """Embedding model using Azure OpenAI.

    This class represents an embedding model that utilizes the Azure OpenAI API
    for generating text embeddings.

    Args:
        embedding_model (str): The name of the Azure OpenAI deployment model (e.g., "text-embedding-ada-002").
    """

    engine_name = "AzureOpenAI"

    # Lookup table for model embedding dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        # Add more models and their dimensions here if needed
    }

    def __init__(self, embedding_model: str):
        try:
            from openai import AzureOpenAI  # type: ignore
        except ImportError:
            raise ImportError("Could not import openai, please install it with `pip install openai`.")
        # Set Azure OpenAI API credentials
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # type: ignore
        )

        self.embedding_model = embedding_model
        self.embedding_size = self._get_embedding_dimension()

    def _get_embedding_dimension(self):
        """Retrieve the embedding dimension for the specified model."""
        if self.embedding_model in self.MODEL_DIMENSIONS:
            return self.MODEL_DIMENSIONS[self.embedding_model]
        else:
            embedding_size = len(self.encode(["test"])[0])
            return embedding_size

    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        """Asynchronously encode a list of documents into their corresponding embeddings.

        Args:
            documents (List[str]): The list of documents to be encoded.

        Returns:
            List[List[float]]: The list of embeddings, where each embedding is a list of floats.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_executor(), self.encode, documents)
        return result

    def encode(self, documents: List[str]) -> List[List[float]]:
        """Encode a list of documents into their corresponding embeddings.

        Args:
            documents (List[str]): The list of documents to be encoded.

        Returns:
            List[List[float]]: The list of embeddings, where each embedding is a list of floats.

        Raises:
            RuntimeError: If the API call fails.
        """
        try:
            response = self.client.embeddings.create(model=self.embedding_model, input=documents)
            embeddings = [record.embedding for record in response.data]
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve embeddings: {e}")

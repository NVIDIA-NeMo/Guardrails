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

from .base import EmbeddingModel


class GoogleEmbeddingModel(EmbeddingModel):
    """Embedding model using Gemini API.

    This class is a wrapper for using embedding models powered by Gemini API.

    To use, you must have either:

        1. The ``GOOGLE_API_KEY`` environment variable set with your API key, or
        2. Pass your API key using the api_key kwarg to the genai.Client().

    Args:
        embedding_model (str): The name of the embedding model to be used.
        **kwargs: Additional keyword arguments. Supports:
            - output_dimensionality (int, optional): Desired output dimensions (128-3072 for gemini-embedding-001).
              Recommended values: 768, 1536, or 3072. If not specified, API defaults to 3072.
            - api_key (str, optional): API key for authentication (or use GOOGLE_API_KEY env var).
            - Other arguments passed to genai.Client() constructor.

    Attributes:
        model (str): The name of the embedding model.
        embedding_size (int): The size of the embeddings.
    """

    engine_name = "google"

    def __init__(self, embedding_model: str, **kwargs):
        try:
            from google import genai  # type: ignore[import]

        except ImportError:
            raise ImportError("Could not import google-genai, please install it with `pip install google-genai`.")

        self.model = embedding_model
        self.output_dimensionality = kwargs.pop("output_dimensionality", None)

        self.client = genai.Client(**kwargs)

        embedding_size_dict = {
            "gemini-embedding-001": 3072,
        }

        if self.model in embedding_size_dict:
            self._embedding_size = (
                self.output_dimensionality
                if self.output_dimensionality is not None
                else embedding_size_dict[self.model]
            )
        else:
            self._embedding_size = None

    @property
    def embedding_size(self) -> int:
        if self._embedding_size is None:
            self._embedding_size = len(self.encode(["test"])[0])
        return self._embedding_size

    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        """Encode a list of documents into their corresponding sentence embeddings.

        Args:
            documents (List[str]): The list of documents to be encoded.

        Returns:
            List[List[float]]: The list of sentence embeddings, where each embedding is a list of floats.
        """
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, self.encode, documents)

        return embeddings

    def encode(self, documents: List[str]) -> List[List[float]]:
        """Encode a list of documents into their corresponding sentence embeddings.

        Args:
            documents (List[str]): The list of documents to be encoded.

        Returns:
            List[List[float]]: The list of sentence embeddings, where each embedding is a list of floats.

        Raises:
            RuntimeError: If the embedding request fails.
        """
        try:
            embed_kwargs = {"model": self.model, "contents": documents}
            if self.output_dimensionality is not None:
                embed_kwargs["output_dimensionality"] = self.output_dimensionality

            results = self.client.models.embed_content(**embed_kwargs)
            return [emb.values for emb in results.embeddings]
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve embeddings: {e}") from e

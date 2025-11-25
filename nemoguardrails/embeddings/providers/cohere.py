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
from contextvars import ContextVar
from typing import TYPE_CHECKING, List

from .base import EmbeddingModel

# We set the Cohere async client in an asyncio context variable because we need it
# to be scoped at the asyncio loop level. The client caches it somewhere, and if the loop
# is changed, it will fail.
async_client_var: ContextVar = ContextVar("async_client", default=None)

if TYPE_CHECKING:
    pass


class CohereEmbeddingModel(EmbeddingModel):
    """
    Embedding model using Cohere API.

    To use, you must have either:
        1. The ``COHERE_API_KEY`` environment variable set with your API key, or
        2. Pass your API key using the api_key kwarg to the Cohere constructor.

    Args:
        embedding_model (str): The name of the embedding model.
        input_type (str): The type of input for the embedding model, default is "search_document".
            "search_document", "search_query", "classification", "clustering", "image"

    Attributes:
        model (str): The name of the embedding model.
        embedding_size (int): The size of the embeddings.

    Methods:
        encode: Encode a list of documents into embeddings.
    """

    engine_name = "cohere"

    def __init__(
        self,
        embedding_model: str,
        input_type: str = "search_document",
        **kwargs,
    ):
        try:
            import cohere
        except ImportError:
            raise ImportError("Could not import cohere, please install it with `pip install cohere`.")

        self.model = embedding_model
        self.input_type = input_type
        self.client = cohere.Client(**kwargs)  # type: ignore[reportCallIssue]

        self.embedding_size_dict = {
            "embed-v4.0": 1536,
            "embed-english-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-v3.0": 1024,
            "embed-multilingual-light-v3.0": 384,
        }

        if self.model in self.embedding_size_dict:
            self.embedding_size = self.embedding_size_dict[self.model]
        else:
            # Perform a first encoding to get the embedding size
            self.embedding_size = len(self.encode(["test"])[0])

    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        """Encode a list of documents into embeddings.

        Args:
            documents (List[str]): The list of documents to be encoded.

        Returns:
            List[List[float]]: The encoded embeddings.

        """
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, self.encode, documents)

        # NOTE: The async implementation below has some edge cases because of
        # httpx and async and returns "Event loop is closed." errors. Falling back to
        # a thread-based implementation for now.

        # # We do lazy initialization of the async client to make sure it's on the correct loop
        # async_client = async_client_var.get()
        # if async_client is None:
        #     async_client = AsyncClient()
        #     async_client_var.set(async_client)
        #
        # # Make embedding request to Cohere API
        # embeddings = await async_client.embed(texts=documents, model=self.model, input_type=self.input_type).embeddings

        return embeddings

    def encode(self, documents: List[str]) -> List[List[float]]:
        """Encode a list of documents into embeddings.

        Args:
            documents (List[str]): The list of documents to be encoded.

        Returns:
            List[List[float]]: The encoded embeddings.

        """

        # Make embedding request to Cohere API
        # Since we don't pass embedding_types parameter, the response should be
        # EmbeddingsFloatsEmbedResponse with embeddings as List[List[float]]
        response = self.client.embed(texts=documents, model=self.model, input_type=self.input_type)
        return response.embeddings  # type: ignore[return-value]

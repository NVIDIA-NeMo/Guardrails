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
from typing import List

from .base import EmbeddingModel

# We set the OpenAI async client in an asyncio context variable because we need it
# to be scoped at the asyncio loop level. The client caches it somewhere, and if the loop
# is changed, it will fail.
async_client_var: ContextVar = ContextVar("async_client", default=None)


class OpenAIEmbeddingModel(EmbeddingModel):
    """Embedding model using OpenAI API.

    Args:
        embedding_model (str): The name of the embedding model.

    Attributes:
        model (str): The name of the embedding model.
        embedding_size (int): The size of the embeddings.

    Methods:
        encode: Encode a list of documents into embeddings.
    """

    engine_name = "openai"

    def __init__(
        self,
        embedding_model: str,
        **kwargs,
    ):
        try:
            import openai  # type: ignore
            from openai import OpenAI  # type: ignore
        except ImportError:
            raise ImportError("Could not import openai, please install it with `pip install openai`.")
        if openai.__version__ < "1.0.0":  # type: ignore
            raise RuntimeError(
                "`openai<1.0.0` is no longer supported. Please upgrade using `pip install openai>=1.0.0`."
            )

        self.model = embedding_model
        self.client = OpenAI(**kwargs)

        self.embedding_size_dict = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
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
        #     async_client = AsyncOpenAI()
        #     async_client_var.set(async_client)
        #
        # # Make embedding request to OpenAI API
        # res = await async_client.embeddings.create(input=documents, model=self.model)
        # embeddings = [record.embedding for record in res.data]

        return embeddings

    def encode(self, documents: List[str]) -> List[List[float]]:
        """Encode a list of documents into embeddings.

        Args:
            documents (List[str]): The list of documents to be encoded.

        Returns:
            List[List[float]]: The encoded embeddings.

        """

        # Make embedding request to OpenAI API
        res = self.client.embeddings.create(input=documents, model=self.model)
        embeddings = [record.embedding for record in res.data]

        return embeddings

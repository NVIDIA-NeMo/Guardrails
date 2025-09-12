# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# We set the Azure OpenAI async client in an asyncio context variable because we need it
# to be scoped at the asyncio loop level. The client caches it somewhere, and if the loop
# is changed, it will fail.
async_client_var: ContextVar = ContextVar("azure_async_client", default=None)


class AzureOpenAIEmbeddingModel(EmbeddingModel):
    """Embedding model using Azure OpenAI API.

    Args:
        embedding_model (str): The name of the embedding model deployment.
        azure_endpoint (str): The Azure OpenAI endpoint URL.
        api_version (str): The API version to use (defaults to "2024-02-01").
        **kwargs: Additional arguments passed to AzureOpenAI client.

    Attributes:
        model (str): The name of the embedding model deployment.
        embedding_size (int): The size of the embeddings.

    Methods:
        encode: Encode a list of documents into embeddings.
        encode_async: Asynchronously encode a list of documents into embeddings.
    """

    engine_name = "azure"

    def __init__(
        self,
        embedding_model: str,
        azure_endpoint: str = None,
        api_version: str = "2024-02-01",
        **kwargs,
    ):
        try:
            import openai
            from openai import AzureOpenAI, AsyncAzureOpenAI
        except ImportError:
            raise ImportError(
                "Could not import openai, please install it with "
                "`pip install openai`."
            )
        if openai.__version__ < "1.0.0":
            raise RuntimeError(
                "`openai<1.0.0` is no longer supported. "
                "Please upgrade using `pip install openai>=1.0.0`."
            )

        self.model = embedding_model
        
        # Set default values for Azure OpenAI configuration
        client_kwargs = {
            "api_version": api_version,
            **kwargs
        }
        
        # Add azure_endpoint if provided
        if azure_endpoint:
            client_kwargs["azure_endpoint"] = azure_endpoint
            
        self.client = AzureOpenAI(**client_kwargs)

        # Azure OpenAI supports the same embedding models as OpenAI
        self.embedding_size_dict = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

        # For Azure, the model name might be the deployment name, so we check if we know the size
        if self.model in self.embedding_size_dict:
            self.embedding_size = self.embedding_size_dict[self.model]
        else:
            # Perform a first encoding to get the embedding size
            # This handles custom deployment names
            try:
                self.embedding_size = len(self.encode(["test"])[0])
            except Exception as e:
                # If we can't determine size, default to common size
                self.embedding_size = 1536

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

        return embeddings

    def encode(self, documents: List[str]) -> List[List[float]]:
        """Encode a list of documents into embeddings.

        Args:
            documents (List[str]): The list of documents to be encoded.

        Returns:
            List[List[float]]: The encoded embeddings.

        """

        # Make embedding request to Azure OpenAI API
        res = self.client.embeddings.create(input=documents, model=self.model)
        embeddings = [record.embedding for record in res.data]

        return embeddings
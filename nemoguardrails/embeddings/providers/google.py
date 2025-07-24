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

from typing import List

from .base import EmbeddingModel


class GoogleEmbeddingModel(EmbeddingModel):
    """Embedding model using langchain_google_genai.

    This class is a wrapper for using embedding models powered by Google AI (hosted in the Google Cloud).

    To use, you must have either:

        1. The ``GOOGLE_API_KEY`` environment variable set with your API key, or
        2. Pass your API key using the google_api_key kwarg to the
        GoogleGenerativeAIEmbeddings constructor.

    Args:
        embedding_model (str): The name of the embedding model to be used.

    Attributes:
        model: The name of the model to be called for creating embeddings.
    """

    engine_name = "google"

    def __init__(self, embedding_model: str, **kwargs):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            self.model = embedding_model
            self.document_embedder = GoogleGenerativeAIEmbeddings(
                model=embedding_model, **kwargs
            )

        except ImportError:
            raise ImportError(
                "Could not import langchain_google_genai, please install it with "
                "`pip install langchain-google-genai`."
            )

    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        """Encode a list of documents into their corresponding sentence embeddings.

        Args:
            documents (List[str]): The list of documents to be encoded.

        Returns:
            List[List[float]]: The list of sentence embeddings, where each embedding is a list of floats.
        """

        result = await self.document_embedder.aembed_documents(documents)
        return result

    def encode(self, documents: List[str]) -> List[List[float]]:
        """Encode a list of documents into their corresponding sentence embeddings.

        Args:
            documents (List[str]): The list of documents to be encoded.

        Returns:
            List[List[float]]: The list of sentence embeddings, where each embedding is a list of floats.
        """
        return self.document_embedder.embed_documents(documents)

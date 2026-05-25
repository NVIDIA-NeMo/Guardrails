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

import asyncio
import json
from typing import List, Optional

from .base import EmbeddingModel


class BedrockEmbeddingModel(EmbeddingModel):
    """Embedding model using AWS Bedrock Runtime.

    Supports embedding models served through Amazon Bedrock. Two vendor
    families are currently handled explicitly:

    * Amazon Titan Text Embeddings (``amazon.titan-embed-text-v1``,
      ``amazon.titan-embed-text-v2:0``). The Titan API accepts a single
      input per request, so multi-document calls are dispatched as a
      sequence of ``invoke_model`` requests.
    * Cohere Embed on Bedrock (``cohere.embed-english-v3``,
      ``cohere.embed-multilingual-v3``, ``cohere.embed-english-light-v3``,
      ``cohere.embed-multilingual-light-v3``). Cohere on Bedrock supports
      batch input, so all documents go in a single request.

    Authentication relies on the standard ``boto3`` credential resolution
    chain (environment variables, ``~/.aws/credentials``, IAM instance or
    container role). The target model must be enabled in the AWS account
    and region you target.

    Args:
        embedding_model (str): The Bedrock model id (e.g.
            ``amazon.titan-embed-text-v2:0`` or ``cohere.embed-english-v3``).
        region_name (str, optional): AWS region to target. If omitted,
            ``boto3`` resolves the region from its standard configuration
            chain (``AWS_REGION``, ``AWS_DEFAULT_REGION``, profile, etc.).
        dimensions (int, optional): Output dimensionality. Only forwarded
            to Titan v2 models (``amazon.titan-embed-text-v2:*``) which
            support 256, 512, or 1024.
        normalize (bool, optional): Whether Titan v2 should return
            normalized vectors. Only forwarded to Titan v2 models.
        input_type (str): Input type for Cohere on Bedrock, one of
            ``search_document``, ``search_query``, ``classification``,
            ``clustering``. Defaults to ``search_document``.
        **kwargs: Additional keyword arguments forwarded to
            ``boto3.client("bedrock-runtime", ...)`` (for example
            ``aws_access_key_id``, ``aws_secret_access_key``,
            ``aws_session_token``, ``profile_name``, ``endpoint_url``).

    Attributes:
        model (str): The Bedrock model id.
        embedding_size (int): The size of the embeddings.
    """

    engine_name = "bedrock"

    _embedding_size_dict = {
        "amazon.titan-embed-text-v1": 1536,
        "amazon.titan-embed-text-v2:0": 1024,
        "cohere.embed-english-v3": 1024,
        "cohere.embed-multilingual-v3": 1024,
        "cohere.embed-english-light-v3": 384,
        "cohere.embed-multilingual-light-v3": 384,
    }

    def __init__(
        self,
        embedding_model: str,
        region_name: Optional[str] = None,
        dimensions: Optional[int] = None,
        normalize: Optional[bool] = None,
        input_type: str = "search_document",
        **kwargs,
    ):
        try:
            import boto3  # type: ignore[import]
        except ImportError:
            raise ImportError("Could not import boto3, please install it with `pip install nemoguardrails[bedrock]`.")

        self.model = embedding_model
        self.dimensions = dimensions
        self.normalize = normalize
        self.input_type = input_type
        self._vendor = embedding_model.split(".", 1)[0]

        client_kwargs = dict(kwargs)
        if region_name is not None:
            client_kwargs["region_name"] = region_name
        self.client = boto3.client("bedrock-runtime", **client_kwargs)

        if self.dimensions is not None and self.model.startswith("amazon.titan-embed-text-v2"):
            self._embedding_size: Optional[int] = self.dimensions
        elif self.model in self._embedding_size_dict:
            self._embedding_size = self._embedding_size_dict[self.model]
        else:
            self._embedding_size = None

    @property
    def embedding_size(self) -> int:
        if self._embedding_size is None:
            self._embedding_size = len(self.encode(["test"])[0])
        return self._embedding_size

    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        """Encode a list of documents into embeddings (async).

        Args:
            documents (List[str]): The list of documents to be encoded.

        Returns:
            List[List[float]]: The encoded embeddings.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.encode, documents)

    def encode(self, documents: List[str]) -> List[List[float]]:
        """Encode a list of documents into embeddings.

        Args:
            documents (List[str]): The list of documents to be encoded.

        Returns:
            List[List[float]]: The encoded embeddings.

        Raises:
            RuntimeError: If the Bedrock request fails.
        """
        if not documents:
            return []

        try:
            if self._vendor == "cohere":
                return self._encode_cohere(documents)
            return self._encode_titan(documents)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve embeddings: {e}") from e

    def _encode_titan(self, documents: List[str]) -> List[List[float]]:
        is_titan_v2 = self.model.startswith("amazon.titan-embed-text-v2")
        embeddings: List[List[float]] = []
        for document in documents:
            body: dict = {"inputText": document}
            if is_titan_v2:
                if self.dimensions is not None:
                    body["dimensions"] = self.dimensions
                if self.normalize is not None:
                    body["normalize"] = self.normalize
            response = self.client.invoke_model(
                modelId=self.model,
                body=json.dumps(body).encode("utf-8"),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(response["body"].read())
            embeddings.append(payload["embedding"])
        return embeddings

    def _encode_cohere(self, documents: List[str]) -> List[List[float]]:
        body = {"texts": list(documents), "input_type": self.input_type}
        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(body).encode("utf-8"),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        return payload["embeddings"]

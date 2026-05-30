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

import json
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestCohereEmbeddingModelMocked:
    def test_init_with_known_model(self):
        mock_cohere = MagicMock()
        mock_client = Mock()
        mock_cohere.Client.return_value = mock_client

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            from nemoguardrails.embeddings.providers.cohere import CohereEmbeddingModel

            model = CohereEmbeddingModel("embed-multilingual-v3.0")

            assert model.model == "embed-multilingual-v3.0"
            assert model.embedding_size == 1024
            assert model.input_type == "search_document"
            assert model.client == mock_client
            mock_cohere.Client.assert_called_once()

    def test_init_with_custom_input_type(self):
        mock_cohere = MagicMock()
        mock_client = Mock()
        mock_cohere.Client.return_value = mock_client

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            from nemoguardrails.embeddings.providers.cohere import CohereEmbeddingModel

            model = CohereEmbeddingModel("embed-english-v3.0", input_type="search_query")

            assert model.model == "embed-english-v3.0"
            assert model.embedding_size == 1024
            assert model.input_type == "search_query"

    def test_init_with_unknown_model(self):
        mock_cohere = MagicMock()
        mock_client = Mock()
        mock_cohere.Client.return_value = mock_client

        mock_response = Mock()
        mock_response.embeddings = [[0.1] * 512]
        mock_client.embed.return_value = mock_response

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            from nemoguardrails.embeddings.providers.cohere import CohereEmbeddingModel

            model = CohereEmbeddingModel("custom-unknown-model")

            assert model.model == "custom-unknown-model"
            assert model.embedding_size == 512
            mock_client.embed.assert_called_once_with(
                texts=["test"],
                model="custom-unknown-model",
                input_type="search_document",
            )

    def test_import_error_when_cohere_not_installed(self):
        with patch.dict("sys.modules", {"cohere": None}):
            with pytest.raises(ImportError, match="Could not import cohere"):
                if "nemoguardrails.embeddings.providers.cohere" in sys.modules:
                    del sys.modules["nemoguardrails.embeddings.providers.cohere"]

                from nemoguardrails.embeddings.providers.cohere import (
                    CohereEmbeddingModel,
                )

                CohereEmbeddingModel("embed-v4.0")

    def test_encode_success(self):
        mock_cohere = MagicMock()
        mock_client = Mock()
        mock_cohere.Client.return_value = mock_client

        mock_response = Mock()
        expected_embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
        mock_response.embeddings = expected_embeddings
        mock_client.embed.return_value = mock_response

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            from nemoguardrails.embeddings.providers.cohere import CohereEmbeddingModel

            model = CohereEmbeddingModel("embed-english-light-v3.0")
            documents = ["hello world", "test document"]
            result = model.encode(documents)

            assert result == expected_embeddings
            mock_client.embed.assert_called_with(
                texts=documents,
                model="embed-english-light-v3.0",
                input_type="search_document",
            )

    def test_encode_with_custom_input_type(self):
        mock_cohere = MagicMock()
        mock_client = Mock()
        mock_cohere.Client.return_value = mock_client

        mock_response = Mock()
        expected_embeddings = [[0.1, 0.2]]
        mock_response.embeddings = expected_embeddings
        mock_client.embed.return_value = mock_response

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            from nemoguardrails.embeddings.providers.cohere import CohereEmbeddingModel

            model = CohereEmbeddingModel("embed-v4.0", input_type="classification")
            documents = ["classify this"]
            result = model.encode(documents)

            assert result == expected_embeddings
            mock_client.embed.assert_called_with(texts=documents, model="embed-v4.0", input_type="classification")

    @pytest.mark.asyncio
    async def test_encode_async_success(self):
        mock_cohere = MagicMock()
        mock_client = Mock()
        mock_cohere.Client.return_value = mock_client

        mock_response = Mock()
        expected_embeddings = [[0.1, 0.2, 0.3]]
        mock_response.embeddings = expected_embeddings
        mock_client.embed.return_value = mock_response

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            from nemoguardrails.embeddings.providers.cohere import CohereEmbeddingModel

            model = CohereEmbeddingModel("embed-multilingual-v3.0")
            documents = ["async test"]
            result = await model.encode_async(documents)

            assert result == expected_embeddings
            mock_client.embed.assert_called_once()

    def test_init_with_api_key_kwarg(self):
        mock_cohere = MagicMock()
        mock_client = Mock()
        mock_cohere.Client.return_value = mock_client

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            from nemoguardrails.embeddings.providers.cohere import CohereEmbeddingModel

            model = CohereEmbeddingModel("embed-v4.0", api_key="test-key-123")

            mock_cohere.Client.assert_called_once_with(api_key="test-key-123")

    def test_all_predefined_models(self):
        mock_cohere = MagicMock()
        mock_client = Mock()
        mock_cohere.Client.return_value = mock_client

        models_to_test = {
            "embed-v4.0": 1536,
            "embed-english-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-v3.0": 1024,
            "embed-multilingual-light-v3.0": 384,
        }

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            from nemoguardrails.embeddings.providers.cohere import CohereEmbeddingModel

            for model_name, expected_size in models_to_test.items():
                model = CohereEmbeddingModel(model_name)
                assert model.embedding_size == expected_size
                assert model.model == model_name


class TestOpenAIEmbeddingModelMocked:
    def test_init_with_known_model(self):
        mock_openai = MagicMock()
        mock_openai.__version__ = "1.0.0"
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from nemoguardrails.embeddings.providers.openai import OpenAIEmbeddingModel

            model = OpenAIEmbeddingModel("text-embedding-3-small")

            assert model.model == "text-embedding-3-small"
            assert model.embedding_size == 1536
            assert model.client == mock_client
            mock_openai.OpenAI.assert_called_once()

    def test_init_with_unknown_model(self):
        mock_openai = MagicMock()
        mock_openai.__version__ = "1.0.0"
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        mock_response = Mock()
        mock_record = Mock()
        mock_record.embedding = [0.1] * 2048
        mock_response.data = [mock_record]
        mock_client.embeddings.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from nemoguardrails.embeddings.providers.openai import OpenAIEmbeddingModel

            model = OpenAIEmbeddingModel("custom-unknown-model")

            assert model.model == "custom-unknown-model"
            assert model.embedding_size == 2048
            mock_client.embeddings.create.assert_called_once_with(input=["test"], model="custom-unknown-model")

    def test_import_error_when_openai_not_installed(self):
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="Could not import openai"):
                if "nemoguardrails.embeddings.providers.openai" in sys.modules:
                    del sys.modules["nemoguardrails.embeddings.providers.openai"]

                from nemoguardrails.embeddings.providers.openai import (
                    OpenAIEmbeddingModel,
                )

                OpenAIEmbeddingModel("text-embedding-3-small")

    def test_old_version_error(self):
        mock_openai = MagicMock()
        mock_openai.__version__ = "0.28.0"

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from nemoguardrails.embeddings.providers.openai import OpenAIEmbeddingModel

            with pytest.raises(RuntimeError, match="openai<1.0.0"):
                OpenAIEmbeddingModel("text-embedding-3-small")

    def test_encode_success(self):
        mock_openai = MagicMock()
        mock_openai.__version__ = "1.0.0"
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        mock_response = Mock()
        mock_record1 = Mock()
        expected_embedding1 = [0.1, 0.2, 0.3]
        mock_record1.embedding = expected_embedding1
        mock_record2 = Mock()
        expected_embedding2 = [0.4, 0.5, 0.6]
        mock_record2.embedding = expected_embedding2
        mock_response.data = [mock_record1, mock_record2]
        mock_client.embeddings.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from nemoguardrails.embeddings.providers.openai import OpenAIEmbeddingModel

            model = OpenAIEmbeddingModel("text-embedding-ada-002")
            documents = ["hello world", "test document"]
            result = model.encode(documents)

            assert result == [expected_embedding1, expected_embedding2]
            mock_client.embeddings.create.assert_called_with(input=documents, model="text-embedding-ada-002")

    @pytest.mark.asyncio
    async def test_encode_async_success(self):
        mock_openai = MagicMock()
        mock_openai.__version__ = "1.0.0"
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        mock_response = Mock()
        mock_record = Mock()
        expected_embedding = [0.1, 0.2, 0.3]
        mock_record.embedding = expected_embedding
        mock_response.data = [mock_record]
        mock_client.embeddings.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from nemoguardrails.embeddings.providers.openai import OpenAIEmbeddingModel

            model = OpenAIEmbeddingModel("text-embedding-3-small")
            documents = ["async test"]
            result = await model.encode_async(documents)

            assert result == [expected_embedding]
            mock_client.embeddings.create.assert_called_once()

    def test_init_with_api_key_kwarg(self):
        mock_openai = MagicMock()
        mock_openai.__version__ = "1.0.0"
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from nemoguardrails.embeddings.providers.openai import OpenAIEmbeddingModel

            model = OpenAIEmbeddingModel("text-embedding-3-small", api_key="test-key-123")

            mock_openai.OpenAI.assert_called_once_with(api_key="test-key-123")

    def test_all_predefined_models(self):
        mock_openai = MagicMock()
        mock_openai.__version__ = "1.0.0"
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        models_to_test = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from nemoguardrails.embeddings.providers.openai import OpenAIEmbeddingModel

            for model_name, expected_size in models_to_test.items():
                model = OpenAIEmbeddingModel(model_name)
                assert model.embedding_size == expected_size
                assert model.model == model_name


class TestAzureEmbeddingModelMocked:
    def test_init_with_known_model(self):
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.AzureOpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from nemoguardrails.embeddings.providers.azureopenai import (
                AzureEmbeddingModel,
            )

            model = AzureEmbeddingModel("text-embedding-ada-002")

            assert model.embedding_model == "text-embedding-ada-002"
            assert model.embedding_size == 1536
            assert model.client == mock_client
            mock_openai.AzureOpenAI.assert_called_once()

    def test_init_with_unknown_model(self):
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.AzureOpenAI.return_value = mock_client

        mock_response = Mock()
        mock_record = Mock()
        mock_record.embedding = [0.1] * 2048
        mock_response.data = [mock_record]
        mock_client.embeddings.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from nemoguardrails.embeddings.providers.azureopenai import (
                AzureEmbeddingModel,
            )

            model = AzureEmbeddingModel("custom-unknown-model")

            assert model.embedding_model == "custom-unknown-model"
            assert model.embedding_size == 2048
            mock_client.embeddings.create.assert_called_once_with(model="custom-unknown-model", input=["test"])

    def test_import_error_when_openai_not_installed(self):
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="Could not import openai"):
                if "nemoguardrails.embeddings.providers.azureopenai" in sys.modules:
                    del sys.modules["nemoguardrails.embeddings.providers.azureopenai"]

                from nemoguardrails.embeddings.providers.azureopenai import (
                    AzureEmbeddingModel,
                )

                AzureEmbeddingModel("text-embedding-ada-002")

    def test_encode_success(self):
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.AzureOpenAI.return_value = mock_client

        mock_response = Mock()
        mock_record1 = Mock()
        expected_embedding1 = [0.1, 0.2, 0.3]
        mock_record1.embedding = expected_embedding1
        mock_record2 = Mock()
        expected_embedding2 = [0.4, 0.5, 0.6]
        mock_record2.embedding = expected_embedding2
        mock_response.data = [mock_record1, mock_record2]
        mock_client.embeddings.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from nemoguardrails.embeddings.providers.azureopenai import (
                AzureEmbeddingModel,
            )

            model = AzureEmbeddingModel("text-embedding-ada-002")
            documents = ["hello world", "test document"]
            result = model.encode(documents)

            assert result == [expected_embedding1, expected_embedding2]
            mock_client.embeddings.create.assert_called_with(model="text-embedding-ada-002", input=documents)

    def test_encode_exception_handling(self):
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.AzureOpenAI.return_value = mock_client

        mock_client.embeddings.create.side_effect = Exception("API Error")

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from nemoguardrails.embeddings.providers.azureopenai import (
                AzureEmbeddingModel,
            )

            model = AzureEmbeddingModel("text-embedding-ada-002")
            documents = ["test"]

            with pytest.raises(RuntimeError, match="Failed to retrieve embeddings"):
                model.encode(documents)

    @pytest.mark.asyncio
    async def test_encode_async_success(self):
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.AzureOpenAI.return_value = mock_client

        mock_response = Mock()
        mock_record = Mock()
        expected_embedding = [0.1, 0.2, 0.3]
        mock_record.embedding = expected_embedding
        mock_response.data = [mock_record]
        mock_client.embeddings.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from nemoguardrails.embeddings.providers.azureopenai import (
                AzureEmbeddingModel,
            )

            model = AzureEmbeddingModel("text-embedding-ada-002")
            documents = ["async test"]
            result = await model.encode_async(documents)

            assert result == [expected_embedding]
            mock_client.embeddings.create.assert_called_once()

    def test_all_predefined_models(self):
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.AzureOpenAI.return_value = mock_client

        models_to_test = {
            "text-embedding-ada-002": 1536,
        }

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from nemoguardrails.embeddings.providers.azureopenai import (
                AzureEmbeddingModel,
            )

            for model_name, expected_size in models_to_test.items():
                model = AzureEmbeddingModel(model_name)
                assert model.embedding_size == expected_size
                assert model.embedding_model == model_name

    def test_engine_name_attribute(self):
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.AzureOpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from nemoguardrails.embeddings.providers.azureopenai import (
                AzureEmbeddingModel,
            )

            model = AzureEmbeddingModel("text-embedding-ada-002")

            assert model.engine_name == "AzureOpenAI"

    def test_encode_empty_document_list(self):
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.AzureOpenAI.return_value = mock_client

        mock_response = Mock()
        mock_response.data = []
        mock_client.embeddings.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from nemoguardrails.embeddings.providers.azureopenai import (
                AzureEmbeddingModel,
            )

            model = AzureEmbeddingModel("text-embedding-ada-002")
            result = model.encode([])

            assert result == []
            mock_client.embeddings.create.assert_called_with(model="text-embedding-ada-002", input=[])

    def test_init_with_environment_variables(self):
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.AzureOpenAI.return_value = mock_client

        test_env = {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_API_VERSION": "2023-05-15",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        }

        with patch.dict("sys.modules", {"openai": mock_openai}):
            with patch.dict("os.environ", test_env):
                from nemoguardrails.embeddings.providers.azureopenai import (
                    AzureEmbeddingModel,
                )

                model = AzureEmbeddingModel("text-embedding-ada-002")

                mock_openai.AzureOpenAI.assert_called_once_with(
                    api_key="test-key",
                    api_version="2023-05-15",
                    azure_endpoint="https://test.openai.azure.com/",
                )

    def test_encode_single_document(self):
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.AzureOpenAI.return_value = mock_client

        mock_response = Mock()
        mock_record = Mock()
        expected_embedding = [0.1, 0.2, 0.3]
        mock_record.embedding = expected_embedding
        mock_response.data = [mock_record]
        mock_client.embeddings.create.return_value = mock_response

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from nemoguardrails.embeddings.providers.azureopenai import (
                AzureEmbeddingModel,
            )

            model = AzureEmbeddingModel("text-embedding-ada-002")
            result = model.encode(["single document"])

            assert result == [expected_embedding]
            assert len(result) == 1


class TestGoogleEmbeddingModelMocked:
    def test_init_with_known_model(self):
        mock_genai_module = MagicMock()
        mock_genai = MagicMock()
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        mock_genai_module.genai = mock_genai

        with patch.dict("sys.modules", {"google": mock_genai_module, "google.genai": mock_genai}):
            from nemoguardrails.embeddings.providers.google import GoogleEmbeddingModel

            model = GoogleEmbeddingModel("gemini-embedding-001")

            assert model.model == "gemini-embedding-001"
            assert model.embedding_size == 3072
            assert model.client == mock_client
            assert model.output_dimensionality is None
            mock_genai.Client.assert_called_once()

    def test_init_with_unknown_model(self):
        mock_genai_module = MagicMock()
        mock_genai = MagicMock()
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        mock_genai_module.genai = mock_genai

        mock_response = Mock()
        mock_embedding = Mock()
        mock_embedding.values = [0.1] * 512
        mock_response.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_response

        with patch.dict("sys.modules", {"google": mock_genai_module, "google.genai": mock_genai}):
            from nemoguardrails.embeddings.providers.google import GoogleEmbeddingModel

            model = GoogleEmbeddingModel("custom-unknown-model")

            assert model.model == "custom-unknown-model"
            assert model.embedding_size == 512
            mock_client.models.embed_content.assert_called_once_with(model="custom-unknown-model", contents=["test"])

    def test_import_error_when_google_genai_not_installed(self):
        with patch.dict("sys.modules", {"google": None, "google.genai": None}):
            with pytest.raises(ImportError, match="Could not import google-genai"):
                if "nemoguardrails.embeddings.providers.google" in sys.modules:
                    del sys.modules["nemoguardrails.embeddings.providers.google"]

                from nemoguardrails.embeddings.providers.google import (
                    GoogleEmbeddingModel,
                )

                GoogleEmbeddingModel("gemini-embedding-001")

    def test_encode_success(self):
        mock_genai_module = MagicMock()
        mock_genai = MagicMock()
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        mock_genai_module.genai = mock_genai

        mock_response = Mock()
        mock_embedding1 = Mock()
        expected_embedding1 = [0.1, 0.2, 0.3]
        mock_embedding1.values = expected_embedding1
        mock_embedding2 = Mock()
        expected_embedding2 = [0.4, 0.5, 0.6]
        mock_embedding2.values = expected_embedding2
        mock_response.embeddings = [mock_embedding1, mock_embedding2]
        mock_client.models.embed_content.return_value = mock_response

        with patch.dict("sys.modules", {"google": mock_genai_module, "google.genai": mock_genai}):
            from nemoguardrails.embeddings.providers.google import GoogleEmbeddingModel

            model = GoogleEmbeddingModel("gemini-embedding-001")
            documents = ["hello world", "test document"]
            result = model.encode(documents)

            assert result == [expected_embedding1, expected_embedding2]
            mock_client.models.embed_content.assert_called_with(model="gemini-embedding-001", contents=documents)

    def test_encode_with_output_dimensionality(self):
        mock_genai_module = MagicMock()
        mock_genai = MagicMock()
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        mock_genai_module.genai = mock_genai

        mock_response = Mock()
        mock_embedding = Mock()
        expected_embedding = [0.1] * 1536
        mock_embedding.values = expected_embedding
        mock_response.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_response

        with patch.dict("sys.modules", {"google": mock_genai_module, "google.genai": mock_genai}):
            from nemoguardrails.embeddings.providers.google import GoogleEmbeddingModel

            model = GoogleEmbeddingModel("gemini-embedding-001", output_dimensionality=1536)
            documents = ["test with custom dimensions"]
            result = model.encode(documents)

            assert result == [expected_embedding]
            assert model.embedding_size == 1536
            mock_client.models.embed_content.assert_called_with(
                model="gemini-embedding-001",
                contents=documents,
                output_dimensionality=1536,
            )

    def test_encode_exception_handling(self):
        mock_genai_module = MagicMock()
        mock_genai = MagicMock()
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        mock_genai_module.genai = mock_genai

        mock_client.models.embed_content.side_effect = Exception("API Error")

        with patch.dict("sys.modules", {"google": mock_genai_module, "google.genai": mock_genai}):
            from nemoguardrails.embeddings.providers.google import GoogleEmbeddingModel

            model = GoogleEmbeddingModel("gemini-embedding-001")
            documents = ["test"]

            with pytest.raises(RuntimeError, match="Failed to retrieve embeddings"):
                model.encode(documents)

    @pytest.mark.asyncio
    async def test_encode_async_success(self):
        mock_genai_module = MagicMock()
        mock_genai = MagicMock()
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        mock_genai_module.genai = mock_genai

        mock_response = Mock()
        mock_embedding = Mock()
        expected_embedding = [0.1, 0.2, 0.3]
        mock_embedding.values = expected_embedding
        mock_response.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_response

        with patch.dict("sys.modules", {"google": mock_genai_module, "google.genai": mock_genai}):
            from nemoguardrails.embeddings.providers.google import GoogleEmbeddingModel

            model = GoogleEmbeddingModel("gemini-embedding-001")
            documents = ["async test"]
            result = await model.encode_async(documents)

            assert result == [expected_embedding]
            mock_client.models.embed_content.assert_called_once()

    def test_init_with_api_key_kwarg(self):
        mock_genai_module = MagicMock()
        mock_genai = MagicMock()
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        mock_genai_module.genai = mock_genai

        with patch.dict("sys.modules", {"google": mock_genai_module, "google.genai": mock_genai}):
            from nemoguardrails.embeddings.providers.google import GoogleEmbeddingModel

            model = GoogleEmbeddingModel("gemini-embedding-001", api_key="test-key-123")

            mock_genai.Client.assert_called_once_with(api_key="test-key-123")

    def test_all_predefined_models(self):
        mock_genai_module = MagicMock()
        mock_genai = MagicMock()
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        mock_genai_module.genai = mock_genai

        models_to_test = {
            "gemini-embedding-001": 3072,
        }

        with patch.dict("sys.modules", {"google": mock_genai_module, "google.genai": mock_genai}):
            from nemoguardrails.embeddings.providers.google import GoogleEmbeddingModel

            for model_name, expected_size in models_to_test.items():
                model = GoogleEmbeddingModel(model_name)
                assert model.embedding_size == expected_size
                assert model.model == model_name

    def test_engine_name_attribute(self):
        mock_genai_module = MagicMock()
        mock_genai = MagicMock()
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        mock_genai_module.genai = mock_genai

        with patch.dict("sys.modules", {"google": mock_genai_module, "google.genai": mock_genai}):
            from nemoguardrails.embeddings.providers.google import GoogleEmbeddingModel

            model = GoogleEmbeddingModel("gemini-embedding-001")

            assert model.engine_name == "google"

    def test_init_with_custom_output_dimensionality(self):
        mock_genai_module = MagicMock()
        mock_genai = MagicMock()
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        mock_genai_module.genai = mock_genai

        with patch.dict("sys.modules", {"google": mock_genai_module, "google.genai": mock_genai}):
            from nemoguardrails.embeddings.providers.google import GoogleEmbeddingModel

            model = GoogleEmbeddingModel("gemini-embedding-001", output_dimensionality=3072)

            assert model.model == "gemini-embedding-001"
            assert model.embedding_size == 3072
            assert model.output_dimensionality == 3072

    def test_encode_empty_document_list(self):
        mock_genai_module = MagicMock()
        mock_genai = MagicMock()
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        mock_genai_module.genai = mock_genai

        mock_response = Mock()
        mock_response.embeddings = []
        mock_client.models.embed_content.return_value = mock_response

        with patch.dict("sys.modules", {"google": mock_genai_module, "google.genai": mock_genai}):
            from nemoguardrails.embeddings.providers.google import GoogleEmbeddingModel

            model = GoogleEmbeddingModel("gemini-embedding-001")
            result = model.encode([])

            assert result == []
            mock_client.models.embed_content.assert_called_with(model="gemini-embedding-001", contents=[])

    def test_encode_single_document(self):
        mock_genai_module = MagicMock()
        mock_genai = MagicMock()
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        mock_genai_module.genai = mock_genai

        mock_response = Mock()
        mock_embedding = Mock()
        expected_embedding = [0.1, 0.2, 0.3]
        mock_embedding.values = expected_embedding
        mock_response.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_response

        with patch.dict("sys.modules", {"google": mock_genai_module, "google.genai": mock_genai}):
            from nemoguardrails.embeddings.providers.google import GoogleEmbeddingModel

            model = GoogleEmbeddingModel("gemini-embedding-001")
            result = model.encode(["single document"])

            assert result == [expected_embedding]
            assert len(result) == 1

    def test_lazy_embedding_size_initialization(self):
        mock_genai_module = MagicMock()
        mock_genai = MagicMock()
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        mock_genai_module.genai = mock_genai

        mock_response = Mock()
        mock_embedding = Mock()
        mock_embedding.values = [0.1] * 512
        mock_response.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_response

        with patch.dict("sys.modules", {"google": mock_genai_module, "google.genai": mock_genai}):
            from nemoguardrails.embeddings.providers.google import GoogleEmbeddingModel

            model = GoogleEmbeddingModel("unknown-model")

            assert mock_client.models.embed_content.call_count == 0

            embedding_size = model.embedding_size

            assert embedding_size == 512
            mock_client.models.embed_content.assert_called_once_with(model="unknown-model", contents=["test"])

            _ = model.embedding_size
            assert mock_client.models.embed_content.call_count == 1


def _bedrock_response(payload: dict) -> dict:
    body = Mock()
    body.read.return_value = json.dumps(payload).encode("utf-8")
    return {"body": body}


class TestBedrockEmbeddingModelMocked:
    def _fresh_import(self):
        if "nemoguardrails.embeddings.providers.bedrock" in sys.modules:
            del sys.modules["nemoguardrails.embeddings.providers.bedrock"]
        from nemoguardrails.embeddings.providers.bedrock import BedrockEmbeddingModel

        return BedrockEmbeddingModel

    def test_init_with_known_titan_v1(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            model = BedrockEmbeddingModel("amazon.titan-embed-text-v1")

            assert model.model == "amazon.titan-embed-text-v1"
            assert model.embedding_size == 1536
            assert model.client is mock_client
            assert model.input_type == "search_document"
            assert model.dimensions is None
            assert model.normalize is None
            mock_boto3.client.assert_called_once_with("bedrock-runtime")
            mock_client.invoke_model.assert_not_called()

    def test_init_with_known_titan_v2(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            model = BedrockEmbeddingModel("amazon.titan-embed-text-v2:0")

            assert model.embedding_size == 1024
            mock_client.invoke_model.assert_not_called()

    def test_init_with_titan_v2_custom_dimensions(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            model = BedrockEmbeddingModel("amazon.titan-embed-text-v2:0", dimensions=512)

            assert model.embedding_size == 512
            assert model.dimensions == 512
            mock_client.invoke_model.assert_not_called()

    def test_init_with_known_cohere_model(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            model = BedrockEmbeddingModel("cohere.embed-english-light-v3")

            assert model.embedding_size == 384
            assert model.input_type == "search_document"

    def test_init_with_custom_input_type(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            model = BedrockEmbeddingModel("cohere.embed-english-v3", input_type="search_query")

            assert model.input_type == "search_query"

    def test_import_error_when_boto3_not_installed(self):
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(ImportError, match="pip install nemoguardrails\\[bedrock\\]"):
                BedrockEmbeddingModel = self._fresh_import()
                BedrockEmbeddingModel("amazon.titan-embed-text-v1")

    def test_region_name_passed_to_boto3_client(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            BedrockEmbeddingModel("amazon.titan-embed-text-v1", region_name="us-west-2")

            mock_boto3.client.assert_called_once_with("bedrock-runtime", region_name="us-west-2")

    def test_extra_kwargs_forwarded_to_boto3_client(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            BedrockEmbeddingModel(
                "amazon.titan-embed-text-v1",
                region_name="us-east-1",
                aws_access_key_id="ak",
                aws_secret_access_key="sk",
            )

            mock_boto3.client.assert_called_once_with(
                "bedrock-runtime",
                aws_access_key_id="ak",
                aws_secret_access_key="sk",
                region_name="us-east-1",
            )

    def test_encode_titan_v1_calls_invoke_model_per_document(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client
        mock_client.invoke_model.side_effect = [
            _bedrock_response({"embedding": [0.1] * 1536}),
            _bedrock_response({"embedding": [0.2] * 1536}),
            _bedrock_response({"embedding": [0.3] * 1536}),
        ]

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            model = BedrockEmbeddingModel("amazon.titan-embed-text-v1")
            result = model.encode(["a", "b", "c"])

            assert len(result) == 3
            assert result[0][0] == 0.1
            assert result[2][0] == 0.3
            assert mock_client.invoke_model.call_count == 3
            for call_args, document in zip(mock_client.invoke_model.call_args_list, ["a", "b", "c"]):
                kwargs = call_args.kwargs
                assert kwargs["modelId"] == "amazon.titan-embed-text-v1"
                assert kwargs["contentType"] == "application/json"
                assert kwargs["accept"] == "application/json"
                assert json.loads(kwargs["body"]) == {"inputText": document}

    def test_encode_titan_v2_includes_dimensions_and_normalize(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client
        mock_client.invoke_model.return_value = _bedrock_response({"embedding": [0.1] * 512})

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            model = BedrockEmbeddingModel(
                "amazon.titan-embed-text-v2:0",
                dimensions=512,
                normalize=True,
            )
            result = model.encode(["hello"])

            assert result == [[0.1] * 512]
            body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
            assert body == {"inputText": "hello", "dimensions": 512, "normalize": True}

    def test_encode_cohere_calls_invoke_model_once_batched(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client
        expected = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
        mock_client.invoke_model.return_value = _bedrock_response({"embeddings": expected})

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            model = BedrockEmbeddingModel("cohere.embed-english-v3")
            result = model.encode(["a", "b", "c"])

            assert result == expected
            assert mock_client.invoke_model.call_count == 1
            kwargs = mock_client.invoke_model.call_args.kwargs
            assert kwargs["modelId"] == "cohere.embed-english-v3"
            assert json.loads(kwargs["body"]) == {
                "texts": ["a", "b", "c"],
                "input_type": "search_document",
            }

    def test_encode_cohere_uses_custom_input_type(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client
        mock_client.invoke_model.return_value = _bedrock_response({"embeddings": [[0.1] * 1024]})

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            model = BedrockEmbeddingModel("cohere.embed-english-v3", input_type="search_query")
            model.encode(["query"])

            body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
            assert body["input_type"] == "search_query"

    def test_encode_empty_document_list_returns_empty(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            model = BedrockEmbeddingModel("amazon.titan-embed-text-v1")

            assert model.encode([]) == []
            mock_client.invoke_model.assert_not_called()

    def test_encode_wraps_client_error_in_runtime_error(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client
        mock_client.invoke_model.side_effect = Exception("AccessDeniedException")

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            model = BedrockEmbeddingModel("amazon.titan-embed-text-v1")

            with pytest.raises(RuntimeError, match="Failed to retrieve embeddings"):
                model.encode(["test"])

    @pytest.mark.asyncio
    async def test_encode_async_runs_through_executor(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client
        mock_client.invoke_model.return_value = _bedrock_response({"embedding": [0.5] * 1536})

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            model = BedrockEmbeddingModel("amazon.titan-embed-text-v1")
            result = await model.encode_async(["async test"])

            assert result == [[0.5] * 1536]
            mock_client.invoke_model.assert_called_once()

    def test_lazy_embedding_size_initialization(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client
        mock_client.invoke_model.return_value = _bedrock_response({"embedding": [0.1] * 768})

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            model = BedrockEmbeddingModel("amazon.titan-embed-text-future")

            assert mock_client.invoke_model.call_count == 0

            embedding_size = model.embedding_size

            assert embedding_size == 768
            assert mock_client.invoke_model.call_count == 1

            _ = model.embedding_size
            assert mock_client.invoke_model.call_count == 1

    def test_engine_name_attribute(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            model = BedrockEmbeddingModel("amazon.titan-embed-text-v1")

            assert model.engine_name == "bedrock"

    def test_all_predefined_models(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        expected = {
            "amazon.titan-embed-text-v1": 1536,
            "amazon.titan-embed-text-v2:0": 1024,
            "cohere.embed-english-v3": 1024,
            "cohere.embed-multilingual-v3": 1024,
            "cohere.embed-english-light-v3": 384,
            "cohere.embed-multilingual-light-v3": 384,
        }

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            for model_name, expected_size in expected.items():
                model = BedrockEmbeddingModel(model_name)
                assert model.embedding_size == expected_size
                assert model.model == model_name

    def test_init_with_unsupported_vendor_raises(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            with pytest.raises(ValueError, match="Unsupported Bedrock vendor"):
                BedrockEmbeddingModel("foo.embed-x")

            mock_client.invoke_model.assert_not_called()
            mock_boto3.client.assert_not_called()

    def test_profile_name_uses_session(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_session = Mock()
        mock_session.client.return_value = mock_client
        mock_boto3.Session.return_value = mock_session

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            model = BedrockEmbeddingModel("amazon.titan-embed-text-v1", profile_name="prod")

            mock_boto3.Session.assert_called_once_with(profile_name="prod")
            mock_session.client.assert_called_once_with("bedrock-runtime")
            mock_boto3.client.assert_not_called()
            assert model.client is mock_client

    def test_profile_name_with_region_uses_session(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_session = Mock()
        mock_session.client.return_value = mock_client
        mock_boto3.Session.return_value = mock_session

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            BedrockEmbeddingModel(
                "amazon.titan-embed-text-v1",
                profile_name="prod",
                region_name="eu-central-1",
            )

            mock_session.client.assert_called_once_with("bedrock-runtime", region_name="eu-central-1")

    def test_dimensions_on_titan_v1_warns(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            with pytest.warns(UserWarning, match="'dimensions' is only supported by amazon.titan-embed-text-v2"):
                model = BedrockEmbeddingModel("amazon.titan-embed-text-v1", dimensions=256)

            assert model.dimensions is None
            assert model.embedding_size == 1536

    def test_normalize_on_titan_v1_warns(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            with pytest.warns(UserWarning, match="'normalize' is only supported by amazon.titan-embed-text-v2"):
                model = BedrockEmbeddingModel("amazon.titan-embed-text-v1", normalize=True)

            assert model.normalize is None

    def test_dimensions_on_cohere_warns(self):
        mock_boto3 = MagicMock()
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            BedrockEmbeddingModel = self._fresh_import()

            with pytest.warns(UserWarning, match="'dimensions' is only supported"):
                model = BedrockEmbeddingModel("cohere.embed-english-v3", dimensions=256)

            assert model.dimensions is None
            assert model.embedding_size == 1024

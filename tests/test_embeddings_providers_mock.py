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

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

try:
    import nemoguardrails.embeddings.providers.cohere

    COHERE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    COHERE_AVAILABLE = False


@pytest.mark.skipif(
    not COHERE_AVAILABLE, reason="Cohere provider not available in this branch"
)
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

            model = CohereEmbeddingModel(
                "embed-english-v3.0", input_type="search_query"
            )

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
            mock_client.embed.assert_called_with(
                texts=documents, model="embed-v4.0", input_type="classification"
            )

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
            mock_client.embeddings.create.assert_called_once_with(
                input=["test"], model="custom-unknown-model"
            )

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
            mock_client.embeddings.create.assert_called_with(
                input=documents, model="text-embedding-ada-002"
            )

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

            model = OpenAIEmbeddingModel(
                "text-embedding-3-small", api_key="test-key-123"
            )

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
            mock_client.embeddings.create.assert_called_once_with(
                model="custom-unknown-model", input=["test"]
            )

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
            mock_client.embeddings.create.assert_called_with(
                model="text-embedding-ada-002", input=documents
            )

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
            mock_client.embeddings.create.assert_called_with(
                model="text-embedding-ada-002", input=[]
            )

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

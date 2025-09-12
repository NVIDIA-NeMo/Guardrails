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

import pytest
from unittest.mock import Mock, patch

from nemoguardrails.embeddings.providers import init_embedding_model
from nemoguardrails.embeddings.providers.registry import EmbeddingProviderRegistry


def test_azure_embedding_provider_registration():
    """Test that the Azure embedding provider is properly registered."""
    registry = EmbeddingProviderRegistry()
    
    # Check that azure is in the registry
    assert "azure" in registry.items
    
    # Check that we can get the provider class
    provider_class = registry.get("azure")
    assert provider_class is not None


@patch("openai.AzureOpenAI")
def test_azure_embedding_model_initialization(mock_azure_openai):
    """Test Azure embedding model initialization with different parameters."""
    # Mock the AzureOpenAI client
    mock_client = Mock()
    mock_azure_openai.return_value = mock_client
    
    # Mock the response for size detection
    mock_response = Mock()
    mock_response.data = [Mock()]
    mock_response.data[0].embedding = [0.1] * 1536
    mock_client.embeddings.create.return_value = mock_response
    
    # Test basic initialization
    model = init_embedding_model(
        embedding_model="text-embedding-ada-002",
        embedding_engine="azure",
        embedding_params={
            "azure_endpoint": "https://example.openai.azure.com/",
            "api_key": "test-key"
        }
    )
    
    assert model.model == "text-embedding-ada-002"
    assert model.embedding_size == 1536
    
    # Verify AzureOpenAI was called with correct parameters
    mock_azure_openai.assert_called_with(
        api_version="2024-02-01",
        azure_endpoint="https://example.openai.azure.com/",
        api_key="test-key"
    )


@patch("openai.AzureOpenAI")
def test_azure_embedding_model_custom_deployment(mock_azure_openai):
    """Test Azure embedding model with custom deployment name."""
    # Mock the AzureOpenAI client
    mock_client = Mock()
    mock_azure_openai.return_value = mock_client
    
    # Mock the response for custom deployment
    mock_response = Mock()
    mock_response.data = [Mock()]
    mock_response.data[0].embedding = [0.1] * 3072  # Different size
    mock_client.embeddings.create.return_value = mock_response
    
    # Test with custom deployment name (not in the known models dict)
    model = init_embedding_model(
        embedding_model="my-custom-embedding-deployment",
        embedding_engine="azure",
        embedding_params={
            "azure_endpoint": "https://example.openai.azure.com/",
            "api_key": "test-key",
            "api_version": "2023-12-01-preview"
        }
    )
    
    assert model.model == "my-custom-embedding-deployment"
    assert model.embedding_size == 3072
    
    # Verify the test call was made to determine embedding size
    mock_client.embeddings.create.assert_called()


@patch("openai.AzureOpenAI")
def test_azure_embedding_encode_method(mock_azure_openai):
    """Test the encode method of Azure embedding model."""
    # Mock the AzureOpenAI client and response
    mock_client = Mock()
    mock_azure_openai.return_value = mock_client
    
    # Mock embeddings response for the actual encoding call
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1, 0.2, 0.3]),
        Mock(embedding=[0.4, 0.5, 0.6])
    ]
    mock_client.embeddings.create.return_value = mock_response
    
    model = init_embedding_model(
        embedding_model="text-embedding-encode-test",  # Use unique name to avoid cache
        embedding_engine="azure",
        embedding_params={
            "azure_endpoint": "https://example.openai.azure.com/",
            "api_key": "test-key"
        }
    )
    
    # Test encoding
    documents = ["Hello world", "Test document"]
    embeddings = model.encode(documents)
    
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]
    
    # Verify the API calls were made correctly
    # Since this is an unknown model, there will be 2 calls: 1 for size detection, 1 for actual encoding
    assert mock_client.embeddings.create.call_count == 2
    # The final call should be for our documents
    mock_client.embeddings.create.assert_called_with(
        input=documents, 
        model="text-embedding-encode-test"
    )


def test_azure_embedding_missing_openai_import():
    """Test proper error handling when openai is not installed."""
    # Mock the import to raise ImportError
    with patch.dict('sys.modules', {'openai': None}):
        with pytest.raises(ImportError, match="Could not import openai"):
            from nemoguardrails.embeddings.providers.azure import AzureOpenAIEmbeddingModel
            AzureOpenAIEmbeddingModel(
                embedding_model="test-model",
                azure_endpoint="https://example.openai.azure.com/"
            )


def test_azure_embedding_provider_in_supported_list():
    """Test that Azure is now in the list of supported embedding engines."""
    from nemoguardrails.embeddings.providers import EmbeddingProviderRegistry
    
    registry = EmbeddingProviderRegistry()
    supported_engines = registry.list()
    
    assert "azure" in supported_engines
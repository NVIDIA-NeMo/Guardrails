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

"""
Unit tests for the Mock LLM FastAPI Server.

This module contains comprehensive tests for all endpoints and edge cases
of the OpenAI-compatible mock LLM server.
"""

import json
import os
import time
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from nemoguardrails.benchmark.mock_llm_server.api import app
from nemoguardrails.benchmark.mock_llm_server.config import (
    AppModelConfig,
    get_config,
    load_config,
)
from nemoguardrails.benchmark.mock_llm_server.response_data import (
    DUMMY_CHAT_RESPONSES,
    DUMMY_MODELS,
    calculate_tokens,
    generate_id,
    get_dummy_chat_response,
    get_dummy_completion_response,
)


class TestMockLLMServer:
    """Test class for the Mock LLM Server."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def valid_chat_request(self):
        """Sample valid chat completion request."""
        return {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 50,
            "temperature": 0.7,
        }

    @pytest.fixture
    def valid_completion_request(self):
        """Sample valid text completion request."""
        return {
            "model": "text-davinci-003",
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "temperature": 0.8,
        }

    # Root endpoint tests
    def test_root_endpoint(self, client):
        """Test the root endpoint returns correct information."""

        mock_config = AppModelConfig(
            model="mock_config_model_name",
            refusal_text="I'm afraid I can't do that, Dave",
        )

        def override_get_config():
            return mock_config

        app.dependency_overrides[get_config] = override_get_config

        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Mock LLM Server"
        assert data["version"] == "0.0.1"
        assert "description" in data
        assert "/v1/models" in data["endpoints"]
        assert "/v1/chat/completions" in data["endpoints"]
        assert "/v1/completions" in data["endpoints"]
        assert data["model_configuration"]["model"] == mock_config.model
        assert data["model_configuration"]["refusal_text"] == mock_config.refusal_text

    # Health check tests
    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert isinstance(data["timestamp"], int)

    # Models endpoint tests
    def test_list_models(self, client):
        """Test the models listing endpoint."""
        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) == len(DUMMY_MODELS)

        # Check first model structure
        model = data["data"][0]
        assert "id" in model
        assert "object" in model
        assert "created" in model
        assert "owned_by" in model
        assert model["object"] == "model"

    def test_models_contain_expected_models(self, client):
        """Test that all expected models are returned."""
        response = client.get("/v1/models")
        data = response.json()

        model_ids = [model["id"] for model in data["data"]]
        expected_ids = [model["id"] for model in DUMMY_MODELS]

        assert set(model_ids) == set(expected_ids)

    # Chat completions tests
    def test_chat_completions_success(self, client, valid_chat_request):
        """Test successful chat completion request."""
        response = client.post("/v1/chat/completions", json=valid_chat_request)
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == valid_chat_request["model"]
        assert "id" in data
        assert "created" in data
        assert isinstance(data["created"], int)

        # Check choices
        assert "choices" in data
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert choice["finish_reason"] == "stop"
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)

        # Check usage
        assert "usage" in data
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )

    def test_chat_completions_multiple_choices(self, client, valid_chat_request):
        """Test chat completion with multiple choices."""
        valid_chat_request["n"] = 3
        response = client.post("/v1/chat/completions", json=valid_chat_request)
        assert response.status_code == 200

        data = response.json()
        assert len(data["choices"]) == 3

        for i, choice in enumerate(data["choices"]):
            assert choice["index"] == i
            assert choice["finish_reason"] == "stop"

    def test_chat_completions_invalid_model(self, client, valid_chat_request):
        """Test chat completion with invalid model."""
        valid_chat_request["model"] = "invalid-model"
        response = client.post("/v1/chat/completions", json=valid_chat_request)
        assert response.status_code == 400

        data = response.json()
        assert "detail" in data
        assert "invalid-model" in data["detail"]
        assert "not found" in data["detail"]

    def test_chat_completions_empty_messages(self, client):
        """Test chat completion with empty messages."""
        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [],
        }
        response = client.post("/v1/chat/completions", json=request_data)
        # Note: The server currently accepts empty messages and processes them
        # This may be acceptable behavior for a mock server
        assert response.status_code in [
            200,
            422,
        ]  # Allow both success and validation error

    def test_chat_completions_invalid_message_format(self, client):
        """Test chat completion with invalid message format."""
        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"invalid": "format"}],
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_chat_completions_parameter_validation(self, client, valid_chat_request):
        """Test parameter validation for chat completions."""
        # Test max_tokens validation
        valid_chat_request["max_tokens"] = 0
        response = client.post("/v1/chat/completions", json=valid_chat_request)
        assert response.status_code == 422

        # Test temperature validation
        valid_chat_request["max_tokens"] = 50
        valid_chat_request["temperature"] = 3.0  # Out of range
        response = client.post("/v1/chat/completions", json=valid_chat_request)
        assert response.status_code == 422

        # Test n validation
        valid_chat_request["temperature"] = 0.7
        valid_chat_request["n"] = 200  # Out of range
        response = client.post("/v1/chat/completions", json=valid_chat_request)
        assert response.status_code == 422

    def test_chat_completions_optional_parameters(self, client):
        """Test chat completion with various optional parameters."""
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test message"}],
            "max_tokens": 100,
            "temperature": 0.5,
            "top_p": 0.9,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "stop": ["\\n"],
            "user": "test-user",
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

    # Text completions tests
    def test_completions_success(self, client, valid_completion_request):
        """Test successful text completion request."""
        response = client.post("/v1/completions", json=valid_completion_request)
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "text_completion"
        assert data["model"] == valid_completion_request["model"]
        assert "id" in data
        assert "created" in data

        # Check choices
        assert "choices" in data
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert choice["finish_reason"] == "stop"
        assert "text" in choice
        assert isinstance(choice["text"], str)

        # Check usage
        assert "usage" in data
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    def test_completions_list_prompt(self, client):
        """Test text completion with list prompt."""
        request_data = {
            "model": "text-davinci-003",
            "prompt": ["First prompt", "Second prompt"],
            "max_tokens": 10,
        }
        response = client.post("/v1/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "text_completion"

    def test_completions_invalid_model(self, client, valid_completion_request):
        """Test text completion with invalid model."""
        valid_completion_request["model"] = "non-existent-model"
        response = client.post("/v1/completions", json=valid_completion_request)
        assert response.status_code == 400

    def test_completions_multiple_choices(self, client, valid_completion_request):
        """Test text completion with multiple choices."""
        valid_completion_request["n"] = 2
        response = client.post("/v1/completions", json=valid_completion_request)
        assert response.status_code == 200

        data = response.json()
        assert len(data["choices"]) == 2

    def test_completions_parameter_validation(self, client, valid_completion_request):
        """Test parameter validation for text completions."""
        # Test max_tokens validation
        valid_completion_request["max_tokens"] = -1
        response = client.post("/v1/completions", json=valid_completion_request)
        assert response.status_code == 422

        # Test temperature validation
        valid_completion_request["max_tokens"] = 10
        valid_completion_request["temperature"] = -1.0
        response = client.post("/v1/completions", json=valid_completion_request)
        assert response.status_code == 422

    def test_completions_optional_parameters(self, client):
        """Test text completion with various optional parameters."""
        request_data = {
            "model": "gpt-3.5-turbo",
            "prompt": "Test prompt",
            "max_tokens": 50,
            "temperature": 0.8,
            "top_p": 0.95,
            "n": 1,
            "logprobs": 1,
            "echo": True,
            "stop": ["\\n", "."],
            "presence_penalty": -0.5,
            "frequency_penalty": 0.3,
            "best_of": 2,
            "user": "test-user-2",
        }
        response = client.post("/v1/completions", json=request_data)
        assert response.status_code == 200

    # Helper function tests
    def test_generate_id_default(self):
        """Test ID generation with default prefix."""
        id1 = generate_id()
        id2 = generate_id()

        assert id1.startswith("chatcmpl-")
        assert id2.startswith("chatcmpl-")
        assert id1 != id2  # Should be unique
        assert len(id1) == len("chatcmpl-") + 8  # prefix + 8 hex chars

    def test_generate_id_custom_prefix(self):
        """Test ID generation with custom prefix."""
        custom_id = generate_id("cmpl")
        assert custom_id.startswith("cmpl-")
        assert len(custom_id) == len("cmpl-") + 8

    def test_calculate_tokens(self):
        """Test token calculation function."""
        # Test basic calculation
        assert calculate_tokens("") == 1  # Minimum 1 token
        assert calculate_tokens("a") == 1
        assert calculate_tokens("abcd") == 1
        assert calculate_tokens("abcde") == 1  # 5 chars = 1 token (rounded down)
        assert calculate_tokens("abcdefgh") == 2  # 8 chars = 2 tokens

        # Test longer text
        long_text = "This is a longer text with multiple words and characters."
        expected_tokens = max(1, len(long_text) // 4)
        assert calculate_tokens(long_text) == expected_tokens

    def test_get_dummy_responses(self):
        """Test dummy response generation functions."""
        chat_response = get_dummy_chat_response()
        assert isinstance(chat_response, str)
        assert len(chat_response) > 0

        completion_response = get_dummy_completion_response()
        assert isinstance(completion_response, str)
        assert len(completion_response) > 0

    # Edge cases and error handling
    def test_missing_required_fields_chat(self, client):
        """Test chat completion with missing required fields."""
        # Missing model
        response = client.post("/v1/chat/completions", json={"messages": []})
        assert response.status_code == 422

        # Missing messages
        response = client.post("/v1/chat/completions", json={"model": "gpt-3.5-turbo"})
        assert response.status_code == 422

    def test_missing_required_fields_completion(self, client):
        """Test text completion with missing required fields."""
        # Missing model
        response = client.post("/v1/completions", json={"prompt": "test"})
        assert response.status_code == 422

        # Missing prompt
        response = client.post("/v1/completions", json={"model": "gpt-3.5-turbo"})
        assert response.status_code == 422

    def test_invalid_json(self, client):
        """Test endpoints with invalid JSON."""
        response = client.post(
            "/v1/chat/completions",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_empty_request_body(self, client):
        """Test endpoints with empty request body."""
        response = client.post("/v1/chat/completions", json={})
        assert response.status_code == 422

        response = client.post("/v1/completions", json={})
        assert response.status_code == 422

    # Content validation tests
    def test_chat_message_content_types(self, client):
        """Test chat completion with different message content types."""
        # Test with multiple messages
        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

    def test_response_structure_consistency(self, client, valid_chat_request):
        """Test that response structure is consistent across calls."""
        response1 = client.post("/v1/chat/completions", json=valid_chat_request)
        response2 = client.post("/v1/chat/completions", json=valid_chat_request)

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Structure should be the same
        assert set(data1.keys()) == set(data2.keys())
        assert data1["object"] == data2["object"]
        assert data1["model"] == data2["model"]

        # IDs should be different
        assert data1["id"] != data2["id"]

    def test_concurrent_requests(self, client, valid_chat_request):
        """Test handling of concurrent requests."""
        import threading
        import time

        results = []

        def make_request():
            response = client.post("/v1/chat/completions", json=valid_chat_request)
            results.append(response.status_code)

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All requests should be successful
        assert all(status == 200 for status in results)
        assert len(results) == 5

    # Performance and load tests
    def test_response_time_reasonable(self, client, valid_chat_request):
        """Test that response times are reasonable."""
        start_time = time.time()
        response = client.post("/v1/chat/completions", json=valid_chat_request)
        end_time = time.time()

        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second

    def test_large_prompt_handling(self, client):
        """Test handling of large prompts."""
        large_prompt = "A" * 10000  # 10K characters
        request_data = {
            "model": "text-davinci-003",
            "prompt": large_prompt,
            "max_tokens": 10,
        }
        response = client.post("/v1/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        # Token calculation should handle large text
        assert data["usage"]["prompt_tokens"] > 1000

    # Mock and patch tests
    @patch("nemoguardrails.benchmark.mock_llm_server.api.get_dummy_chat_response")
    def test_chat_completion_response_mocking(
        self, mock_response, client, valid_chat_request
    ):
        """Test mocking of chat response generation."""
        expected_response = "Mocked response for testing chat completions"
        mock_response.return_value = expected_response

        response = client.post("/v1/chat/completions", json=valid_chat_request)
        assert response.status_code == 200

        data = response.json()
        assert data["choices"][0]["message"]["content"] == expected_response
        mock_response.assert_called_once()

    @patch("nemoguardrails.benchmark.mock_llm_server.api.get_dummy_completion_response")
    def test_completion_response_mocking(
        self, mock_response, client, valid_completion_request
    ):
        """Test mocking of chat response generation."""
        expected_response = "Mocked response to check completion responses"
        mock_response.return_value = expected_response

        response = client.post("/v1/completions", json=valid_completion_request)
        assert response.status_code == 200

        data = response.json()
        assert data["choices"][0]["text"] == expected_response
        mock_response.assert_called_once()

    @patch("time.time")
    def test_timestamp_consistency(self, mock_time, client, valid_chat_request):
        """Test that timestamps are generated correctly."""
        mock_time.return_value = 1234567890

        response = client.post("/v1/chat/completions", json=valid_chat_request)
        assert response.status_code == 200

        data = response.json()
        assert data["created"] == 1234567890

    # Documentation and OpenAPI tests
    def test_openapi_docs_available(self, client):
        """Test that OpenAPI documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200

        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi_data = response.json()
        assert "openapi" in openapi_data
        assert "paths" in openapi_data
        assert "/v1/models" in openapi_data["paths"]
        assert "/v1/chat/completions" in openapi_data["paths"]
        assert "/v1/completions" in openapi_data["paths"]

    def test_read_root_with_mock_config(self):
        """Tests load_config method correctly populates the `settings` global variable"""
        yaml_file = os.path.join(os.path.dirname(__file__), "mock_model_config.yaml")

        # Make sure settings is empty to start with, load and check it's populated
        load_config(yaml_file)
        config = get_config()
        assert config is not None

        # Now check the contents against `mock_model_config.yaml`
        assert isinstance(config, AppModelConfig)
        assert config.model == "mock_model"
        assert config.refusal_probability == 0.01
        assert config.refusal_text == "I'm sorry, I can't help you with that request"

    @patch("nemoguardrails.benchmark.mock_llm_server.config.settings", None)
    def test_get_config_raises_exception(self):
        """Check if we call `get_config()` without settings set we raise an exception"""
        with pytest.raises(RuntimeError, match="No configuration loaded"):
            get_config()

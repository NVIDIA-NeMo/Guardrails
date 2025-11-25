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

import time

import pytest
from fastapi.testclient import TestClient

from nemoguardrails.benchmark.mock_llm_server.api import app
from nemoguardrails.benchmark.mock_llm_server.config import ModelSettings, get_settings


def get_test_settings():
    return ModelSettings(
        model="gpt-3.5-turbo",
        unsafe_probability=0.1,
        unsafe_text="I cannot help with that request",
        safe_text="This is a safe response",
        latency_min_seconds=0,
        latency_max_seconds=0,
        latency_mean_seconds=0,
        latency_std_seconds=0,
    )


@pytest.fixture
def client():
    """Create a test client."""
    app.dependency_overrides[get_settings] = get_test_settings
    return TestClient(app)


def test_get_root_endpoint_server_data(client):
    """Test GET / endpoint returns correct server details (not including model info)"""

    model_name = get_test_settings().model

    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Mock LLM Server"
    assert data["version"] == "0.0.1"
    assert data["description"] == f"OpenAI-compatible mock LLM server for model: {model_name}"
    assert data["endpoints"] == [
        "/v1/models",
        "/v1/chat/completions",
        "/v1/completions",
    ]


def test_get_root_endpoint_model_data(client):
    """Test GET / endpoint returns correct model details"""

    response = client.get("/")
    data = response.json()
    model_data = data["model_configuration"]

    expected_model_data = get_test_settings().model_dump()
    assert model_data == expected_model_data


def test_get_health_endpoint(client):
    """Test GET /health endpoint."""
    pre_request_time = int(time.time())
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert isinstance(data["timestamp"], int)
    assert data["timestamp"] >= pre_request_time


def test_get_models_endpoint(client):
    """Test GET /v1/models endpoint."""
    pre_request_time = int(time.time())
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1

    expected_model = get_test_settings().model_dump()
    model = data["data"][0]
    assert model["id"] == expected_model["model"]
    assert model["object"] == "model"
    assert isinstance(model["created"], int)
    assert model["created"] >= pre_request_time
    assert model["owned_by"] == "system"


class TestChatCompletionsEndpoint:
    """Test the /v1/chat/completions endpoint."""

    def test_chat_completions_basic(self, client):
        """Test basic chat completion request."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "gpt-3.5-turbo"
        assert "id" in data
        assert data["id"].startswith("chatcmpl-")

    def test_chat_completions_response_structure(self, client):
        """Test the structure of chat completion response."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test message"}],
        }
        response = client.post("/v1/chat/completions", json=payload)
        data = response.json()

        # Check response structure
        assert "choices" in data
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert "content" in choice["message"]
        assert choice["finish_reason"] == "stop"

    def test_chat_completions_usage(self, client):
        """Test that usage information is included."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test"}],
        }
        response = client.post("/v1/chat/completions", json=payload)
        data = response.json()

        assert "usage" in data
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_chat_completions_multiple_choices(self, client):
        """Test chat completion with n > 1."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "n": 3,
        }
        response = client.post("/v1/chat/completions", json=payload)
        data = response.json()

        assert len(data["choices"]) == 3
        for i, choice in enumerate(data["choices"]):
            assert choice["index"] == i

    def test_chat_completions_multiple_messages(self, client):
        """Test chat completion with multiple messages."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    def test_chat_completions_invalid_model(self, client):
        """Test chat completion with invalid model name."""
        payload = {
            "model": "invalid-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    def test_chat_completions_missing_messages(self, client):
        """Test chat completion without messages field."""
        payload = {
            "model": "gpt-3.5-turbo",
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 422  # Validation error

    def test_chat_completions_empty_messages(self, client):
        """Test chat completion with empty messages list."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [],
        }
        response = client.post("/v1/chat/completions", json=payload)
        # Should either be 422 or 200 depending on validation
        # Let's check it doesn't crash
        assert response.status_code in [200, 422]


class TestCompletionsEndpoint:
    """Test the /v1/completions endpoint."""

    def test_completions_basic(self, client):
        """Test basic completion request."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Once upon a time",
        }
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "text_completion"
        assert data["model"] == "gpt-3.5-turbo"
        assert data["id"].startswith("cmpl-")

    def test_completions_response_structure(self, client):
        """Test the structure of completion response."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Test prompt",
        }
        response = client.post("/v1/completions", json=payload)
        data = response.json()

        assert "choices" in data
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert "text" in choice
        assert isinstance(choice["text"], str)
        assert choice["finish_reason"] == "stop"
        assert choice["logprobs"] is None

    def test_completions_string_prompt(self, client):
        """Test completion with string prompt."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Single string prompt",
        }
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 200

    def test_completions_list_prompt(self, client):
        """Test completion with list of prompts."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": ["Prompt 1", "Prompt 2", "Prompt 3"],
        }
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        # Should still return a response (joined prompts)
        assert "choices" in data

    def test_completions_multiple_choices(self, client):
        """Test completion with n > 1."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Test",
            "n": 5,
        }
        response = client.post("/v1/completions", json=payload)
        data = response.json()

        assert len(data["choices"]) == 5
        for i, choice in enumerate(data["choices"]):
            assert choice["index"] == i

    def test_completions_usage(self, client):
        """Test that usage information is included."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Test prompt",
        }
        response = client.post("/v1/completions", json=payload)
        data = response.json()

        assert "usage" in data
        usage = data["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_completions_invalid_model(self, client):
        """Test completion with invalid model name."""
        payload = {
            "model": "wrong-model",
            "prompt": "Test",
        }
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 400

    def test_completions_missing_prompt(self, client):
        """Test completion without prompt field."""
        payload = {
            "model": "gpt-3.5-turbo",
        }
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 422  # Validation error


class TestMiddleware:
    """Test the HTTP logging middleware."""

    def test_middleware_logs_request(self, client):
        """Test that middleware processes requests."""
        # The middleware should not affect response
        response = client.get("/health")
        assert response.status_code == 200

    def test_middleware_with_post(self, client):
        """Test middleware with POST request."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test"}],
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200


class TestValidateRequestModel:
    """Test the _validate_request_model function."""

    def test_validate_request_model_valid(self, client):
        """Test validation with correct model."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test"}],
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200

    def test_validate_request_model_invalid(self, client):
        """Test validation with incorrect model."""
        payload = {
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "Test"}],
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()
        assert "gpt-3.5-turbo" in response.json()["detail"]


class TestResponseContent:
    """Test that responses contain expected content."""

    def test_chat_response_content_type(self, client):
        """Test that response contains either safe or unsafe text."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test"}],
        }
        response = client.post("/v1/chat/completions", json=payload)
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        # Should be one of the configured responses
        assert content in ["This is a safe response", "I cannot help with that request"]

    def test_completion_response_content_type(self, client):
        """Test that completion response contains expected text."""
        payload = {
            "model": "gpt-3.5-turbo",
            "prompt": "Test",
        }
        response = client.post("/v1/completions", json=payload)
        data = response.json()

        text = data["choices"][0]["text"]
        # Should be one of the configured responses
        assert text in ["This is a safe response", "I cannot help with that request"]
